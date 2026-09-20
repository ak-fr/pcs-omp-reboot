#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <atomic>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>

#include "router/router.hpp"
#include "router_driver.hpp"

/*
 * The Router's test suite.  Every test is one configuration of the same round: producers push
 * a deterministic stream and close, consumers read blocks in place and verify until drained, the service thread runs
 * Router_Progress until quiescent, then rank 0 checks the global accounting.  CHECK, never assert: the
 * build may carry -DNDEBUG.
 */
using namespace mitm;

static int g_rank = 0;
static int g_nodes = 1;
static Atomic<int> g_failed(0);

#define CHECK(cond) do { if (!(cond)) { \
	fprintf(stderr, "CHECK failed on rank %d: %s (%s:%d)\n", g_rank, #cond, __FILE__, __LINE__); \
	g_failed.fetch_add(1); } } while (0)

static const u64 SALT = 0x5a17c0ffee5a17ull;

/* the stream of producer s of node `node` in round r: a encodes (r, node, s, i), b encodes a and the dest */
static inline u64 point_a(int round, int node, int s, u64 i)
{
	return ((u64) round << 56) | ((u64) node << 48) | ((u64) s << 40) | i;
}

static inline u64 point_b(u64 a, int dest)
{
	return murmur64(a) ^ SALT ^ (u64) dest;
}

struct Shared {
	std::vector<u64> sent_to;        /* S * F */
	std::vector<u64> got;            /* per local consumer */
	std::vector<u64> seen;           /* per local consumer: a bitmap over (node, s, i) */
	u64 max_points = 0;
	std::vector<int> t_group;        /* per thread: its Router_group, gathered after Init */
	std::vector<int> t_cpu;          /* per thread: sched_getcpu() after Init, to check pinning gave it one of its own */
};

static u64 points_of(const RouterArgs &a, int s)
{
	if (a.stagger)
		return (a.points * (u64) (s + 1)) / (u64) a.producers;
	return a.points;
}

static void producer_round(Router_thread &rt, const RouterArgs &a, Shared &sh, int s, int round)
{
	int F = a.consumers * g_nodes;
	u64 *sent_to = sh.sent_to.data() + (size_t) s * F;
	u64 i = 0;
	if (a.partial) {
		for (int d = 0; d < F; d++) {
			int n = 1 + (d + s + round) % 7;
			for (int j = 0; j < n; j++, i++) {
				u64 x = point_a(round, g_rank, s, i);
				sent_to[d] += 1;
				Router_Push(x, point_b(x, d), d, rt);
			}
		}
	} else {
		u64 n = points_of(a, s);
		for (; i < n; i++) {
			u64 x = point_a(round, g_rank, s, i);
			int d = a.skew > 0 ? (int) (murmur64(x) % (u64) a.skew) : (int) (murmur64(x) % (u64) F);
			sent_to[d] += 1;
			Router_Push(x, point_b(x, d), d, rt);
		}
	}
	Router_Close(rt);
}

/* one delivered point: from this round, from a real producer, to this consumer's rank, and never seen before */
static void check_point(u64 x, u64 y, int me, int S, Shared &sh, u64 *seen, int round)
{
	int rnd = (int) (x >> 56);
	int node = (int) ((x >> 48) & 0xff);
	int s = (int) ((x >> 40) & 0xff);
	u64 i = x & ((1ull << 40) - 1);
	CHECK(rnd == round);
	CHECK(y == point_b(x, me));
	CHECK(node < g_nodes && s < S && i < sh.max_points);
	u64 bit = ((u64) (node * S + s)) * sh.max_points + i;
	CHECK((seen[bit / 64] & (1ull << (bit % 64))) == 0);
	seen[bit / 64] |= 1ull << (bit % 64);
}

static void consumer_round(Router_thread &rt, const RouterArgs &a, Shared &sh, int r, int round)
{
	int me = g_rank * a.consumers + r;      /* this consumer's global id: the `dest` that reaches it */
	int S = a.producers;
	u64 *seen = sh.seen.data() + (size_t) r * ((g_nodes * S * sh.max_points + 63) / 64);
	u64 got = 0;
	for (;;) {
		u64 x, y;
		if (not Router_Pop(&x, &y, rt)) {
			if (Router_Test_drained(rt))
				break;
			cpu_relax();
			continue;
		}
		check_point(x, y, me, S, sh, seen, round);
		got += 1;
		if (a.slow_recv)
			for (int t = 0; t < 2000; t++)
				cpu_relax();
	}
	sh.got[r] = got;
}

/* rank 0's verdict on the round: every point pushed was popped, and what was sent was received */
static void check_round(const Router_thread &rt, const RouterArgs &a, Shared &sh, int round)
{
	int F = a.consumers * g_nodes;
	int S = a.producers;
	int R = a.consumers;
	u64 st[ROUTER_STATS_SIZE];
	Router_Stats(st, rt);

	std::vector<u64> node_sent(F, 0);
	for (int s = 0; s < S; s++)
		for (int d = 0; d < F; d++)
			node_sent[d] += sh.sent_to[(size_t) s * F + d];
	std::vector<u64> node_got(F, 0);
	u64 local_pushed = 0;
	u64 local_popped = 0;
	for (int d = 0; d < F; d++)
		local_pushed += node_sent[d];
	for (int r = 0; r < R; r++) {
		node_got[g_rank * R + r] = sh.got[r];
		local_popped += sh.got[r];
	}
	CHECK(st[ROUTER_PUSHED] == local_pushed);
	CHECK(st[ROUTER_POPPED] == local_popped);
	CHECK(st[ROUTER_ELAPSED] > 0);
	CHECK(st[ROUTER_BUSY] <= st[ROUTER_ELAPSED]);     /* both the service's own stamps: exact */

	std::vector<u64> tot_sent(F, 0);
	std::vector<u64> tot_got(F, 0);
	u64 tot[ROUTER_STATS_SIZE];
	MPI_Reduce(node_sent.data(), tot_sent.data(), F, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(node_got.data(), tot_got.data(), F, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(st, tot, ROUTER_STATS_SIZE, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	if (g_rank != 0)
		return;
	CHECK(tot[ROUTER_PUSHED] == tot[ROUTER_POPPED]);
	CHECK(tot[ROUTER_SENT] == tot[ROUTER_RECV]);
	CHECK(tot[ROUTER_MSGS_SENT] == tot[ROUTER_MSGS_RECV]);
	CHECK(tot[ROUTER_BYTES_SENT] == tot[ROUTER_BYTES_RECV]);
	for (int d = 0; d < F; d++)
		CHECK(tot_sent[d] == tot_got[d]);
	double send_wait;                        /* share of the round the producers waited for a block */
	double recv_wait;                        /* share the consumers had nothing to pop */
	double busy;                             /* share the service spent in turns that moved something */
	const char *limit = Router_Bottleneck(tot, rt, &send_wait, &recv_wait, &busy);
	if (a.check_limit) {                     /* dawdling consumers over an exhausted pool: the producers wait */
		CHECK(send_wait > 0.5);
		CHECK(strcmp(limit, "consumers") == 0);
	}
	if (a.opts.verbose)
		printf("  round %d: pushed %" PRIu64 " popped %" PRIu64 " local %" PRIu64 " net %" PRIu64
		       " msgs %" PRIu64 " blocks %" PRIu64 " turns %" PRIu64 " (%" PRIu64 " idle) | wait: producers %.1f%%"
		       " consumers %.1f%%, service %.1f%% busy --> %s\n",
		       round, tot[ROUTER_PUSHED], tot[ROUTER_POPPED], tot[ROUTER_LOCAL], tot[ROUTER_SENT],
		       tot[ROUTER_MSGS_SENT], tot[ROUTER_BLOCKS], tot[ROUTER_TURNS], tot[ROUTER_IDLE_TURNS],
		       100. * send_wait, 100. * recv_wait, 100. * busy, limit);
}

/* rank-local, on thread 0 after Init: the plan the Router built for this node -- the group ids and the
 * pinning.  Which consumer a producer's group holds is PCS's own business (it derives it from Router_group
 * across the team, include/pcs/shared.hpp's assign_consumers), not something this test can see through the
 * Router's API, so it only checks the invariant that pairing actually depends on: enough of each role in
 * every group. */
static void check_placement(const Router_thread &rt, const RouterArgs &a, Shared &sh, bool use_colors)
{
	int S = a.producers;
	int R = a.consumers;
	int nt = 1 + S + R;
	int G = Router_num_groups(rt);
	CHECK(G >= 1);
	CHECK(sh.t_group[0] == -1);                 /* the service has no group */
	CHECK(Router_size(rt, ROUTER_PRODUCER, ROUTER_NODE) == S);      /* one role asking about another */
	CHECK(Router_size(rt, ROUTER_CONSUMER, ROUTER_NODE) == R);
	CHECK(Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL) == g_nodes);
	std::vector<int> gs(G, 0);
	std::vector<int> gr(G, 0);
	for (int t = 1; t < nt; t++) {
		CHECK(sh.t_group[t] >= 0 && sh.t_group[t] < G);
		if (sh.t_group[t] < 0 || sh.t_group[t] >= G)
			continue;
		if (t <= R)
			gr[sh.t_group[t]] += 1;
		else
			gs[sh.t_group[t]] += 1;
	}
	if (a.opts.pin)
		for (int t = 0; t < nt; t++)
			for (int u = t + 1; u < nt; u++)
				CHECK(sh.t_cpu[t] != sh.t_cpu[u]);   /* every thread on a CPU of its own */
	if (use_colors) {
		CHECK(G == ((S >= 2 && R >= 2) ? 2 : 1));
	} else if (S >= G && R >= G) {
		for (int g = 0; g < G; g++) {               /* enough of each role for one of each per group */
			CHECK(gs[g] >= 1);
			CHECK(gr[g] >= 1);
		}
	}
}

/* one configuration, start to finish: a team, its Router, `rounds` rounds; everything dies with the region */
static void run_config(const RouterArgs &a, const char *name, bool use_colors = false)
{
	if (g_rank == 0 && a.opts.verbose)
		printf("== %s (%d producers, %d consumers, %" PRIu64 " points, %d rounds)\n", name,
		       a.producers, a.consumers, a.points, a.rounds);
	int S = a.producers;
	int R = a.consumers;
	int nt = 1 + S + R;
	Shared sh;
	sh.max_points = a.partial ? (u64) 7 * R * g_nodes : (a.points > 0 ? a.points : 1);
	sh.sent_to.assign((size_t) S * R * g_nodes, 0);
	sh.got.assign(R, 0);
	sh.seen.assign((size_t) R * ((g_nodes * S * sh.max_points + 63) / 64), 0);
	sh.t_group.assign(nt, 0);
	sh.t_cpu.assign(nt, 0);

	#pragma omp parallel num_threads(nt)
	{
		int tid = omp_get_thread_num();
		int role = (tid == 0) ? ROUTER_SERVICE : (tid <= R ? ROUTER_CONSUMER : ROUTER_PRODUCER);
		int li = (role == ROUTER_CONSUMER) ? (tid - 1) : (tid - 1 - R);   /* the caller's color, in colors mode */
		int group = ROUTER_GROUP_AUTO;
		if (use_colors && role != ROUTER_SERVICE)
			group = (S >= 2 && R >= 2) ? (li % 2) : 0;
		Router_thread rt = Router_Init(role, group, MPI_COMM_WORLD, 42, &a.opts);
		sh.t_group[tid] = Router_group(rt);
		sh.t_cpu[tid] = sched_getcpu();
		int per_node = (role == ROUTER_SERVICE) ? 1 : (role == ROUTER_CONSUMER ? R : S);
		int mine = (role == ROUTER_SERVICE) ? 0 : li;                 /* what Router_rank must answer */
		CHECK(Router_size(rt, role, ROUTER_NODE) == per_node);
		CHECK(Router_size(rt, role, ROUTER_GLOBAL) == per_node * g_nodes);
		CHECK(Router_rank(rt, ROUTER_NODE) == mine);
		CHECK(Router_rank(rt, ROUTER_GLOBAL) == g_rank * per_node + mine);
		#pragma omp barrier
		if (tid == 0)
			check_placement(rt, a, sh, use_colors);
		#pragma omp barrier
		for (int round = 0; round < a.rounds; round++) {
			if (tid == 0) {
				for (size_t k = 0; k < sh.sent_to.size(); k++)
					sh.sent_to[k] = 0;
				for (size_t k = 0; k < sh.seen.size(); k++)
					sh.seen[k] = 0;
			}
			#pragma omp barrier
			if (tid == 0) {
				double t0 = wtime();
				while (not Router_Test_quiescent(rt)) {
					Router_Progress(rt);
					if (wtime() - t0 > 10) {
						Router_Dump(stderr, rt);
						MPI_Abort(MPI_COMM_WORLD, 2);
					}
				}
			} else if (role == ROUTER_CONSUMER) {
				consumer_round(rt, a, sh, tid - 1, round);
			} else {
				producer_round(rt, a, sh, tid - 1 - R, round);
			}
			#pragma omp barrier               /* the test's own: check_round reads what the workers wrote to sh */
			if (tid == 0)
				check_round(rt, a, sh, round);
			Router_Reset(rt);
		}
	}
}

/*
 * The placer on fabricated machines.  RouterPlacement is pure computation over an affinity mask and the
 * topology hwloc reports -- it pins nothing, connect() does -- so a synthetic machine and a mask of its CPUs
 * exercise shapes no test host has: two sockets, four, 32 cache domains, a team that fills every CPU.
 */
struct PlacementCase {
	const char *machine;             /* the machine, as HWLOC_SYNTHETIC spells it */
	int n_numa;                      /* its NUMA nodes */
	int n_dom;                       /* its cache domains */
	int n_core;                      /* its cores */
	int n_cpu;                       /* its PUs, which are the mask the placer is given */
	int producers;                     /* the team, besides the one service thread */
	int consumers;
	int group_size;                  /* cores per group at most */
};

static const PlacementCase PLACEMENT_CASES[] = {
	{"node:2 l3:16 core:8 pu:2", 2, 32, 256, 512,  71, 184, 16},   /* the EPYC whose roles the placer skewed */
	{"node:2 l3:2 core:4 pu:1",  2,  4,  16,  16,   6,   9, 16},
	{"node:2 l3:2 core:4 pu:2",  2,  4,  16,  32,  20,  10,  2},   /* every CPU but the service's sibling, groups within a domain */
	{"node:4 l3:2 core:4 pu:2",  4,  8,  32,  64,  13,  21, 16},   /* four NUMA nodes, more threads than cores */
	{"node:1 l3:4 core:4 pu:2",  1,  4,  16,  32,  17,   7,  4},   /* one NUMA node: the order stays the domains' */
};

/* the plan's threads of `role` per NUMA node, or of every role when role < 0.  The synthetic machines number
 * their NUMA nodes 0..n-1, which is what thread_numa holds. */
static void count_by_numa(const RouterPlacement &p, int role, std::vector<int> &cnt)
{
	cnt.assign(p.n_numa_nodes, 0);
	for (int t = 0; t < p.n_threads; t++) {
		CHECK(p.thread_numa[t] >= 0 && p.thread_numa[t] < p.n_numa_nodes);
		if (role < 0 || p.thread_role[t] == role)
			cnt[p.thread_numa[t]] += 1;
	}
}

/* max - min over `cnt`: what a round-robin deal bounds by one, be it over the domains or the NUMA nodes */
static int spread(const std::vector<int> &cnt)
{
	int lo = cnt[0];
	int hi = cnt[0];
	for (size_t k = 1; k < cnt.size(); k++) {
		lo = std::min(lo, cnt[k]);
		hi = std::max(hi, cnt[k]);
	}
	return hi - lo;
}

/*
 * The pinned plan: a CPU of its own for every thread, a core of its own while cores last and the service's core
 * holding the service alone, the whole team
 * spread over the domains, and -- what the domain order is for -- every role spread over the NUMA nodes, not
 * just the team as a whole.  The machines are regular, so every one of those is a round-robin deal: +- 1.
 */
static void check_plan(const RouterPlacement &p, const PlacementCase &c)
{
	int nt = 1 + c.producers + c.consumers;
	CHECK(p.n_threads == nt);
	CHECK(p.n_numa_nodes == c.n_numa);
	CHECK(p.n_domains == c.n_dom);
	if (p.n_numa_nodes != c.n_numa || p.n_domains != c.n_dom)
		return;

	int pu_per_core = c.n_cpu / c.n_core;       /* a synthetic machine numbers its PUs core by core */
	std::vector<int> per_cpu(c.n_cpu, 0);
	std::vector<int> per_core(c.n_core, 0);
	for (int t = 0; t < nt; t++) {
		CHECK(p.thread_cpu[t] >= 0 && p.thread_cpu[t] < c.n_cpu);
		if (p.thread_cpu[t] < 0 || p.thread_cpu[t] >= c.n_cpu)
			return;
		per_cpu[p.thread_cpu[t]] += 1;
		per_core[p.thread_cpu[t] / pu_per_core] += 1;
	}
	for (int cpu = 0; cpu < c.n_cpu; cpu++)
		CHECK(per_cpu[cpu] <= 1);
	CHECK(per_core[p.thread_cpu[0] / pu_per_core] == 1);   /* nothing shares the service thread's core */
	if (nt <= c.n_core)                         /* SMT siblings fill up only once every core is taken */
		for (int k = 0; k < c.n_core; k++)
			CHECK(per_core[k] <= 1);

	std::vector<int> per_dom(c.n_dom, 0);
	for (int t = 0; t < nt; t++) {
		CHECK(p.thread_domain[t] >= 0 && p.thread_domain[t] < c.n_dom);
		if (p.thread_domain[t] < 0 || p.thread_domain[t] >= c.n_dom)
			return;
		per_dom[p.thread_domain[t]] += 1;
	}
	CHECK(spread(per_dom) <= 1);

	std::vector<int> cnt;                       /* the team, then each role on its own, over the NUMA nodes */
	count_by_numa(p, -1, cnt);
	CHECK(spread(cnt) <= 1);
	int tol = (nt + pu_per_core - 1 >= c.n_cpu) ? 2 : 1;   /* a team filling every CPU but the service's idle
	                                                        * siblings: its node has that many PUs fewer for the
	                                                        * workers, and one role's deal is off by one more */
	count_by_numa(p, ROUTER_CONSUMER, cnt);
	CHECK(spread(cnt) <= tol);
	count_by_numa(p, ROUTER_PRODUCER, cnt);
	CHECK(spread(cnt) <= tol);
}

/*
 * The groups of a pinned AUTO plan: one cache domain each, every worker in one and the service thread in none,
 * both roles spread over a domain's groups, and every consumer in exactly one group by its local index.
 */
static void check_groups(const RouterPlacement &p, const PlacementCase &c)
{
	int nt = 1 + c.producers + c.consumers;
	CHECK(p.n_groups > 0);
	std::vector<int> gs(p.n_groups, 0);
	std::vector<int> gr(p.n_groups, 0);
	for (int t = 0; t < nt; t++) {
		if (p.thread_role[t] == ROUTER_SERVICE) {
			CHECK(p.thread_group[t] == -1);
			continue;
		}
		int g = p.thread_group[t];
		CHECK(g >= 0 && g < p.n_groups);
		if (g < 0 || g >= p.n_groups)
			return;
		CHECK(p.group_domain[g] == p.thread_domain[t]);
		if (p.thread_role[t] == ROUTER_PRODUCER)
			gs[g] += 1;
		else
			gr[g] += 1;
	}
	for (int g = 0; g < p.n_groups; g++) {
		CHECK(gs[g] == p.group_nsend[g]);
		CHECK(gr[g] == (int) p.group_consumers[g].size());
	}

	for (int d = 0; d < c.n_dom; d++) {         /* a domain's groups hold the same of each role, +- 1 */
		std::vector<int> ds;
		std::vector<int> dr;
		for (int g = 0; g < p.n_groups; g++) {
			if (p.group_domain[g] != d)
				continue;
			ds.push_back(gs[g]);
			dr.push_back(gr[g]);
		}
		CHECK(not ds.empty());
		if (ds.empty())
			return;
		CHECK(spread(ds) <= 1);
		CHECK(spread(dr) <= 1);
	}

	std::vector<int> seen(c.consumers, 0);      /* the local consumer indices are a partition of 0..R-1 */
	for (int g = 0; g < p.n_groups; g++)
		for (size_t k = 0; k < p.group_consumers[g].size(); k++) {
			int r = p.group_consumers[g][k];
			CHECK(r >= 0 && r < c.consumers);
			if (r >= 0 && r < c.consumers)
				seen[r] += 1;
		}
	for (int r = 0; r < c.consumers; r++)
		CHECK(seen[r] == 1);
}

/* the unpinned plan: no CPU and no topology at all, the groups cut out of the worker index, both roles +- 1 */
static void check_plan_unpinned(const RouterPlacement &p, const PlacementCase &c)
{
	int nt = 1 + c.producers + c.consumers;
	int gs = (c.producers + c.group_size - 1) / c.group_size;
	int gr = (c.consumers + c.group_size - 1) / c.group_size;
	CHECK(p.n_groups == std::max(std::max(gs, gr), 1));
	CHECK(p.n_domains == 0);
	for (int t = 0; t < nt; t++) {
		CHECK(p.thread_cpu[t] == -1);
		CHECK(p.thread_numa[t] == -1);
		CHECK(p.thread_domain[t] == -1);
		if (p.thread_role[t] == ROUTER_SERVICE)
			CHECK(p.thread_group[t] == -1);
	}
	std::vector<int> ns(p.n_groups, 0);
	std::vector<int> nr(p.n_groups, 0);
	for (int t = 0; t < nt; t++) {
		if (p.thread_role[t] == ROUTER_SERVICE)
			continue;
		CHECK(p.thread_group[t] >= 0 && p.thread_group[t] < p.n_groups);
		if (p.thread_group[t] < 0 || p.thread_group[t] >= p.n_groups)
			return;
		if (p.thread_role[t] == ROUTER_PRODUCER)
			ns[p.thread_group[t]] += 1;
		else
			nr[p.thread_group[t]] += 1;
	}
	CHECK(spread(ns) <= 1);
	CHECK(spread(nr) <= 1);
}

/* the caller's groups: the color is the group, whatever the domain the thread was pinned in */
static void check_plan_colors(const RouterPlacement &p, const PlacementCase &c, const std::vector<int> &colors)
{
	int nt = 1 + c.producers + c.consumers;
	int G = 0;
	for (int t = 1; t < nt; t++)
		G = std::max(G, colors[t] + 1);
	CHECK(p.n_groups == G);
	for (int t = 1; t < nt; t++) {
		CHECK(p.thread_group[t] == colors[t]);
		CHECK(p.group_domain[colors[t]] == -1);
	}
}

/* the pinned plan in one line: what every NUMA node got of each role, which is what the domain order decides */
static void report_plan(const RouterPlacement &p, const PlacementCase &c)
{
	std::vector<int> ns;
	std::vector<int> nr;
	count_by_numa(p, ROUTER_PRODUCER, ns);
	count_by_numa(p, ROUTER_CONSUMER, nr);
	printf("== placement (%s, %d producers, %d consumers, %d group(s)):", c.machine, c.producers, c.consumers,
	       p.n_groups);
	for (int j = 0; j < p.n_numa_nodes; j++)
		printf("  NUMA %d: %ds %dr", j, ns[j], nr[j]);
	printf("\n");
}

/*
 * Every case on its machine: the pinned AUTO plan and its groups, the same team unpinned, then the same team
 * pinned with the caller's colors.  HWLOC_SYNTHETIC is set per case and restored, so the configurations that
 * follow see the real machine again; every RouterPlacement loads the topology itself, so it takes effect.
 */
static void test_placement(bool verbose)
{
	const char *saved = getenv("HWLOC_SYNTHETIC");
	std::string old = (saved != NULL) ? saved : "";
	for (size_t k = 0; k < sizeof(PLACEMENT_CASES) / sizeof(PLACEMENT_CASES[0]); k++) {
		const PlacementCase &c = PLACEMENT_CASES[k];
		setenv("HWLOC_SYNTHETIC", c.machine, 1);
		cpu_set_t mask;
		CPU_ZERO(&mask);
		for (int cpu = 0; cpu < c.n_cpu; cpu++)
			CPU_SET(cpu, &mask);
		int nt = 1 + c.producers + c.consumers;
		std::vector<int> roles(nt, ROUTER_PRODUCER);
		roles[0] = ROUTER_SERVICE;
		for (int t = 1; t <= c.consumers; t++)
			roles[t] = ROUTER_CONSUMER;
		std::vector<int> autos(nt, ROUTER_GROUP_AUTO);

		RouterPlacement pinned(mask, roles, autos, true, 0, c.group_size, 0);
		check_plan(pinned, c);
		check_groups(pinned, c);
		if (verbose)
			report_plan(pinned, c);

		RouterPlacement loose(mask, roles, autos, false, 0, c.group_size, 0);
		check_plan_unpinned(loose, c);

		std::vector<int> colors(nt, ROUTER_GROUP_AUTO);   /* two groups, as PCS would split a team by hand */
		for (int t = 1; t < nt; t++)
			colors[t] = (t - 1) % 2;
		RouterPlacement split(mask, roles, colors, true, 0, c.group_size, 0);
		check_plan(split, c);
		check_plan_colors(split, c, colors);
	}
	if (saved != NULL)
		setenv("HWLOC_SYNTHETIC", old.c_str(), 1);
	else
		unsetenv("HWLOC_SYNTHETIC");
}

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
	if (provided < MPI_THREAD_FUNNELED)
		errx(1, "MPI_THREAD_FUNNELED not provided");
	MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &g_nodes);
	RouterArgs a;
	router_parse(argc, argv, a);
	a.opts.verbose = a.opts.verbose && g_rank == 0;
	bool all = a.test == "all";

	if (all || a.test == "placement") {
		if (g_rank == 0)                /* the placer is pure computation: one rank runs it, no team at all */
			test_placement(a.opts.verbose);
	}
	if (all || a.test == "connect_only") {
		RouterArgs c = a;
		c.rounds = 0;
		run_config(c, "connect_only");
	}
	if (all || a.test == "basic") {
		RouterArgs c = a;
		run_config(c, "basic");
	}
	if (all || a.test == "tiny") {
		RouterArgs c = a;
		c.opts.block_points = 8;
		c.opts.swc_linesize = 4;
		c.opts.n_recv = 2;
		c.opts.inbox_blocks = 4;
		c.points = a.points / 5;
		run_config(c, "tiny");
	}
	if (all || a.test == "rounds") {
		RouterArgs c = a;
		c.rounds = 5;
		run_config(c, "rounds");
	}
	if (all || a.test == "zero") {
		RouterArgs c = a;
		c.points = 0;
		run_config(c, "zero");
	}
	if (all || a.test == "skew") {
		RouterArgs c = a;
		c.skew = 1;
		run_config(c, "skew");
	}
	if (all || a.test == "partial_lines") {
		RouterArgs c = a;
		c.partial = true;
		c.opts.swc_linesize = 32;
		run_config(c, "partial_lines");
	}
	if (all || a.test == "swc_is_block") {
		RouterArgs c = a;
		c.opts.swc_linesize = 64;
		c.opts.block_points = 64;
		run_config(c, "swc_is_block");
	}
	if (all || a.test == "staggered_close") {
		RouterArgs c = a;
		c.stagger = true;
		run_config(c, "staggered_close");
	}
	if (all || a.test == "slow_recv") {
		RouterArgs c = a;
		c.slow_recv = true;
		c.points = a.points / 20;
		run_config(c, "slow_recv");
	}
	if (all || a.test == "limits") {  /* tiny blocks exhaust the pool and dawdling consumers hold them: the producers wait */
		RouterArgs c = a;
		c.opts.block_points = 8;
		c.opts.swc_linesize = 4;
		c.opts.n_recv = 2;
		c.opts.inbox_blocks = 4;
		c.points = a.points / 5;
		c.slow_recv = true;
		c.check_limit = true;
		run_config(c, "limits");
	}
	if (all || a.test == "groups") {
		RouterArgs c = a;
		c.opts.pin = false;             /* one L3 domain on a laptop: exercise several groups without pinning */
		c.opts.group_size = 2;
		run_config(c, "groups");
	}
	if (all || a.test == "colors") {
		RouterArgs c = a;
		run_config(c, "colors", true);
	}
	if ((all && g_nodes == 1) || a.test == "asymmetric") {
		RouterArgs c = a;
		c.producers = 1;
		c.consumers = 6;
		run_config(c, "asymmetric");
		c.producers = 6;
		c.consumers = 1;
		run_config(c, "asymmetric");
	}

	int failed = g_failed.load();
	int total = 0;
	MPI_Reduce(&failed, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	if (g_rank == 0)
		printf(total == 0 ? "router_test: all checks passed\n" : "router_test: %d CHECKS FAILED\n", total);
	MPI_Finalize();
	return total == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
