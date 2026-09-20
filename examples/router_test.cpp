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
#include <hwloc.h>

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

/* rank-local, on thread 0 after Init: the team's shape as the Router answers it, a CPU of its own for every
 * thread when pinned, and Router_closeness's invariants: symmetric, a thread closest to itself, and 0 for every
 * pair when the threads are not pinned.  Which consumer a producer pairs with is PCS's own business (it derives
 * it from Router_closeness, include/pcs/shared.hpp's assign_consumers), which test_placement checks on
 * synthetic machines. */
static void check_placement(const Router_thread &rt, const RouterArgs &a, Shared &sh)
{
	int P = a.producers;
	int C = a.consumers;
	int nt = 1 + P + C;
	CHECK(Router_size(rt, ROUTER_PRODUCER, ROUTER_NODE) == P);      /* one role asking about another */
	CHECK(Router_size(rt, ROUTER_CONSUMER, ROUTER_NODE) == C);
	CHECK(Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL) == g_nodes);
	if (a.opts.pin)
		for (int t = 0; t < nt; t++)
			for (int u = t + 1; u < nt; u++)
				CHECK(sh.t_cpu[t] != sh.t_cpu[u]);   /* every thread on a CPU of its own */
	for (int s = 0; s < P; s++) {
		int self = Router_closeness(rt, ROUTER_PRODUCER, s, ROUTER_PRODUCER, s);
		CHECK(a.opts.pin ? self > 0 : self == 0);
		for (int r = 0; r < C; r++) {
			int close = Router_closeness(rt, ROUTER_PRODUCER, s, ROUTER_CONSUMER, r);
			CHECK(close == Router_closeness(rt, ROUTER_CONSUMER, r, ROUTER_PRODUCER, s));
			CHECK(close <= self);
			CHECK(a.opts.pin || close == 0);
		}
	}
}

/* one configuration, start to finish: a team, its Router, `rounds` rounds; everything dies with the region */
static void run_config(const RouterArgs &a, const char *name)
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
	sh.t_cpu.assign(nt, 0);

	#pragma omp parallel num_threads(nt)
	{
		int tid = omp_get_thread_num();
		int role = (tid == 0) ? ROUTER_SERVICE : (tid <= R ? ROUTER_CONSUMER : ROUTER_PRODUCER);
		int li = (role == ROUTER_CONSUMER) ? (tid - 1) : (tid - 1 - R);   /* my local rank, when a worker */
		Router_thread rt = Router_Init(role, MPI_COMM_WORLD, 42, &a.opts);
		sh.t_cpu[tid] = sched_getcpu();
		int per_node = (role == ROUTER_SERVICE) ? 1 : (role == ROUTER_CONSUMER ? R : S);
		int mine = (role == ROUTER_SERVICE) ? 0 : li;                 /* what Router_rank must answer */
		CHECK(Router_size(rt, role, ROUTER_NODE) == per_node);
		CHECK(Router_size(rt, role, ROUTER_GLOBAL) == per_node * g_nodes);
		CHECK(Router_rank(rt, ROUTER_NODE) == mine);
		CHECK(Router_rank(rt, ROUTER_GLOBAL) == g_rank * per_node + mine);
		#pragma omp barrier
		if (tid == 0)
			check_placement(rt, a, sh);
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
 * The placer on fabricated machines.  RouterPlacement is pure computation over an affinity mask and the topology
 * hwloc reports -- it pins nothing, connect() does -- so a synthetic machine and a mask of its CPUs exercise the
 * shapes no test host has: the clusters the framework runs on, a rank owning half a socket, a team that fills
 * every CPU.  Each case is one machine, one mask and one team, and what the policy promises of it.
 */
struct PlacementCase {
	const char *machine;             /* the machine, as HWLOC_SYNTHETIC spells it: PUs numbered core by core */
	int cpu_lo;                      /* the mask: PUs cpu_lo .. cpu_hi - 1, whole NUMA nodes of equal size or part of one */
	int cpu_hi;
	int n_numa;                      /* NUMA nodes the mask spans, whatever their numbers */
	int pu_per_core;
	int producers;                   /* the team, besides the one service thread */
	int consumers;
	int far_pairs;                   /* walker / dict-thread pairs the greedy leaves in different L3 caches */
};

static const PlacementCase PLACEMENT_CASES[] = {
	{"node:1 l3:1 core:18 pu:2",   0,  18, 1, 2,   4,  12, 0},   /* gros, the rank owning cores 0-8: a direct-engine team */
	{"node:1 l3:1 core:18 pu:2",   0,  18, 1, 2,  11,   5, 0},   /* ... and a PCS one */
	{"node:1 l3:1 core:18 pu:2",  18,  36, 1, 2,   4,  12, 0},   /* the rank owning cores 9-17 */
	{"node:1 l3:1 core:18 pu:2",  18,  36, 1, 2,   2,   6, 0},
	{"node:2 l3:1 core:16 pu:2",   0,  64, 2, 2,  16,  46, 0},   /* grvingt: a quarter of producers, the mask full */
	{"node:2 l3:1 core:16 pu:2",   0,  64, 2, 2,  41,  21, 0},   /* two thirds */
	{"node:2 l3:1 core:16 pu:2",   0,  64, 2, 2,  10,  21, 0},   /* the measured direct team: one thread per core */
	{"node:2 l3:1 core:16 pu:2",   0,  64, 2, 2,  36,  25, 0},   /* the measured PCS teams */
	{"node:2 l3:1 core:16 pu:2",   0,  64, 2, 2,  57,   4, 0},
	{"node:2 l3:1 core:16 pu:2",  32,  64, 1, 2,   8,  22, 0},   /* grvingt, a rank per socket: the second one's mask */
	{"node:2 l3:1 core:16 pu:2",  32,  64, 1, 2,  22,   8, 0},
	{"node:2 l3:16 core:16 pu:1",  0, 512, 2, 1, 128, 383, 0},   /* grdix as 32 L3 caches of 16 cores */
	{"node:2 l3:16 core:16 pu:1",  0, 512, 2, 1, 337, 174, 0},
	{"node:2 l3:16 core:16 pu:1",  0, 512, 2, 1, 400, 110, 0},
	{"node:2 l3:16 core:8 pu:2",   0, 512, 2, 2, 128, 382, 0},   /* grdix as 32 L3 caches of 8 cores, SMT */
	{"node:2 l3:16 core:8 pu:2",   0, 512, 2, 2, 337, 173, 0},
	{"node:2 l3:16 core:8 pu:2",   0, 512, 2, 2, 159, 351, 0},   /* the reference direct team */
	{"node:2 l3:16 core:8 pu:2",   0, 512, 2, 2, 400, 110, 0},
	{"node:1 l3:4 core:4 pu:1",    0,  16, 1, 1,   4,   3, 2},   /* a laptop-like tree: an L3 holds a walker and no shard */
	{"node:1 l3:4 core:4 pu:1",    0,  16, 1, 1,  10,   5, 0},
	{"node:1 l3:4 core:4 pu:1",    0,  16, 1, 1,   3,   3, 1},   /* the team the old placer refused */
	{"node:1 pu:8",                0,   8, 1, 1,   4,   3, 0},   /* no core level at all: the service's PU is its core */
};

/* the depth of the L3 level of the machine HWLOC_SYNTHETIC names, or 0 when hwloc merged it into the machine */
static int l3_depth()
{
	hwloc_topology_t topo;
	if (hwloc_topology_init(&topo) != 0 || hwloc_topology_load(topo) != 0)
		errx(1, "hwloc_topology_load");
	int d = hwloc_get_type_depth(topo, HWLOC_OBJ_L3CACHE);
	hwloc_topology_destroy(topo);
	return d < 0 ? 0 : d;
}

/*
 * The pinned plan: the service on the first PU of the mask's leftmost core, that core its alone; a CPU of its own
 * for every thread and a core of its own while cores last; each role over the NUMA nodes in proportion to the PUs
 * they hold, the service's core discounted; and the pairing PCS derives from closeness -- the same greedy as
 * assign_consumers -- keeping every walker on its dict thread's NUMA node, in its L3 but for the pairs the case
 * expects, every shard with a walker when there are enough, and the walkers spread over the shards.
 */
static void check_plan(const RouterPlacement &p, const PlacementCase &c)
{
	int nt = 1 + c.producers + c.consumers;
	int n_cpu = c.cpu_hi - c.cpu_lo;
	CHECK(p.n_threads == nt);
	CHECK(p.n_numa_nodes == c.n_numa);
	if (p.n_threads != nt || p.n_numa_nodes != c.n_numa)
		return;

	std::vector<int> per_cpu(n_cpu, 0);
	std::vector<int> numa_ids;                            /* the NUMA nodes the threads landed on, whatever their numbers */
	for (int t = 0; t < nt; t++) {
		CHECK(p.thread_cpu[t] >= c.cpu_lo && p.thread_cpu[t] < c.cpu_hi);
		if (p.thread_cpu[t] < c.cpu_lo || p.thread_cpu[t] >= c.cpu_hi)
			return;
		per_cpu[p.thread_cpu[t] - c.cpu_lo] += 1;
		CHECK(p.thread_core[t] >= 0);
		CHECK(p.thread_numa[t] >= 0);
		if (std::find(numa_ids.begin(), numa_ids.end(), p.thread_numa[t]) == numa_ids.end())
			numa_ids.push_back(p.thread_numa[t]);
	}
	CHECK((int) numa_ids.size() == c.n_numa);
	if ((int) numa_ids.size() != c.n_numa)
		return;
	for (int cpu = 0; cpu < n_cpu; cpu++)
		CHECK(per_cpu[cpu] <= 1);
	CHECK(p.thread_cpu[0] == c.cpu_lo);                   /* the leftmost core, its first PU */
	int n_cores = n_cpu / c.pu_per_core;
	int shared = 0;
	for (int t = 0; t < nt; t++)
		for (int u = t + 1; u < nt; u++) {
			if (p.thread_core[t] != p.thread_core[u])
				continue;
			CHECK(t != 0);                                /* nothing shares the service thread's core */
			shared += 1;
		}
	if (nt - 1 <= n_cores - 1)                            /* SMT siblings fill up only once every core is taken */
		CHECK(shared == 0);
	CHECK((p.n_shared_cores == 0) == (shared == 0));

	int avail = n_cpu - c.pu_per_core;                    /* the workers' PUs, and each node's: the first node lost the service's core */
	for (int role = ROUTER_PRODUCER; role <= ROUTER_CONSUMER; role++) {
		int n_role = (role == ROUTER_PRODUCER) ? c.producers : c.consumers;
		std::vector<int> cnt(c.n_numa, 0);
		for (int t = 1; t < nt; t++)
			if (p.thread_role[t] == role)
				cnt[std::find(numa_ids.begin(), numa_ids.end(), p.thread_numa[t]) - numa_ids.begin()] += 1;
		for (int j = 0; j < c.n_numa; j++) {
			int node_avail = n_cpu / c.n_numa - (numa_ids[j] == p.thread_numa[0] ? c.pu_per_core : 0);
			int expect = (int) ((2.0 * node_avail * n_role / avail + 1) / 2);
			CHECK(std::abs(cnt[j] - expect) <= 1);
		}
	}

	int d3 = l3_depth();                                  /* the pairing, as assign_consumers computes it */
	std::vector<int> n_walkers(c.consumers, 0);
	int far = 0;
	for (int s = 0; s < c.producers; s++) {
		int ts = 1 + c.consumers + s;
		int best = 0;
		int best_close = p.closeness(ts, 1);
		for (int r = 1; r < c.consumers; r++) {
			int close = p.closeness(ts, 1 + r);
			if (close > best_close || (close == best_close && n_walkers[r] < n_walkers[best])) {
				best = r;
				best_close = close;
			}
		}
		n_walkers[best] += 1;
		int tr = 1 + best;
		CHECK(p.thread_numa[ts] == p.thread_numa[tr]);
		far += p.thread_path[ts][d3] != p.thread_path[tr][d3];
	}
	CHECK(far == c.far_pairs);
	int lo = n_walkers[0];
	int hi = n_walkers[0];
	for (int r = 1; r < c.consumers; r++) {
		lo = std::min(lo, n_walkers[r]);
		hi = std::max(hi, n_walkers[r]);
	}
	if (c.producers >= c.consumers)
		CHECK(lo >= 1);
	CHECK(hi - lo <= 2);
}

/* the unpinned plan: no CPU, no NUMA node, no path, and every closeness 0 */
static void check_plan_unpinned(const RouterPlacement &p, const PlacementCase &c)
{
	int nt = 1 + c.producers + c.consumers;
	CHECK(p.n_threads == nt);
	CHECK(p.n_numa_nodes == 0);
	for (int t = 0; t < nt; t++) {
		CHECK(p.thread_cpu[t] == -1);
		CHECK(p.thread_core[t] == -1);
		CHECK(p.thread_numa[t] == -1);
		CHECK(p.thread_path[t].empty());
		for (int u = 0; u < nt; u++)
			CHECK(p.closeness(t, u) == 0);
	}
}

/* the pinned plan in one line: what every NUMA node got of each role, and the cores shared */
static void report_plan(const RouterPlacement &p, const PlacementCase &c)
{
	printf("== placement (%s, CPUs %d-%d, %d producers, %d consumers): service on CPU %d;", c.machine, c.cpu_lo,
	       c.cpu_hi - 1, c.producers, c.consumers, p.thread_cpu[0]);
	std::vector<int> ids;                                 /* the NUMA nodes, as numbered, in order of first use */
	for (int t = 0; t < p.n_threads; t++)
		if (std::find(ids.begin(), ids.end(), p.thread_numa[t]) == ids.end())
			ids.push_back(p.thread_numa[t]);
	std::sort(ids.begin(), ids.end());
	for (size_t j = 0; j < ids.size(); j++) {
		int np = 0;
		int nc = 0;
		for (int t = 1; t < p.n_threads; t++) {
			if (p.thread_numa[t] != ids[j])
				continue;
			np += p.thread_role[t] == ROUTER_PRODUCER;
			nc += p.thread_role[t] == ROUTER_CONSUMER;
		}
		printf("  NUMA %d: %dp %dc", ids[j], np, nc);
	}
	printf(";  %d shared core(s)\n", p.n_shared_cores);
}

/*
 * Every case on its machine: the pinned plan, then the same team unpinned.  HWLOC_SYNTHETIC is set per case and
 * restored, so the configurations that follow see the real machine again; every RouterPlacement loads the
 * topology itself, so it takes effect.
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
		for (int cpu = c.cpu_lo; cpu < c.cpu_hi; cpu++)
			CPU_SET(cpu, &mask);
		int nt = 1 + c.producers + c.consumers;
		std::vector<int> roles(nt, ROUTER_PRODUCER);
		roles[0] = ROUTER_SERVICE;
		for (int t = 1; t <= c.consumers; t++)
			roles[t] = ROUTER_CONSUMER;

		RouterPlacement pinned(mask, roles, true, 0);
		check_plan(pinned, c);
		if (verbose)
			report_plan(pinned, c);

		RouterPlacement loose(mask, roles, false, 0);
		check_plan_unpinned(loose, c);
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
