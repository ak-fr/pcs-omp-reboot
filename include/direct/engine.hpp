#ifndef MITM_DIRECT_ENGINE
#define MITM_DIRECT_ENGINE

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cassert>
#include <cstring>
#include <vector>

#include "tools.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "router/router.hpp"
#include "direct/params.hpp"
#include "direct/shared.hpp"
#include "direct/wrappers.hpp"
#include "direct/dict.hpp"
#include "direct/producer.hpp"

/*
 * The direct engine itself: the team, the printing, the service thread's round and the epilogue's
 * Allgather, run(), and the two entry points, mitm::direct::claw_search and mitm::direct::collision_search.
 */

namespace mitm::direct {

using fmt::print;       /* the reports format with {fmt}, unqualified */


/******************************** the printing ********************************/

/* the startup report, on rank 0 before the team exists; the Router prints the team's shape right after it */
template <class Wrapper>
static void banner(const Wrapper &wrapper, const Params &params, u64 seed)
{
	print("Starting direct {} search with {} : {{0,1}}^{} --> {{0, 1}}^{} (vlen={})\n", Wrapper::kind,
	      Wrapper::funcs, wrapper.n, wrapper.m, Wrapper::vlen);
	print("Starting the direct (exhaustive) meet-in-the-middle on the Router with seed={:016x}\n", seed);
	double work = (double) params.n_rounds * (params.per_round + params.domain);
	std::string hper = human_format(params.per_round);
	print("Dictionary: linear probing, 8-byte slots = {}-bit preimage | {} check bits | occupancy.  "
	      "Fill {:.2f}: {} entries per round.  Prefetch {} points ahead\n", params.n, 63 - params.n, params.fill,
	      hper, params.prefetch);
	print("RAM per node == {:.1f} MB of dictionary; {} slots in all (2^{:.2f}), {} per shard\n",
	      (double) params.w_shard * params.dicts_per_node * sizeof(u64) / 1e6, human_format(params.w),
	      std::log2((double) params.w), hper);
	print("{} round(s) of two phases: FILL (f on 2^{:.2f} preimages) then PROBE (g on all 2^{}).  "
	      "Total work: {} evaluations = 2^{:.2f}\n", params.n_rounds, std::log2((double) params.per_round),
	      params.n, human_format(work), std::log2(work));
	if (params.capped)
		print("NOTICE: --nrounds caps the search at {} round(s): it is NOT exhaustive\n",
		      params.n_rounds);
	fflush(stdout);
}

/*
 * How to get the 2 MB pages the kernel refused: the pool is the administrator's to reserve and cannot be grown
 * from here, so this is advice for the next run.
 */
static void hugetlb_advice(u64 nbytes, int n_shards)
{
	u64 pages = (u64) n_shards * nbytes / HUGE_PAGE;
	print("            to get them: as root, `sysctl vm.nr_hugepages={}` (or `echo {} > /proc/sys/vm/nr_hugepages`),\n", pages, pages);
	print("            then run again.  {} pages cover this rank's {} shard(s); ask for more -- the pool is the host's,\n", pages, n_shards);
	print("            shared by every rank on it and spread over its NUMA nodes.  `vm.nr_hugepages = {}` in /etc/sysctl.conf,\n", pages);
	print("            or `hugepages={}` on the kernel command line, reserves it at every boot, out of unfragmented memory.\n", pages);
}

/*
 * What this node's dict threads got for their shards' pages, printed by thread 0 once every one of them has
 * reported -- before the live line starts, which is why the service thread holds that line until then.
 */
static void shard_report(const Shared &shared)
{
	int refused = 0;
	for (size_t i = 0; i < shared.pages.size(); i++) {
		const ShardPages &p = shared.pages[i];
		if (p.hugetlb_errno == 0) {
			print("Dictionary: shard {} of {}B on MAP_HUGETLB, {}B measured on 2 MB pages\n", i,
			      human_format(p.nbytes), human_format(p.huge));
			continue;
		}
		print("Dictionary: shard {} of {}B: no MAP_HUGETLB ({}), {}B measured on transparent 2 MB pages\n",
		      i, human_format(p.nbytes), strerror(p.hugetlb_errno), human_format(p.huge));
		refused += 1;
	}
	if (refused > 0)
		hugetlb_advice(shared.pages[0].nbytes, (int) shared.pages.size());
	fflush(stdout);
}

/* the live one-line refresh: rank 0's own node, read from its tallies without synchronisation, scaled to
   every node -- the points its dict threads retired are the points the Router delivered to them -- and the
   Router's verdict on what limits the phase */
static void display(const Router_thread &rt, const Params &params, const Shared &shared, const u64 *stats,
                    double delta, u64 round, int phase)
{
	const int nodes = Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL);
	u64 eval = 0;
	u64 retired = 0;
	for (size_t t = 0; t < shared.tally.size(); t++) {
		eval += shared.tally[t].ctr[N_EVAL];
		retired += shared.tally[t].ctr[N_INSERT] + shared.tally[t].ctr[N_PROBE];
	}
	u64 span = params.domain;
	if (phase == FILL)
		span = std::min(params.per_round, params.domain - round * params.per_round);
	double completion = (double) eval * nodes / (double) span;
	double send_wait;                      /* share of the phase the producers waited for a block */
	double recv_wait;                      /* share the dict threads had nothing to pop */
	double busy;                           /* share the service spent in turns that moved something */
	const char *limit = Router_Bottleneck(stats, rt, &send_wait, &recv_wait, &busy);
	print("\rRound {}/{} {}:  {:.1f}s ({:.1f}%, ETA {:.1f}s).  {} points routed/s.  node-->{}B/s.  "
	      "wait: producers {:5.1f}% consumers {:5.1f}%, service {:5.1f}% busy --> {:<15}   ",
	      round, params.n_rounds, (phase == FILL) ? "FILL" : "PROBE", delta, 100. * completion,
	      (completion > 0) ? delta * (1 - completion) / completion : 0.,
	      human_format((double) retired * nodes / delta),
	      human_format((double) stats[ROUTER_BYTES_SENT] / delta),
	      100. * send_wait, 100. * recv_wait, 100. * busy, limit);
	fflush(stdout);
}

/* the end-of-phase report: `r` is the exact sum over every thread of every node, `total` the all-time sums */
static void round_report(const Router_thread &rt, const Params &params, const u64 r[], const u64 total[],
                         double delta, u64 round, int phase)
{
	const int nodes = Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL);
	print("\n");
	print("Round {} {}.  {:.1f}s.  2^{:.2f} evaluations (total 2^{:.2f}).  {} points routed/s.  "
	      "node-->{}B/s\n", round, (phase == FILL) ? "FILL" : "PROBE", delta,
	      std::log2((double) (r[N_EVAL] ? r[N_EVAL] : 1)),
	      std::log2((double) (total[N_EVAL] ? total[N_EVAL] : 1)),
	      human_format((double) (r[N_INSERT] + r[N_PROBE]) / delta),
	      human_format((double) r[REC_ROUTER + ROUTER_BYTES_SENT] / nodes / delta));
	double send_wait;                      /* share of the phase the producers waited for a block */
	double recv_wait;                      /* share the dict threads had nothing to pop */
	double busy;                           /* share the service spent in turns that moved something */
	const char *limit = Router_Bottleneck(r + REC_ROUTER, rt, &send_wait, &recv_wait, &busy);
	print("            LIMITED BY {}:  producers waited {:.1f}% of the phase, consumers {:.1f}%, the service was "
	      "busy {:.1f}%\n", limit, 100. * send_wait, 100. * recv_wait, 100. * busy);
	if (r[N_INSERT] > 0)
		print("            {} inserted, load {:.2f}/slot\n", r[N_INSERT], (double) r[N_INSERT] / params.w);
	if (r[N_PROBE] > 0)
		print("            {} probed.  {} collisions (total 2^{:.2f})\n", r[N_PROBE], r[N_COLLISIONS],
		      std::log2((double) (total[N_COLLISIONS] ? total[N_COLLISIONS] : 1)));
	if (r[REC_ROUTER + ROUTER_STALL_OUT] | r[REC_ROUTER + ROUTER_STALL_IN])
		print("            STALLED  {} blocks held back for a full destination / {} receives left "
		      "unposted for want of a free block\n", r[REC_ROUTER + ROUTER_STALL_OUT],
		      r[REC_ROUTER + ROUTER_STALL_IN]);
	if (r[REC_ROUTER + ROUTER_PUSHED] != r[REC_ROUTER + ROUTER_POPPED])
		print("            ACCOUNTING BROKEN: {} points pushed, {} delivered\n",
		      r[REC_ROUTER + ROUTER_PUSHED], r[REC_ROUTER + ROUTER_POPPED]);
	print("\n");
	fflush(stdout);
}

/* the last line: found, or the domain exhausted -- a proof of absence, unless --nrounds cut it short */
static void done(const Params &params, u64 phases, bool found, double seconds)
{
	u64 rounds = (phases + 1) / 2;
	if (found)
		print("Completed in {:.2f}s\n", seconds);
	else if (params.capped)
		print("Gave up after {} of the {} round(s) an exhaustive search needs ({:.2f}s)\n",
		      rounds, params.n_rounds_full, seconds);
	else
		print("No solution: the domain is exhausted after {} round(s) ({:.2f}s)\n", rounds, seconds);
	fflush(stdout);
}


/******************************** the service thread and the epilogue ********************************/

/*
 * Thread 0's phase: turn the Router until the node has sent and received everything, refreshing the live
 * line, then read the node's Router tallies, which Router_Reset clears.
 */
static void service_round(Router_thread &rt, const Params &params, const Shared &shared, u64 *stats, u64 round,
                          int phase, double t0)
{
	double last = t0;
	u64 turns = 0;
	bool pending = params.verbose && round == 0 && phase == FILL;   /* the shard report is still to come */
	while (not Router_Test_quiescent(rt)) {
		Router_Progress(rt);
		turns += 1;
		if (pending) {
			if (shared.pages_ready.load_acquire() < (u32) Router_size(rt, ROUTER_CONSUMER, ROUTER_NODE))
				continue;              /* no live line over a shard still being mapped and touched */
			shard_report(shared);
			pending = false;
		}
		if (not params.verbose || (turns & 0xff) != 0)
			continue;
		double now = wtime();
		if (now - last < params.ping_delay)
			continue;
		last = now;
		Router_Stats(stats, rt);
		display(rt, params, shared, stats, now - t0, round, phase);
	}
	Router_Stats(stats, rt);
}

/*
 * Thread 0's end of phase, once the Router is reset: the node's record -- its tallies, its Router stats and
 * the golden pair it may hold -- goes into one Allgather, and every node reads the same verdict out of it.
 * The lowest rank that found a pair provides the answer, written over the node's own golden pair: after the
 * epilogue, found and golden[] are the verdict, the same on every node.
 */
static void epilogue(const Router_thread &rt, const Params &params, Shared &shared, const u64 *stats,
                     u64 *records, u64 *total, u64 round, int phase, double delta)
{
	const int nodes = Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL);
	u64 rec[REC_WORDS] = {};
	for (size_t t = 0; t < shared.tally.size(); t++)
		for (int c = 0; c < N_COUNTERS; c++)
			rec[c] += shared.tally[t].ctr[c];
	for (int k = 0; k < ROUTER_STATS_SIZE; k++)
		rec[REC_ROUTER + k] = stats[k];
	if (shared.found.load_acquire()) {
		rec[REC_FOUND] = 1;
		rec[REC_X] = shared.golden[0];
		rec[REC_Y] = shared.golden[1];
	}
	MPI_Allgather(rec, REC_WORDS, MPI_UINT64_T, records, REC_WORDS, MPI_UINT64_T, params.mpi_comm);

	u64 sum[REC_WORDS] = {};
	int winner = -1;                       /* the lowest rank with a pair: its pair becomes everyone's */
	for (int r = 0; r < nodes; r++) {
		const u64 *q = records + (size_t) r * REC_WORDS;
		for (int k = 0; k < REC_FOUND; k++)
			sum[k] += q[k];
		if (q[REC_FOUND] && winner < 0)
			winner = r;
	}
	if (winner >= 0) {
		const u64 *q = records + (size_t) winner * REC_WORDS;
		shared.golden[0] = q[REC_X];
		shared.golden[1] = q[REC_Y];
		shared.found = 1;
	}
	for (int k = 0; k < REC_FOUND; k++)
		total[k] += sum[k];
	shared.phases += 1;
	if (params.verbose)
		round_report(rt, params, sum, total, delta, round, phase);
}

/******************************** the engine ********************************/

/*
 * The search itself: one OpenMP team per node -- thread 0 the Router's service thread, the next R its dict
 * threads, the rest its producers -- running the same deterministic sequence of phases, (0, FILL), (0, PROBE),
 * (1, FILL), ..., each one Router round.  Returns x and y of the golden pair, the same on every node, or
 * nothing once the domain is exhausted.
 */
template <class Wrapper>
optional<pair<u64, u64>> run(const Wrapper &wrapper, u64 nbytes_memory, const Options &opts, PRNG &prng)
{
	const Params params(opts, nbytes_memory, wrapper.n, wrapper.m);

	/* self-test */
	u64 mine[3];
	mine[0] = prng.rand();
	mine[1] = prng.rand();
	mine[2] = wrapper.self_test(mine[0], mine[1]);
	u64 theirs[3] = {mine[0], mine[1], mine[2]};
	MPI_Bcast(theirs, 3, MPI_UINT64_T, 0, params.mpi_comm);
	if (theirs[0] != mine[0] || theirs[1] != mine[1] || theirs[2] != mine[2])
		errx(1, "direct: the ranks do not hold the same problem instance (self-test mismatch)");

	if (params.verbose)
		banner(wrapper, params, prng.seed);

	/* the team: thread 0 the service, then the dict threads, then the producers; the Router counts the roles */
	const int R = params.dicts_per_node;
	const int n_threads = 1 + R + params.producers_per_node;
	Shared shared(n_threads, R);
	double t_start = wtime();

	#pragma omp parallel num_threads(n_threads)
	{
		int tid = omp_get_thread_num();
		int role = ROUTER_PRODUCER;
		if (tid == 0)
			role = ROUTER_SERVICE;
		else if (tid <= R)
			role = ROUTER_CONSUMER;
		Router_thread rt = Router_Init(role, ROUTER_GROUP_AUTO, params.mpi_comm, ROUTER_TAG, &params.router);
		u64 *ctr = shared.tally[tid].ctr;

		/* a dict thread's shard, mapped and zero-filled here: its own first touch.  Empty on the other roles */
		DirectDict dict(role == ROUTER_CONSUMER ? params.w_shard : 0, params.n);
		if (role == ROUTER_CONSUMER)
			shared.publish_pages(Router_rank(rt, ROUTER_NODE), dict.nbytes, dict.huge, dict.hugetlb_errno);

		std::vector<u64> records;              /* thread 0: the nodes' records, from the Allgather */
		std::vector<u64> total;                /* thread 0: the all-time sums */
		std::vector<u64> stats;                /* thread 0: the node's Router tallies for the phase */
		if (role == ROUTER_SERVICE) {
			records.assign((size_t) Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL) * REC_WORDS, 0);
			total.assign(REC_WORDS, 0);
			stats.assign(ROUTER_STATS_SIZE, 0);
		}

		for (u64 step = 0;; step++) {
			u64 round = step / 2;
			int phase = (step % 2 == 0) ? FILL : PROBE;
			for (int c = 0; c < N_COUNTERS; c++)
				ctr[c] = 0;
			double t0 = wtime();

			if (role == ROUTER_SERVICE)
				service_round(rt, params, shared, stats.data(), round, phase, t0);
			else if (role == ROUTER_CONSUMER)
				dict_round(rt, wrapper, shared, dict, ctr, round, phase, params.prefetch);
			else
				producer_round(rt, wrapper, params, ctr, round, phase);

			Router_Reset(rt);      /* its own team barriers and MPI_Barrier: every tally is written by now */

			if (role == ROUTER_SERVICE)
				epilogue(rt, params, shared, stats.data(), records.data(), total.data(), round, phase, wtime() - t0);

			#pragma omp barrier    /* the verdict, and every thread's next tally, are thread 0's to publish */
			if (shared.found || (phase == PROBE && round + 1 >= params.n_rounds))
				break;
		}
	}

	if (params.verbose)
		done(params, shared.phases, shared.found, wtime() - t_start);
	if (not shared.found)
		return nullopt;
	return optional(pair(shared.golden[0], shared.golden[1]));
}


/******************************** the entry points ********************************/

/* find x0 != x1 with f(x0) == f(x1) and is_good_pair(x0, x1), or prove there is none */
template <class Problem>
optional<pair<u64, u64>> collision_search(const Problem &pb, u64 nbytes_memory, const Options &opts, PRNG &prng)
{
	CollisionWrapper<Problem> wrapper(pb);
	auto collision = run(wrapper, nbytes_memory, opts, prng);
	if (not collision)
		return nullopt;

	auto [x0, x1] = *collision;
	assert(x0 != x1);
	assert(pb.f(x0) == pb.f(x1));
	assert(pb.is_good_pair(x0, x1));
	return optional(pair(x0, x1));
}

/* find x0, x1 with f(x0) == g(x1) and is_good_pair(x0, x1), or prove there is none */
template <class Problem>
optional<pair<u64, u64>> claw_search(const Problem &pb, u64 nbytes_memory, const Options &opts, PRNG &prng)
{
	ClawWrapper<Problem> wrapper(pb);
	auto claw = run(wrapper, nbytes_memory, opts, prng);
	if (not claw)
		return nullopt;

	auto [x0, x1] = *claw;
	assert(pb.f(x0) == pb.g(x1));
	assert(pb.is_good_pair(x0, x1));
	return optional(pair(x0, x1));
}

}
#endif
