#ifndef MITM_PCS_ENGINE
#define MITM_PCS_ENGINE

#include <mpi.h>
#include <omp.h>
#include <err.h>
#include <cmath>
#include <cassert>
#include <memory>
#include <vector>
#include <algorithm>

#include "tools.hpp"
#include "problem.hpp"
#include "parameters.hpp"
#include "router/router.hpp"
#include "pcs/params.hpp"
#include "pcs/shared.hpp"
#include "pcs/wrappers.hpp"
#include "pcs/dict.hpp"
#include "pcs/walker.hpp"
#include "pcs/control.hpp"

/*
 * PCS on the Router: van Oorschot-Wiener parallel collision search over a distributed dictionary of
 * trail endpoints.  A round walks trails under one version of the mixing function and ends when the
 * controller has counted beta*w distinguished points over all the nodes; the next round mixes
 * differently.  The team, the round, the epilogue and the two entry points.
 */

namespace mitm::pcs {

using fmt::print;       /* the reports format with {fmt}, unqualified */


/******************************** the printing ********************************/

/* the startup report, on rank 0 before the team exists; the Router prints the team's shape right after it */
template <class Wrapper>
static void banner(const Wrapper &wrapper, const Params &params, u64 seed)
{
	print("Starting PCS {} search with {} : {{0,1}}^{} --> {{0, 1}}^{} (vlen={})\n", Wrapper::kind,
	      Wrapper::funcs, wrapper.n, wrapper.m, Wrapper::vlen);
	print("Starting the parallel collision search on the Router with seed={:016x}\n", seed);
	print("RAM per node == {:.1f} MB of dictionary; {} slots in all (2^{:.2f}), {} per shard\n",
	      (double) params.w_shard * params.dicts_per_node * sizeof(u64) / 1e6, human_format(params.w),
	      std::log2((double) params.w), human_format(params.w_shard));
	print("DP == 2 words: endpoint + ({}-bit length | {}-bit chain index).  ", params.lenbits, params.jbits);
	if (params.dp_max_it > params.len_sat)
		print("Lengths >= {} re-walked\n", params.len_sat);
	else
		print("No length can saturate (dp_max_it == {})\n", params.dp_max_it);
	print("{:.1f}*w = {} = 2^{:.2f} distinguished points per version of the function\n", params.beta,
	      params.points_per_version, std::log2((double) params.points_per_version));
	if (params.theta_auto)
		print("AUTO-TUNING: setting 1/theta == {:.2f}\n", 1 / params.theta);
	else
		print("NOTICE: using 1/theta == {:.2f} vs ``optimal'' 1/theta == {:.2f}\n", 1 / params.theta,
		      1 / params.auto_theta);
	if (params.capped)
		print("NOTICE: --nrounds gives up after {} version(s) of the function\n", params.max_versions);
	fflush(stdout);
}

/* the live one-line refresh, from the progress reports so far: rank 0's, and approximate on purpose; it ends
   with the Router's verdict on what limits the round, from the same reports */
static void display(const Router_thread &rt, const Params &params, const u64 reported[], double delta, u64 nround)
{
	u64 ndp = reported[N_DP];
	double completion = (double) ndp / params.points_per_version;
	const int walkers = Router_size(rt, ROUTER_PRODUCER, ROUTER_GLOBAL);
	const int dicts = Router_size(rt, ROUTER_CONSUMER, ROUTER_GLOBAL);
	const int nodes = Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL);
	double send_wait;                      /* share of the round the walkers waited for a block */
	double recv_wait;                      /* share the dict threads had nothing to pop */
	double busy;                           /* share the service spent in turns that moved something */
	const char *limit = Router_Bottleneck(reported + REC_ROUTER, rt, &send_wait, &recv_wait, &busy);
	print("\rRound {}:  {:.1f}s ({:.1f}%, ETA {:.1f}s).  {:.2f}*w #DP.  {} #f/s per walker.  "
	      "{} probe/s per dict thread.  node-->{}B/s.  wait: producers {:5.1f}% consumers {:5.1f}%, "
	      "service {:5.1f}% busy --> {:<15}   ", nround + 1, delta, 100. * completion,
	      (completion > 0) ? delta * (1 - completion) / completion : 0.,
	      (double) ndp / params.w,
	      human_format((double) ndp / params.theta / walkers / delta),
	      human_format((double) reported[N_PROBE] / dicts / delta),
	      human_format((double) reported[REC_ROUTER + ROUTER_BYTES_SENT] / nodes / delta),
	      100. * send_wait, 100. * recv_wait, 100. * busy, limit);
	fflush(stdout);
}

/*
 * The end-of-round report: `r` is the exact sum over every thread of every node, `total` the all-time
 * sums, `round` and `all` the distinct-collision registers of the round and of every round.
 */
static void round_report(const Router_thread &rt, const Params &params, const u64 r[], const u64 total[],
                         const RoundStats &round, const RoundStats &all, double delta, u64 nround)
{
	u64 ndp = r[N_DP];
	const int walkers = Router_size(rt, ROUTER_PRODUCER, ROUTER_GLOBAL);
	const int dicts = Router_size(rt, ROUTER_CONSUMER, ROUTER_GLOBAL);
	const int nodes = Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL);

	print("\n");
	print("Round {}.  {:.1f}s.  #DP {:.2f}*w (total 2^{:.2f}).  #coll {:.2f}*w (total 2^{:.2f}).  "
	      "Total #f=2^{:.3f}.  {} #f/s per walker.  node-->{}B/s\n", nround, delta,
	      (double) ndp / params.w, std::log2((double) (total[N_DP] ? total[N_DP] : 1)),
	      (double) r[N_COLLISIONS] / params.w,
	      std::log2((double) (total[N_COLLISIONS] ? total[N_COLLISIONS] : 1)),
	      std::log2((double) (total[N_EVAL] ? total[N_EVAL] : 1)),
	      human_format((double) r[N_EVAL] / walkers / delta),
	      human_format((double) r[REC_ROUTER + ROUTER_BYTES_SENT] / nodes / delta));
	double send_wait;                      /* share of the round the walkers waited for a block */
	double recv_wait;                      /* share the dict threads had nothing to pop */
	double busy;                           /* share the service spent in turns that moved something */
	const char *limit = Router_Bottleneck(r + REC_ROUTER, rt, &send_wait, &recv_wait, &busy);
	print("            LIMITED BY {}:  producers waited {:.1f}% of the round, consumers {:.1f}%, the service was "
	      "busy {:.1f}%\n", limit, 100. * send_wait, 100. * recv_wait, 100. * busy);

	if (ndp > 0) {
		double avglen = (double) r[N_POINTS_TRAILS] / ndp;
		print("            {:.2f} avg trail length", avglen);
		if (r[N_COLLISIONS] > 0 && avglen > 0)
			print(" (x{:.2f} & x{:.2f} colliding)",
			      (double) r[COLLIDING_LEN_MIN] / r[N_COLLISIONS] / avglen,
			      (double) r[COLLIDING_LEN_MAX] / r[N_COLLISIONS] / avglen);
		/* BAD_PROBE is a dict thread's: its denominator is the probes retired, not the DPs found */
		print(".  {:.2f}% probe failure.  {:.2f}% walk-robinhood.  {:.2f}% walk-noncolliding.  "
		      "{:.2f}% same-value.  {:.2f}% DP failure.  {:.2f} re-walked/collision\n",
		      r[N_PROBE] ? 100. * r[BAD_PROBE] / r[N_PROBE] : 0., 100. * r[BAD_WALK_ROBINHOOD] / ndp,
		      100. * r[BAD_WALK_NONCOLLIDING] / ndp, 100. * r[BAD_COLLISION] / ndp,
		      100. * r[BAD_DP] / ndp,
		      r[N_COLLISIONS] ? (double) r[N_MEASURE] / r[N_COLLISIONS] : 0.);

		/*
		 * What the round actually routed.  A round closes on points FOUND, so a run whose queues
		 * overflow burns rounds against a nearly empty dictionary and still reports them complete: the
		 * points that reached a shard, and the share of those found, is the number to read.
		 */
		print("            ROUTED  {} DP/s found --> {} DP/s inserted ({:.2f}% reached a shard, "
		      "{} probe/s per dict thread).  dict load {:.2f}/slot\n",
		      human_format((double) ndp / delta), human_format((double) r[N_PROBE] / delta),
		      100. * r[N_PROBE] / ndp, human_format((double) r[N_PROBE] / dicts / delta),
		      (double) r[N_PROBE] / params.w);
	}

	if (r[DROP_COLL])
		print("            DROPPED  {} candidate(s): a collision queue was full\n", r[DROP_COLL]);
	if (r[REC_ROUTER + ROUTER_STALL_OUT] | r[REC_ROUTER + ROUTER_STALL_IN])
		print("            STALLED  {} blocks held back for a full destination / {} receives left "
		      "unposted for want of a free block\n", r[REC_ROUTER + ROUTER_STALL_OUT],
		      r[REC_ROUTER + ROUTER_STALL_IN]);
	if (r[REC_ROUTER + ROUTER_PUSHED] != r[REC_ROUTER + ROUTER_POPPED])
		print("            ACCOUNTING BROKEN: {} points pushed, {} delivered\n",
		      r[REC_ROUTER + ROUTER_PUSHED], r[REC_ROUTER + ROUTER_POPPED]);

	u64 E_i = distinct_collisions_estimation(round.hll);
	u64 E = distinct_collisions_estimation(all.hll);
	print("            #distinct coll (this i / total) {:.02f}*w / 2^{:.2f}\n", (double) E_i / params.w,
	      std::log2((double) (E ? E : 1)));
	print("\n");
	fflush(stdout);
}

/* the last line: found, or gave up after --nrounds versions of the function */
static void done(const Params &params, u64 nround, bool found, double seconds)
{
	if (found)
		print("Completed in {:.2f}s\n", seconds);
	else
		print("Gave up after {} version(s) of the function ({:.2f}s)\n", nround, seconds);
	fflush(stdout);
	(void) params;
}


/**************************** the service thread and the epilogue ***************************/

/* the node's record as it stands: its threads' tallies, then what its Router did */
static void node_snapshot(const Shared &shared, const u64 *stats, u64 *cur)
{
	for (int k = 0; k < REC_FOUND; k++)
		cur[k] = 0;
	for (size_t t = 0; t < shared.tally.size(); t++)
		for (int c = 0; c < N_COUNTERS; c++)
			cur[c] += shared.tally[t].ctr[c];
	for (int k = 0; k < ROUTER_STATS_SIZE; k++)
		cur[REC_ROUTER + k] = stats[k];
}

/*
 * Thread 0's round: turn the Router until the node has sent and received everything, and every 256th
 * turn serve the control channel -- which is what brings the end of the round in and lets the walkers
 * close.  The tallies it reads mid-round are read without synchronisation, hence approximate.
 */
static void service_round(Router_thread &rt, const Params &params, Shared &shared, Control &control, u64 *stats)
{
	u64 cur[REC_FOUND];
	u64 turns = 0;
	while (not Router_Test_quiescent(rt)) {
		Router_Progress(rt);
		turns += 1;
		if ((turns & 0xff) != 0)
			continue;
		Router_Stats(stats, rt);
		node_snapshot(shared, stats, cur);
		control.service(cur, shared);
		if (not params.verbose)
			continue;
		double now = wtime();
		if (now - control.last_display < params.ping_delay)
			continue;
		control.last_display = now;
		display(rt, params, control.reported, now - control.round_start, shared.nround);
	}
	Router_Stats(stats, rt);                       /* Router_Reset would clear them */
	node_snapshot(shared, stats, cur);
	control.service(cur, shared);                  /* the last turn: rank 0 drains the round's reports */
}

/*
 * Thread 0's end of round, once the Router is reset: the node's record -- its tallies, its Router stats
 * and the golden pair it may hold -- goes into one Allgather, and every node reads the same verdict out
 * of it.  The lowest rank that found a pair provides the answer, written over the node's own golden pair:
 * after the epilogue, found and golden[] are the verdict, the same on every node.  Control never reports
 * that flip as a solution of this node's, because no control turn follows an epilogue that sets found.
 * Then the next round's function, drawn from the PRNG every rank holds a copy of, so it travels on no wire.
 */
static void epilogue(const Router_thread &rt, const Params &params, Shared &shared, Control &control,
                     const u64 *stats, u64 *records, PRNG &prng, u64 out_mask)
{
	double delta = wtime() - control.round_start;
	const int nodes = Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL);
	const int rank = Router_rank(rt, ROUTER_GLOBAL);
	u64 rec[REC_WORDS] = {};
	node_snapshot(shared, stats, rec);
	if (shared.found.load_acquire()) {
		rec[REC_FOUND] = 1;
		rec[REC_I] = shared.golden[0];
		rec[REC_X0] = shared.golden[1];
		rec[REC_X1] = shared.golden[2];
	}
	MPI_Allgather(rec, REC_WORDS, MPI_UINT64_T, records, REC_WORDS, MPI_UINT64_T, params.mpi_comm);

	u64 sum[REC_WORDS] = {};
	int winner = -1;                               /* the lowest rank with a pair: its pair becomes everyone's */
	for (int d = 0; d < nodes; d++) {
		const u64 *q = records + (size_t) d * REC_WORDS;
		for (int k = 0; k < REC_FOUND; k++)
			sum[k] += q[k];
		if (q[REC_FOUND] && winner < 0)
			winner = d;
	}
	if (winner >= 0) {
		const u64 *q = records + (size_t) winner * REC_WORDS;
		shared.golden[0] = q[REC_I];
		shared.golden[1] = q[REC_X0];
		shared.golden[2] = q[REC_X1];
		shared.found = 1;
	}

	control.round.collect(shared.hll);             /* every walker's registers, merged and zeroed */
	control.round.reduce(params.mpi_comm, rank);

	shared.nround += 1;
	bool stop = shared.found || shared.nround >= params.max_versions;
	if (rank == 0) {
		for (int k = 0; k < REC_FOUND; k++)
			control.total[k] += sum[k];
		control.all.fold(control.round);
		if (params.verbose)
			round_report(rt, params, sum, control.total, control.round, control.all, delta, shared.nround);
	}

	/* the next round is thread 0's to set up: the barrier that follows publishes all of it */
	shared.round_over = 0;
	for (size_t r = 0; r < shared.chan.size(); r++)
		shared.chan[r].done = 0;
	if (stop)
		return;
	shared.header.i = prng.rand() & out_mask;
	shared.header.root_seed = prng.rand();
	control.begin_round();
}


/******************************** the engine ********************************/

/*
 * The search itself: one OpenMP team per node -- thread 0 the Router's service thread and the control
 * channel's end, the next R its dict threads, the rest its walkers -- running one version of the mixing
 * function per round.  Returns the version and the two colliding points, the same on every node, or
 * nothing if --nrounds ran out.
 */
template <class Wrapper>
optional<tuple<u64,u64,u64>> run(const Wrapper &wrapper, u64 nbytes_memory, const Options &opts, PRNG &prng)
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
		errx(1, "pcs: the ranks do not hold the same problem instance (self-test mismatch)");

	if (params.verbose)
		banner(wrapper, params, prng.seed);

	/* the team: thread 0 the service, then the dict threads, then the walkers; the Router counts the roles */
	const int R = params.dicts_per_node;
	const int n_threads = 1 + R + params.producers_per_node;
	Shared shared(n_threads, R);
	shared.header.i = prng.rand() & wrapper.out_mask;
	shared.header.root_seed = prng.rand();
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
		const int me = Router_rank(rt, ROUTER_NODE);           /* among this node's threads of my role */

		/* every per-thread object is built here, once the Router has pinned its thread: first touch */
		PcsDict dict(params.jbits, (role == ROUTER_CONSUMER) ? params.w_shard : 0);
		std::vector<u8> hll;                        /* a walker's distinct-collision registers */
		std::vector<int> recv_of;                   /* which dict thread each walker resolves for */
		std::vector<int> n_walkers;                 /* ... and how many walkers each of them has */
		std::vector<u64> records;                   /* thread 0: the nodes' records, from the Allgather */
		std::vector<u64> stats;                     /* thread 0: the node's Router tallies for the round */
		std::unique_ptr<Control> control;           /* thread 0: its end of the control channel */

		if (role == ROUTER_CONSUMER) {
			shared.group[me] = Router_group(rt);
			shared.chan[me].q = std::make_unique<CollisionQueue>(params.coll_queue_capacity);
		}
		if (role == ROUTER_PRODUCER) {
			shared.group[Router_size(rt, ROUTER_CONSUMER, ROUTER_NODE) + me] = Router_group(rt);
			hll.assign(HLL_REGISTERS, 0);
			shared.hll[tid] = hll.data();
		}
		if (role == ROUTER_SERVICE) {
			records.assign((size_t) Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL) * REC_WORDS, 0);
			stats.assign(ROUTER_STATS_SIZE, 0);
			control = std::make_unique<Control>(params, rt);
			control->begin_round();
		}

		#pragma omp barrier    /* the groups, the queues and the registers are the team's to read now */

		/*
		 * Which dict thread this walker resolves for.  Every thread runs the same deterministic pass
		 * over the published groups; thread 0 runs it to check that no shard was left without a walker,
		 * which would silently drop every collision it found.
		 */
		int chan = me;                 /* the channel this thread works: a dict thread's own, a walker's assigned */
		if (role != ROUTER_CONSUMER) {
			recv_of.assign(Router_size(rt, ROUTER_PRODUCER, ROUTER_NODE), -1);
			n_walkers.assign(Router_size(rt, ROUTER_CONSUMER, ROUTER_NODE), 0);
			assign_consumers(rt, shared.group.data(), recv_of.data(), n_walkers.data());
		}
		if (role == ROUTER_PRODUCER)
			chan = recv_of[me];
		if (role == ROUTER_SERVICE) {
			int lo = n_walkers[0];
			int hi = n_walkers[0];
			for (size_t r = 1; r < n_walkers.size(); r++) {
				lo = std::min(lo, n_walkers[r]);
				hi = std::max(hi, n_walkers[r]);
			}
			if (lo == 0) {
				warnx("pcs: rank %d has a dictionary shard with no walker to resolve for it, so "
				      "every collision it finds would be dropped.  Raise --producers-per-node, "
				      "lower --dicts-per-node, or widen a Router group (--group, --cache-level)",
				      Router_rank(rt, ROUTER_GLOBAL));
				MPI_Abort(params.mpi_comm, 1);
			}
			if (params.verbose)
				print("PCS: {} to {} walker(s) per dictionary shard, {} group(s) on the node\n", lo,
				      hi, Router_num_groups(rt));
		}

		for (;;) {
			for (int c = 0; c < N_COUNTERS; c++)
				ctr[c] = 0;

			if (role == ROUTER_SERVICE)
				service_round(rt, params, shared, *control, stats.data());
			else if (role == ROUTER_CONSUMER)
				dict_round(rt, params, shared, dict, ctr, chan);
			else
				walker_round(rt, wrapper, params, shared, ctr, hll.data(), chan);

			Router_Reset(rt);      /* its own team barriers and MPI_Barrier: every tally is written by now */

			if (role == ROUTER_SERVICE)
				epilogue(rt, params, shared, *control, stats.data(), records.data(), prng, wrapper.out_mask);

			#pragma omp barrier    /* the verdict and the next round's header are thread 0's to publish */
			if (shared.found || shared.nround >= params.max_versions)
				break;
		}
		if (role == ROUTER_SERVICE)
			control->shutdown();
	}

	if (params.verbose)
		done(params, shared.nround, shared.found, wtime() - t_start);
	if (not shared.found)
		return nullopt;
	return optional(tuple(shared.golden[0], shared.golden[1], shared.golden[2]));
}


/******************************** the entry points ********************************/

/* find x0 != x1 with f(x0) == f(x1) and is_good_pair(x0, x1) */
template <class Problem>
optional<pair<u64, u64>> collision_search(const Problem &pb, u64 nbytes_memory, const Options &opts, PRNG &prng)
{
	CollisionWrapper<Problem> wrapper(pb);
	auto collision = run(wrapper, nbytes_memory, opts, prng);
	if (not collision)
		return nullopt;

	auto [i, a, b] = *collision;
	auto [x0, x1] = wrapper.unmix(i, a, b);
	assert(x0 != x1);
	assert(pb.f(x0) == pb.f(x1));
	assert(pb.is_good_pair(x0, x1));
	return optional(pair(x0, x1));
}

/* find x0, x1 with f(x0) == g(x1) and is_good_pair(x0, x1) */
template <class Problem>
optional<pair<u64, u64>> claw_search(const Problem &pb, u64 nbytes_memory, const Options &opts, PRNG &prng)
{
	u64 x0 = 0;
	u64 x1 = 0;
	bool found = false;

	/* distinct wrapper types, so each branch carries the whole search */
	if (pb.n == pb.m) {
		EqualSizeClawWrapper<Problem> wrapper(pb);
		auto claw = run(wrapper, nbytes_memory, opts, prng);
		if (claw) {
			auto [i, a, b] = *claw;
			std::tie(x0, x1) = wrapper.unmix(i, a, b);
			found = true;
		}
	} else if (pb.n < pb.m) {
		LargerRangeClawWrapper<Problem> wrapper(pb);
		auto claw = run(wrapper, nbytes_memory, opts, prng);
		if (claw) {
			auto [i, a, b] = *claw;
			std::tie(x0, x1) = wrapper.unmix(i, a, b);
			found = true;
		}
	} else {
		errx(1, "pcs: a domain larger than the range is not supported");
	}

	if (not found)
		return nullopt;
	assert((x0 & make_mask(pb.n)) == x0);
	assert((x1 & make_mask(pb.n)) == x1);
	assert(pb.f(x0) == pb.g(x1));
	assert(pb.is_good_pair(x0, x1));
	return optional(pair(x0, x1));
}

}
#endif
