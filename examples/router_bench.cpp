#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <vector>
#include <cstdio>
#include <inttypes.h>

#include "tools.hpp"
#include "router/router.hpp"
#include "router_driver.hpp"

/*
 * The Router's pure throughput benchmark: producers push as fast as they can, one counter increment per point (a
 * producer computes, it does not idle, and an idling core is clocked down), to uniform, skewed or node-local
 * destinations picked by a multiply-high fan-out (no PRNG, no modulo: nothing but the router's own cost sits on
 * the producer's critical path); consumers pop the points one at a time and fold each with an add and a XOR; the
 * service thread runs until quiescent.  Prints, per round, the aggregate rate of points routed (delivered to a
 * consumer), then per node the push and pop rates, the network traffic, the cost of a point on a producer (the
 * destination pick included), who waited -- the producers' and the consumers' shares of the round, the service
 * thread's busy share and the Router's verdict -- and the XOR of every fold over the consumers and the nodes: it
 * keeps the computation from being optimised away, and it depends on the points alone, not on their order or
 * their route.  Before the rounds, the raw benchmark: the producers
 * run that same loop for a second with Router_Push taken out of it, so the difference between the raw rate and
 * the routed rate is the router and nothing else.
 */
using namespace mitm;

std::vector<double> ns_per_point;   /* per producer: a point's cost, the destination pick included */
u64 xor_hash = 0;                   /* the node's consumers' folds, folded; the raw run's outputs */
double raw_rate = 0;                /* points/s the node's producers generate without pushing, summed */

/* Knuth's odd 64-bit approximation of 2^64/phi: one multiply spreads a monotonic counter over the full word
 * (Fibonacci/Weyl hashing).  Its *high* bits are what mix -- a multiplication's low bits only ever see the
 * low bits of the multiplicand, so a raw counter's low 32 bits would stay sequential and every point would
 * land on destination 0 for as long as the counter takes to cross 2^32/F, which a benchmark round never does. */
static constexpr u64 FIB_MULT = 0x9e3779b97f4a7c15ull;

/* Where a point goes: the producer's fan-out, and the raw run's, so that the two cannot drift apart.  No modulo,
 * no PRNG: the counter is spread by FIB_MULT, its high 32 bits times the destination count, taken >> 32, map
 * [0, 2^32) onto [0, F) the way a 64-bit divq would but for one multiply and a shift (cf. Lemire's range
 * reduction) -- the same trick direct/producer.hpp uses on a hash's low word to pick a dict thread. */
struct Fanout {
	int F;                      /* destinations: the consumers over every node */
	int per;                    /* local-only: destinations on this node, [base, base + per) */
	int base;                   /* the first of them */
	bool local_only;            /* every point stays on the producer's node */

	Fanout(const Router_thread &rt, const RouterArgs &a)
	    : F(Router_size(rt, ROUTER_CONSUMER, ROUTER_GLOBAL)), per(Router_size(rt, ROUTER_CONSUMER, ROUTER_NODE)),
	      base(rt.node.rank * Router_size(rt, ROUTER_CONSUMER, ROUTER_NODE)), local_only(a.local_only)
	{
	}

	int operator()(u64 x) const
	{
		u32 key = (u32) ((x * FIB_MULT) >> 32);
		return local_only ? base + (int) (((u64) per * key) >> 32) : (int) (((u64) F * key) >> 32);
	}
};

/* a producer runs the sending loop without Router_Push for a second: the generation rate the routing is measured
 * against.  Everything the producer does per point but the push itself, the destination pick included */
static void bench_raw(Router_thread &rt, const RouterArgs &a)
{
	Fanout fan(rt, a);
	u64 acc = 0;
	u64 n = 0;
	double t0 = wtime();
	double t;
	for (;;) {
		for (int j = 0; j < 1024; j++) {
			u64 x = n + (u64) j;       /* the simple counter: no PRNG on the producer's critical path */
			acc ^= x ^ (u64) fan(x);   /* consumed, or the destination pick is optimised away */
		}
		n += 1024;
		t = wtime();
		if (t >= t0 + 1)
			break;
	}
	#pragma omp atomic
	raw_rate += (double) n / (t - t0);
	#pragma omp atomic
	xor_hash ^= acc;
}

/* rank 0 prints the raw benchmark: the producers' aggregate generation rate over all nodes */
static void raw_report(const Router_thread &rt)
{
	double total;
	u64 folded;
	MPI_Reduce(&raw_rate, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&xor_hash, &folded, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
	if (rt.node.rank != 0)
		return;
	int nsend = Router_size(rt, ROUTER_PRODUCER, ROUTER_GLOBAL);
	fmt::print("raw: {} producers generate {} pts/s without pushing ({:.1f} ns/point) | xor {:016x}\n",
	           nsend, human_format((u64) total), 1e9 * nsend / total, folded);
}

static void bench_producer(Router_thread &rt, const RouterArgs &a, double t_end)
{
	Fanout fan(rt, a);
	u64 n = 0;
	double t0 = wtime();
	for (;;) {
		for (int j = 0; j < 1024; j++) {
			Router_Push(n, n, fan(n), rt);   /* the counter is both the routing key and the payload */
			n += 1;
		}
		if (wtime() >= t_end)
			break;
	}
	double busy = wtime() - t0;
	Router_Close(rt);
	ns_per_point[rt.index] = 1e9 * busy / (double) n;
}

static void bench_consumer(Router_thread &rt)
{
	u64 x = 0;
	for (;;) {
		u64 a, b;
		if (not Router_Pop(&a, &b, rt)) {
			if (Router_Test_drained(rt))
				break;
			cpu_relax();
			continue;
		}
		x ^= a + b;   /* no murmur128: an add and a XOR, so the fold is not the bottleneck either */
	}
	#pragma omp atomic
	xor_hash ^= x;
}

/* rank 0 prints the round: the aggregate routed rate, sums over the nodes, the per-node spread of the push
 * rate, the folded hash */
static void bench_report(const Router_thread &rt, const RouterArgs &a, int round, double elapsed)
{
	u64 st[ROUTER_STATS_SIZE];
	Router_Stats(st, rt);
	double mine[3] = {(double) st[ROUTER_PUSHED] / elapsed, 0, elapsed};
	for (int s = 0; s < a.producers; s++)
		mine[1] += ns_per_point[s] / a.producers;
	u64 tot[ROUTER_STATS_SIZE];
	u64 folded;                         /* the nodes' xor_hash, folded */
	double lo[3];
	double hi[3];
	double sum[3];
	MPI_Reduce(st, tot, ROUTER_STATS_SIZE, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&xor_hash, &folded, 1, MPI_UINT64_T, MPI_BXOR, 0, MPI_COMM_WORLD);
	MPI_Reduce(mine, lo, 3, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(mine, hi, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(mine, sum, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rt.node.rank != 0)
		return;
	int P = Router_size(rt, ROUTER_SERVICE, ROUTER_GLOBAL);
	double t = hi[2];
	double send_wait;                   /* share of the round the producers waited for a block, over the nodes */
	double recv_wait;                   /* share the consumers had nothing to pop */
	double busy;                        /* share the service threads spent in turns that moved something */
	const char *limit = Router_Bottleneck(tot, rt, &send_wait, &recv_wait, &busy);
	fmt::print("round {}: {:.2f}s | routed {} pts/s | per node: push {}/s ({:.0f}-{:.0f} M/s) pop {}/s "
	           "net {} pts/s {:.2f} GB/s {} msgs/s {} blocks/s | {:.1f} ns/point | producers wait {:.0f}% "
	           "consumers {:.0f}% service {:.0f}% busy --> {} ({} turns/s) | xor {:016x}\n",
	           round, t, human_format((u64) (tot[ROUTER_POPPED] / t)),
	           human_format((u64) (tot[ROUTER_PUSHED] / t / P)), lo[0] / 1e6, hi[0] / 1e6,
	           human_format((u64) (tot[ROUTER_POPPED] / t / P)),
	           human_format((u64) (tot[ROUTER_SENT] / t / P)),
	           (double) tot[ROUTER_BYTES_SENT] / t / P / 1e9,
	           human_format((u64) (tot[ROUTER_MSGS_SENT] / t / P)),
	           human_format((u64) (tot[ROUTER_BLOCKS] / t / P)), sum[1] / P, 100. * send_wait,
	           100. * recv_wait, 100. * busy, limit,
	           human_format((u64) (tot[ROUTER_TURNS] / t / P)), folded);
	if (tot[ROUTER_PUSHED] != tot[ROUTER_POPPED])
		fmt::print("  ACCOUNTING BROKEN: pushed {} != popped {}\n", tot[ROUTER_PUSHED], tot[ROUTER_POPPED]);
}

int main(int argc, char **argv)
{
	int provided;
	MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
	if (provided < MPI_THREAD_FUNNELED)
		errx(1, "MPI_THREAD_FUNNELED not provided");
	
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	RouterArgs a;
	router_parse(argc, argv, a);
	a.opts.verbose = a.opts.verbose && rank == 0;
	int S = a.producers;
	int R = a.consumers;

	ns_per_point.assign(S, 0);
	double t0 = 0;

	#pragma omp parallel num_threads(1 + S + R)
	{
		int tid = omp_get_thread_num();
		int role = (tid == 0) ? ROUTER_SERVICE : (tid <= R ? ROUTER_CONSUMER : ROUTER_PRODUCER);
		Router_thread rt = Router_Init(role, MPI_COMM_WORLD, 42, &a.opts);
		if (role == ROUTER_PRODUCER)
			bench_raw(rt, a);
		#pragma omp barrier
		if (tid == 0)
			raw_report(rt);
		for (int round = 0; round < a.rounds; round++) {
			if (tid == 0) {
				xor_hash = 0;
				t0 = wtime();
			}
			#pragma omp barrier
			if (tid == 0) {
				while (not Router_Test_quiescent(rt))
					Router_Progress(rt);
			} else if (role == ROUTER_CONSUMER) {
				bench_consumer(rt);
			} else {
				bench_producer(rt, a, t0 + a.seconds);
			}
			#pragma omp barrier               /* the bench's own: bench_report reads ns_per_point and xor_hash */
			if (tid == 0)
				bench_report(rt, a, round, wtime() - t0);
			Router_Reset(rt);
		}
	}
	MPI_Finalize();
	return EXIT_SUCCESS;
}
