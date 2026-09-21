#ifndef MITM_BENCHMARK
#define MITM_BENCHMARK

#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <type_traits>
#include <vector>

#include "tools.hpp"
#include "parameters.hpp"
#include "problem.hpp"
#include "router/router.hpp"
#include "direct/wrappers.hpp"

/*
 * How fast do a problem's functions iterate, aggregated over a node's threads and over the ranks, and per
 * producer?  What every driver's --benchmark runs; needs no RAM budget, only the Options.  Not part of the
 * library: nothing in ../include depends on it (see driver.hpp).  The dictionary and staging
 * benchmarks that used to live here were built on PCS's parameters and come back when PCS does;
 * router_bench measures the communication path on its own.  The routed pass below reuses direct/producer.hpp's
 * veval + hash + push body and router_bench's pure receive fold, to show what the Router itself costs a real
 * producer over the isolated rate above.
 */

namespace mitm {

/* the rate of N * vlen evaluations since `start`: min, max, mean and std over the ranks, and the per-producer
   share of the mean (what one of the n_threads producer threads contributes, contention included), printed
   by rank 0.  Collective */
static void display_stats(u64 N, double start, int vlen, int n_threads, MPI_Comm comm, int rank, int n_nodes)
{
	double rate = vlen * N / (wtime() - start);
	double rate_min = rate;
	double rate_max = rate;
	double rate_avg = rate;
	MPI_Allreduce(MPI_IN_PLACE, &rate_min, 1, MPI_DOUBLE, MPI_MIN, comm);
	MPI_Allreduce(MPI_IN_PLACE, &rate_max, 1, MPI_DOUBLE, MPI_MAX, comm);
	MPI_Allreduce(MPI_IN_PLACE, &rate_avg, 1, MPI_DOUBLE, MPI_SUM, comm);
	rate_avg /= n_nodes;
	double rate_std = (rate - rate_avg) * (rate - rate_avg);
	MPI_Allreduce(MPI_IN_PLACE, &rate_std, 1, MPI_DOUBLE, MPI_SUM, comm);
	rate_std /= n_nodes;
	rate_std = std::sqrt(rate_std);
	if (rank == 0)
		fmt::print("Benchmark. f/s: min {} max {} avg {} std {} | per-producer {}/s ({} threads)\n",
		           human_format(rate_min), human_format(rate_max), human_format(rate_avg),
		           human_format(rate_std), human_format(rate_avg / n_threads), n_threads);
}

/*
 * The routed pass, in two pushers: `vectorized` false runs the scalar one -- one evaluation, one
 * murmur64, one push per iteration, always pb.f (FILL's function for either wrapper) -- and true runs
 * direct/producer.hpp's own body verbatim, vlen inputs through wrapper.veval and vlen pushes at a time.
 * Either way, consumers fold what they pop the way router_bench's pure benchmark does, an add and a XOR, no
 * murmur128 and no dictionary.  So the gap between one of these rates and the matching isolated rate above
 * (scalar vs scalar, vector vs vector) is the Router's cost and nothing else.  Builds its own team for one
 * second -- 1 service + dicts_per_node consumers + producers_per_node producers per node, the same knobs the
 * direct engine sizes its team with -- then tears it down; needs no RAM budget, since nothing here is a
 * dictionary.  Collective
 */
template <bool vectorized, class Wrapper>
static void routed_benchmark(const Wrapper &wrapper, const Options &opts, int rank, int n_nodes)
{
	constexpr int vlen = Wrapper::vlen;
	constexpr int TAG = 99;
	constexpr double SECONDS = 1.0;

	int n_cpu = omp_get_num_procs();
	if (opts.dicts_per_node < 1)
		errx(1, "benchmark: --dicts-per-node must be at least 1");
	int R = opts.dicts_per_node;
	int S = opts.producers_per_node > 0 ? opts.producers_per_node : n_cpu - 1 - R;
	if (S < 1)
		errx(1, "benchmark: no CPU left for a producer (%d available, 1 service thread, %d consumers)", n_cpu, R);
	int n_threads = 1 + R + S;
	u64 n_recv = (u64) R * (u64) n_nodes;
	u64 mask = make_mask(wrapper.n);

	if (rank == 0)
		printf("Benchmarking routed %s implementation (%d processes x (1 service + %d consumer + %d producer))\n",
		       vectorized ? "vector" : "scalar", n_nodes, R, S);
	MPI_Barrier(opts.mpi_comm);

	/* a reduction clause only combines at the parallel region's own closing brace, not at a barrier written
	   inside it, and tid 0 reads xor_hash before that point below -- so fold consumers' shares by hand */
	std::vector<u64> xor_local(n_threads, 0);
	double t0 = 0;
	#pragma omp parallel num_threads(n_threads)
	{
		int tid = omp_get_thread_num();
		int role = (tid == 0) ? ROUTER_SERVICE : (tid <= R ? ROUTER_CONSUMER : ROUTER_PRODUCER);
		Router_thread rt = Router_Init(role, opts.mpi_comm, TAG, &opts.router);

		if (tid == 0)
			t0 = wtime();
		#pragma omp barrier

		if (role == ROUTER_SERVICE) {
			while (not Router_Test_quiescent(rt))
				Router_Progress(rt);
		} else if (role == ROUTER_CONSUMER) {
			u64 x = 0;
			for (;;) {
				u64 a, b;
				if (not Router_Pop(&a, &b, rt)) {
					if (Router_Test_drained(rt))
						break;
					cpu_relax();
					continue;
				}
				x ^= a + b;
			}
			xor_local[tid] = x;
		} else if constexpr (vectorized) {
			u64 x[vlen] __attribute__ ((aligned(sizeof(u64) * vlen)));
			u64 y[vlen] __attribute__ ((aligned(sizeof(u64) * vlen)));
			u64 base = (u64) rt.index << 40;      /* keeps producers from all walking the same inputs */
			u64 n = 0;
			double t_end = t0 + SECONDS;
			for (;;) {
				for (int j = 0; j < 256; j++) {
					for (int k = 0; k < vlen; k++)
						x[k] = (base + n + (u64) k) & mask;
					wrapper.veval(direct::FILL, x, y);
					for (int k = 0; k < vlen; k++) {
						u64 h = murmur64(y[k]);
						int dest = (int) (((h & 0xffffffffull) * n_recv) >> 32);
						Router_Push(h, x[k], dest, rt);
					}
					n += (u64) vlen;
				}
				if (wtime() >= t_end)
					break;
			}
			Router_Close(rt);
		} else {
			u64 base = (u64) rt.index << 40;      /* keeps producers from all walking the same inputs */
			u64 n = 0;
			double t_end = t0 + SECONDS;
			for (;;) {
				for (int j = 0; j < 256; j++) {
					u64 x = (base + n) & mask;
					u64 y = wrapper.pb.f(x);
					u64 h = murmur64(y);
					int dest = (int) (((h & 0xffffffffull) * n_recv) >> 32);
					Router_Push(h, x, dest, rt);
					n += 1;
				}
				if (wtime() >= t_end)
					break;
			}
			Router_Close(rt);
		}

		#pragma omp barrier
		if (tid == 0) {
			double elapsed = wtime() - t0;
			u64 st[ROUTER_STATS_SIZE];
			Router_Stats(rt, st);
			u64 tot[ROUTER_STATS_SIZE];
			u64 xor_hash = 0;              /* every consumer's slot wrote before the barrier above */
			for (int t = 0; t < n_threads; t++)
				xor_hash ^= xor_local[t];
			u64 folded;
			MPI_Reduce(st, tot, ROUTER_STATS_SIZE, MPI_UINT64_T, MPI_SUM, 0, opts.mpi_comm);
			MPI_Reduce(&xor_hash, &folded, 1, MPI_UINT64_T, MPI_BXOR, 0, opts.mpi_comm);
			if (rank == 0) {
				u64 nsend = (u64) S * (u64) n_nodes;
				fmt::print("Routed ({}): {} pts/s | {} f/s per producer | node-->{}B/s | xor {:016x}\n",
				           vectorized ? "vector" : "scalar", human_format((u64) (tot[ROUTER_POPPED] / elapsed)),
				           human_format((u64) (tot[ROUTER_PUSHED] / elapsed / nsend)),
				           human_format((u64) (tot[ROUTER_BYTES_SENT] / elapsed / n_nodes)), folded);
			}
		}
		Router_Reset(rt);
		Router_Finalize(rt);
	}
}

/*
 * Iterate the scalar functions 2^26 times per thread, then the vector ones 2^20 times per thread if
 * vlen > 1, on every thread of every rank, and print the aggregate rates.  A claw problem alternates
 * f and g, a collision problem iterates f alone.  Every thread walks its own dependent chain (the
 * shape a producer's enumeration and a trail both have), so this measures what concurrent producers
 * do to each other's cache and memory traffic, not just one thread in isolation.  The thread count
 * is producers_per_node, 0 meaning every CPU the affinity mask allows: the benchmark builds no team
 * of its own, so there is no service or dictionary thread to leave room for.  Each loop's combined
 * checksum is printed, and that is the only reason it survives -O3: nothing else here is
 * observable.  Collective
 */
template<typename Problem>
void benchmark(const Problem& pb, const Options &opts)
{
	constexpr bool claw = std::is_base_of<AbstractClawProblem, Problem>::value;
	int rank, n_nodes;
	MPI_Comm_rank(opts.mpi_comm, &rank);
	MPI_Comm_size(opts.mpi_comm, &n_nodes);

	int n_threads = opts.producers_per_node > 0 ? opts.producers_per_node : omp_get_num_procs();

	if (rank == 0)
		printf("Benchmarking scalar implementation (%d processes x %d threads)\n", n_nodes, n_threads);

	MPI_Barrier(opts.mpi_comm);

	u64 mask = make_mask(pb.n);
	u64 N = 1ull << 26;
	u64 checksum = 0;
	double start = wtime();
	#pragma omp parallel num_threads(n_threads) reduction(^:checksum)
	{
		u64 x1 = 1 + (u64) omp_get_thread_num();      /* a distinct chain per thread */
		for (u64 i = 0; i < N; i++) {
			if constexpr (claw)
				x1 = ((i & 1) ? pb.f(x1) : pb.g(x1)) & mask;
			else
				x1 = pb.f(x1) & mask;
		}
		checksum ^= x1;
	}
	display_stats(N * n_threads, start, 1, n_threads, opts.mpi_comm, rank, n_nodes);
	if (rank == 0)
		printf("  checksum %016" PRIx64 "\n", checksum);

	constexpr int vlen = Problem::vlen;
	if constexpr (vlen > 1) {
		if (rank == 0)
			printf("Benchmarking vector implementation (vlen=%d, %d processes x %d threads)\n",
			       vlen, n_nodes, n_threads);

		MPI_Barrier(opts.mpi_comm);

		u64 N2 = 1ull << 20;
		u64 vchecksum = 0;
		double start2 = wtime();
		#pragma omp parallel num_threads(n_threads) reduction(^:vchecksum)
		{
			u64 x[vlen] __attribute__ ((aligned(sizeof(u64) * vlen)));
			u64 z[vlen] __attribute__ ((aligned(sizeof(u64) * vlen)));
			bool choice[vlen];    /* what vfg selects, for a claw problem: half the lanes f, the others g */
			u64 base = (u64) omp_get_thread_num() * vlen;    /* a distinct chain per lane per thread */
			for (int i = 0; i < vlen; i++) {
				choice[i] = i & 1;
				x[i] = base + i;
			}
			for (u64 i = 0; i < N2; i++) {
				if constexpr (claw)
					pb.vfg(x, choice, z);
				else
					pb.vf(x, z);
				for (int j = 0; j < vlen; j++)
					x[j] = z[j] & mask;
			}
			for (int j = 0; j < vlen; j++)
				vchecksum ^= x[j];
		}
		display_stats(N2 * n_threads, start2, vlen, n_threads, opts.mpi_comm, rank, n_nodes);
		if (rank == 0)
			printf("  checksum %016" PRIx64 "\n", vchecksum);
	}

	MPI_Barrier(opts.mpi_comm);
	if constexpr (claw) {
		direct::ClawWrapper<Problem> wrapper(pb);
		routed_benchmark<false>(wrapper, opts, rank, n_nodes);
		if constexpr (vlen > 1)
			routed_benchmark<true>(wrapper, opts, rank, n_nodes);
	} else {
		direct::CollisionWrapper<Problem> wrapper(pb);
		routed_benchmark<false>(wrapper, opts, rank, n_nodes);
		if constexpr (vlen > 1)
			routed_benchmark<true>(wrapper, opts, rank, n_nodes);
	}
}
}
#endif
