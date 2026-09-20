#ifndef MITM_PCS_SHARED
#define MITM_PCS_SHARED

#include <mpi.h>
#include <mutex>
#include <atomic>
#include <memory>
#include <vector>
#include <cmath>
#include <cassert>
#include <strings.h>          // ffsll, for the HyperLogLog

#include "tools.hpp"
#include "router/router.hpp"
#include "pcs/params.hpp"

/*
 * What a rank's threads share: the tallies, the queues that carry a dictionary hit from the dict thread
 * that found it to the walkers that resolve it, the round's header and its verdict.
 */

namespace mitm::pcs {

/* one thread's tallies, on a cache line of its own: written by the owner, read by thread 0 */
struct alignas(64) Tally {
	u64 ctr[N_COUNTERS];                   /* indexed by enum counter */
};


/****************************** collision queue *******************************/

/* what a dictionary hit is worth to a walker: two chains that end at the same distinguished point */
struct CollisionCandidate {
	u64 i;                  /* mixing function version; a walker asserts it is the round's */
	u64 seed0;              /* the incoming point: its chain index, ... */
	u64 len0_maybe;         /* ... its trail length; 0 == it saturated on the wire, the walker re-walks it */
	u64 end;                /* ... and its endpoint, in full: what a re-walked trail must reach */
	u64 seed1;              /* the point that was in the slot: its chain index, ... */
	u64 len1_maybe;         /* ... and its length; 0 == saturated in the dictionary, the walker re-walks it */
};

/*
 * Bounded, mutex-protected queue of dictionary hits: ONE per dict thread, pushed by that dict thread and
 * popped by the walkers assigned to it.  Candidates cross it in runs, never one at a time, so that the
 * lock and the `count` line are touched once per run.  Full == the run is truncated and the dict thread
 * tallies the rest: a walker blocked on a full queue could be one the transport is waiting for.
 */
class CollisionQueue {
	std::mutex mtx;                         /* what orders the several walkers of one queue */
	std::vector<CollisionCandidate> buf;    /* the ring itself */
	size_t head = 0;                        /* next to pop */
	size_t tail = 0;                        /* next to push */
	Atomic<size_t> count;                    /* probed without the lock on the hot path */

public:
	CollisionQueue(size_t capacity) : buf(capacity < 1 ? 1 : capacity), count(0) {}

	/* Relaxed on purpose: a stale answer costs a walker one wasted pop() at worst. */
	bool is_empty() const
	{
		return count == 0;
	}

	/* the dict thread's side.  Returns how many of the run fitted; the caller tallies the rest. */
	size_t push_bulk(const CollisionCandidate *in, size_t n)
	{
		std::lock_guard<std::mutex> lock(mtx);
		size_t room = buf.size() - count;
		size_t k = (room < n) ? room : n;
		for (size_t t = 0; t < k; t++) {
			buf[tail] = in[t];
			if (++tail == buf.size())
				tail = 0;
		}
		count.fetch_add(k, std::memory_order_release);
		return k;
	}

	/* a walker's side.  Several walkers of the same dict thread may be here at once. */
	size_t pop_bulk(CollisionCandidate *out, size_t max)
	{
		std::lock_guard<std::mutex> lock(mtx);
		size_t avail = count;
		size_t k = (avail < max) ? avail : max;
		for (size_t t = 0; t < k; t++) {
			out[t] = buf[head];
			if (++head == buf.size())
				head = 0;
		}
		count.fetch_sub(k, std::memory_order_release);
		return k;
	}
};


/******************************** HyperLogLog *********************************/

static constexpr int HLL_REGISTERS = 0x10000;   /* one per value of the top 16 bits of a pair's hash */

/*
 * How many DISTINCT collisions a round found, which is what says whether the parameters are any good:
 * one HyperLogLog per walker, plain u8, written by its owner alone and merged by thread 0 once the round
 * is over.  None of this is part of the attack.
 */

/* record the collision (x0, x1): raise its register to the pair's rho, never lower it */
static inline void hll_record(u8 h[HLL_REGISTERS], u64 x0, u64 x1)
{
	u64 hash = murmur128(x0, x1);
	u64 idx = hash >> 48;
	u8 rho = (u8) ffsll(hash);
	if (h[idx] < rho)
		h[idx] = rho;
}

/* fold `src` into `dst`: the estimate below only ever needs the max of each register */
static inline void hll_merge(u8 dst[HLL_REGISTERS], const u8 src[HLL_REGISTERS])
{
	for (int k = 0; k < HLL_REGISTERS; k++)
		if (dst[k] < src[k])
			dst[k] = src[k];
}

/* the estimate, on merged registers (one round's, or the controller's all-time) */
static inline u64 distinct_collisions_estimation(const u8 h[HLL_REGISTERS])
{
	double acc = 0;
	double alpha = 0.7213 / (1 + 1.079 / HLL_REGISTERS);
	for (int i = 0; i < HLL_REGISTERS; i++)
		acc += std::ldexp(1.0, -(int) h[i]);       /* 2^-h[i]; (1 << h[i]) overflows past 31 */
	double E = alpha * ((double) HLL_REGISTERS * HLL_REGISTERS) / acc;
	if (E >= 2.5 * HLL_REGISTERS)
		return E;
	int V = 0;                                         /* low cardinality: linear counting instead */
	for (int i = 0; i < HLL_REGISTERS; i++)
		if (h[i] == 0)
			V += 1;
	if (V == 0)
		return E;
	return HLL_REGISTERS * std::log((double) HLL_REGISTERS / V);
}

/*
 * A round's distinct collisions: the walkers' registers merged by thread 0 once the round is over, the
 * nodes' MAX-reduced to rank 0, the rounds' folded into the controller's all-time registers.
 */
struct RoundStats {
	u8 hll[HLL_REGISTERS];                 /* the merged registers */

	RoundStats()
	{
		reset();
	}

	void reset()
	{
		for (int k = 0; k < HLL_REGISTERS; k++)
			hll[k] = 0;
	}

	/* register-wise max, which is why the merge is exact whatever the order */
	void fold(const RoundStats &o)
	{
		hll_merge(hll, o.hll);
	}

	/* merge every walker's registers and zero them for the next round.  Thread 0, the round over */
	void collect(const vector<u8 *> &registers)
	{
		for (size_t t = 0; t < registers.size(); t++) {
			if (registers[t] == NULL)
				continue;
			hll_merge(hll, registers[t]);
			for (int k = 0; k < HLL_REGISTERS; k++)
				registers[t][k] = 0;
		}
	}

	/* MAX to rank 0.  Blocks; every rank reaches it after the epilogue's Allgather */
	void reduce(MPI_Comm comm, int rank)
	{
		if (rank == 0)
			MPI_Reduce(MPI_IN_PLACE, hll, HLL_REGISTERS, MPI_UINT8_T, MPI_MAX, 0, comm);
		else
			MPI_Reduce(hll, NULL, HLL_REGISTERS, MPI_UINT8_T, MPI_MAX, 0, comm);
	}
};


/******************************** shared state ********************************/

/* one dict thread's channel to its walkers, on a line of its own: the queue, and whether it is final */
struct alignas(64) CollChannel {
	std::unique_ptr<CollisionQueue> q;     /* the candidates, allocated by the dict thread once pinned */
	Atomic<u32> done{0};                    /* 1 once the dict thread has left the round: nothing more comes */
};

/* what a rank's threads share.  Built before the team; every per-thread object is its owner's own */
struct Shared {
	std::vector<Tally> tally;              /* per OpenMP thread id */
	std::vector<int> closeness;            /* per walker, by local rank, its Router_closeness to every dict thread: C entries each */
	std::vector<u8 *> hll;                 /* each walker's own registers; NULL for the other roles */
	std::vector<CollChannel> chan;         /* per dict thread */
	Atomic<u32> round_over{0};             /* the controller closed the round: the walkers stop walking */
	Header header;                         /* the round's function; thread 0 draws it, the barrier publishes it */
	std::mutex golden_mtx;                 /* serialises set_golden */
	Atomic<u32> found{0};                  /* 1 once golden[] holds a pair: this node's, then the verdict once thread 0 has run the epilogue */
	u64 golden[3] = {};                    /* the version and the two colliding points found on this node; the epilogue overwrites them with the answer, the same on every node */
	u64 nround = 0;                        /* rounds run to their end, thread 0's count */

	/* the team's n_threads: the service, the R dict threads, then the walkers */
	Shared(int n_threads, int R)
		: tally(n_threads), closeness((size_t) (n_threads - 1 - R) * R, 0), hll(n_threads, NULL), chan(R) {}

	/* a walker's golden pair; the first one wins */
	void set_golden(u64 i, u64 x0, u64 x1)
	{
		std::lock_guard<std::mutex> lock(golden_mtx);
		if (found)
			return;
		golden[0] = i;
		golden[1] = x0;
		golden[2] = x1;
		found.store_release(1);
	}
};


/**************************** walkers to dict threads ***************************/

/*
 * Which dict thread each walker resolves for: the closest in the machine's hierarchy, the least loaded among the
 * closest, the lowest index among those, the walkers taken in order.  A walker thus serves the dict thread of
 * its own core, cache or NUMA node, so the queue and the two threads touching it stay together; unpinned, every
 * closeness is 0 and this is a plain round-robin.  `closeness` is the Shared's: each walker's Router_closeness
 * to every dict thread, C entries per walker by local rank, published before the startup barrier.
 * Deterministic from those alone, so every thread runs it and gets the same answer.
 */
static void assign_consumers(const Router_thread &rt, const int closeness[], int consumer_of[], int n_walkers[])
{
	const int C = Router_size(rt, ROUTER_CONSUMER, ROUTER_NODE);
	const int P = Router_size(rt, ROUTER_PRODUCER, ROUTER_NODE);
	for (int r = 0; r < C; r++)
		n_walkers[r] = 0;
	for (int s = 0; s < P; s++) {
		int best = 0;
		int best_close = closeness[s * C];
		for (int r = 1; r < C; r++) {
			int close = closeness[s * C + r];
			if (close > best_close || (close == best_close && n_walkers[r] < n_walkers[best])) {
				best = r;
				best_close = close;
			}
		}
		consumer_of[s] = best;
		n_walkers[best] += 1;
	}
}

}
#endif
