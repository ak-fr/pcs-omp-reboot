#ifndef MITM_DIRECT_PARAMS
#define MITM_DIRECT_PARAMS

#include <mpi.h>
#include <omp.h>
#include <err.h>

#include "tools.hpp"
#include "parameters.hpp"
#include "router/router.hpp"

/*
 * What a direct run derives once from the Options: the dictionary's slots, the domain and how many
 * FILL/PROBE rounds cover it, plus the per-thread counters and the epilogue's record layout.  The team's
 * shape is not here: the Router holds it, and a thread asks it (Router_size, Router_rank) once it has its handle.
 */

namespace mitm::direct {

/* the most points a dict thread may hold between Router_Pop and its shard: its ring is an array of that size */
static constexpr int MAX_PREFETCH = 64;

/* the two phases of a direct round: f fills the dictionary, then g probes it */
enum phase {FILL, PROBE};

/* the tag of every Router message on the communicator; nothing else on it carries this tag */
static constexpr int ROUTER_TAG = 1;


/********************************* parameters ********************************/

/*
 * Everything a run needs beyond the Options, derived once before the team and const from then on: the
 * dictionary's slots, the domain and how many rounds cover it.  Data only, and nothing about the team.
 */
struct Params : Options {
	u64 w;                                 /* slots in the whole, distributed dictionary */
	u64 w_shard;                           /* slots per shard */
	int n;                                 /* domain bits: a preimage is n-bit */
	u64 domain;                            /* 2^n */
	u64 per_round;                         /* entries a round inserts: fill * w, at most the domain */
	u64 n_rounds_full;                     /* rounds an exhaustive search needs: ceil(domain / per_round) */
	u64 n_rounds;                          /* rounds this run will do: n_rounds_full, capped by --nrounds */
	bool capped;                           /* --nrounds cut the round count: the search is not exhaustive */

	Params(const Options &o, u64 nbytes_memory, int n, int m) : Options(o), n(n)
	{
		int rank;                          /* this node, for the verbose gate */
		int n_nodes;                       /* what turns a per-node count into the team's */
		MPI_Comm_rank(mpi_comm, &rank);
		MPI_Comm_size(mpi_comm, &n_nodes);
		router.verbose = router.verbose && verbose;   /* the Router prints on rank 0 by itself */
		verbose = verbose && (rank == 0);
		if (n > 63)
			errx(1, "direct: a %d-bit preimage leaves an 8-byte slot no room for its occupancy bit (n <= 63)", n);
		if (m > 64)
			errx(1, "direct: %d-bit images do not fit a 64-bit key", m);
		if (dicts_per_node < 1)
			errx(1, "--dicts-per-node %d: at least one dictionary shard per node", dicts_per_node);
		int R = dicts_per_node;            /* dict threads per node: the Router's consumers */
		int S = producers_per_node;        /* producers per node: the Router's producers; 0 == fill the mask */
		int n_cpu = omp_get_num_procs();   /* the CPUs the rank may use: what the team fills when S == 0 */
		if (S == 0)
			S = n_cpu - 1 - R;
		if (S < 1)
			errx(1, "direct: no CPU left for a producer (%d available, 1 service thread, %d dict threads)",
			     n_cpu, R);
		producers_per_node = S;            /* resolved: the team run() builds, whose roles the Router counts */
		int n_dicts = n_nodes * R;         /* shards over all nodes: what the RAM budget is cut into */

		if (nbytes_memory == 0)
			errx(1, "the RAM budget per node must be given (and nonzero)");
		w_shard = (nbytes_memory * n_nodes / sizeof(u64)) / n_dicts;
		w = w_shard * n_dicts;
		if (w_shard == 0)
			errx(1, "RAM budget too small: %" PRIu64 " bytes/node cannot hold one slot per shard", nbytes_memory);
		if (not (fill > 0 && fill <= 0.9))
			errx(1, "--fill %.2f: the dictionary fill ratio must be in (0, 0.9]", fill);
		if (prefetch < 0 || prefetch > MAX_PREFETCH)
			errx(1, "--prefetch %d: a dict thread prefetches 0 to %d points ahead", prefetch, MAX_PREFETCH);
		domain = 1ull << n;
		per_round = fill * (double) w;
		if (per_round == 0)
			errx(1, "RAM budget too small: %.2f * %" PRIu64 " slots holds no entry", fill, w);
		if (per_round > domain)
			per_round = domain;
		n_rounds_full = (domain + per_round - 1) / per_round;
		n_rounds = n_rounds_full;
		capped = (max_versions < n_rounds);
		if (capped)
			n_rounds = max_versions;
	}
};


/********************************** counters **********************************/

/* every tally of the scheme; one u64[N_COUNTERS] per thread, written by its owner, summed by thread 0 */
enum counter {
	N_EVAL = 0,             /* producers: evaluations of f (FILL) or g (PROBE), one point pushed each */
	N_INSERT,               /* dict threads: entries inserted (FILL) */
	N_PROBE,                /* probes retired (PROBE) */
	N_COLLISIONS,           /* slots matching a probe: a collision f(x) == g(y), bar a check-bit false positive */
	N_COUNTERS
};

/* the round record a node contributes to the epilogue's MPI_Allgather: counters, Router stats, its golden pair */
enum record {
	REC_ROUTER = N_COUNTERS,                        /* ROUTER_STATS_SIZE words: the node's Router_Stats */
	REC_FOUND = N_COUNTERS + ROUTER_STATS_SIZE,     /* 1 if the node holds a golden pair */
	REC_X,                                          /* its x */
	REC_Y,                                          /* its y */
	REC_WORDS
};

}
#endif
