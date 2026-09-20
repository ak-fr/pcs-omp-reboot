#ifndef MITM_PCS_PARAMS
#define MITM_PCS_PARAMS

#include <mpi.h>
#include <omp.h>
#include <err.h>
#include <cmath>
#include <algorithm>

#include "tools.hpp"
#include "parameters.hpp"
#include "router/router.hpp"

/*
 * What a PCS run derives once from the Options, the round header it draws every round, its tallies and the
 * record its nodes exchange.  Data only, and const from the top of the run on.  The team's shape is not
 * here: the Router holds it, and a thread asks it (Router_size, Router_rank) once it has its handle.
 */

namespace mitm::pcs {

/* the tags this engine owns on the communicator: the Router's own, then the control channel's three */
static constexpr int ROUTER_TAG = 1;
static constexpr int TAG_REPORT = 2;
static constexpr int TAG_END_ROUND = 3;
static constexpr int TAG_SOLUTION = 4;

/* reports a node owes per round beside the one it owes every ping_delay: what paces them by volume */
static constexpr int REPORTS_PER_ROUND = 64;


/********************************* parameters ********************************/

struct Params : Options {
	u64 w;                                 /* slots in the whole, distributed dictionary */
	u64 w_shard;                           /* slots per shard */
	int n;                                 /* domain bits of the mixed function */
	int m;                                 /* range bits: the space the trails walk in */
	int jbits;                             /* bits of chain index shipped with a distinguished point */
	int lenbits;                           /* width of the trail length field beside it */
	u64 len_sat;                           /* its largest value: at least this long, exact length unknown */
	bool theta_auto;                       /* theta was chosen from alpha rather than given */
	double auto_theta;                     /* what alpha chooses, whether or not it was used */
	u64 threshold;                         /* x is a distinguished point iff x <= threshold */
	u64 dp_max_it;                         /* iterations a trail gets to reach a distinguished point */
	u64 points_per_version;                /* distinguished points a round collects before the controller closes it */
	u64 report_points;                     /* ... and how many of them one node's progress report is worth */
	bool capped;                           /* --nrounds was given: the run gives up after that many versions */

	Params(const Options &o, u64 nbytes_memory, int n, int m) : Options(o), n(n), m(m)
	{
		int rank;                          /* this node, for the verbose gate */
		int n_nodes;                       /* what turns a per-node count into the team's */
		MPI_Comm_rank(mpi_comm, &rank);
		MPI_Comm_size(mpi_comm, &n_nodes);
		router.verbose = router.verbose && verbose;   /* the Router prints on rank 0 by itself */
		verbose = verbose && (rank == 0);
		if (m > 64)
			errx(1, "pcs: %d-bit images do not fit a 64-bit trail endpoint", m);
		if (n > m)
			errx(1, "pcs: a %d-bit domain does not fit a %d-bit range: the trails have nowhere to walk", n, m);
		if (dicts_per_node < 1)
			errx(1, "--dicts-per-node %d: at least one dictionary shard per node", dicts_per_node);
		int R = dicts_per_node;            /* dict threads per node: the Router's consumers */
		int S = producers_per_node;        /* walkers per node: the Router's producers; 0 == fill the mask */
		int n_cpu = omp_get_num_procs();   /* the CPUs the rank may use: what the team fills when S == 0 */
		if (S == 0)
			S = n_cpu - 1 - R;
		if (S < 1)
			errx(1, "pcs: no CPU left for a walker (%d available, 1 service thread, %d dict threads)",
			     n_cpu, R);
		if (R > S)
			errx(1, "pcs: %d dictionary shards for %d walker(s): every shard needs a walker of its own "
			     "to resolve what it finds, so --dicts-per-node must not exceed --producers-per-node",
			     R, S);
		producers_per_node = S;            /* resolved: the team run() builds, whose roles the Router counts */
		int n_dicts = n_nodes * R;         /* shards over all nodes: what the RAM budget is cut into */

		if (nbytes_memory == 0)
			errx(1, "the RAM budget per node must be given (and nonzero)");
		w_shard = (nbytes_memory * n_nodes / sizeof(u64)) / n_dicts;
		w = w_shard * n_dicts;
		if (w_shard == 0)
			errx(1, "RAM budget too small: %" PRIu64 " bytes/node cannot hold one slot per shard", nbytes_memory);

		jbits = std::log2(10 * w) + 8;
		if (jbits > 56)
			errx(1, "dictionary too large: a %d-bit chain index leaves a slot no room for a length", jbits);
		lenbits = dp_lenbits ? dp_lenbits : 64 - jbits;
		if (lenbits < 1 || jbits + lenbits > 64)
			errx(1, "--dp-len-bits %d does not fit beside a %d-bit chain index", lenbits, jbits);
		len_sat = make_mask(lenbits);

		auto_theta = alpha * std::sqrt(std::ldexp((double) w, -n));
		theta_auto = (theta < 0);
		if (theta_auto)
			theta = std::min(auto_theta, 1.0);
		if (not (theta > 0 && theta < 1))
			errx(1, "--difficulty %.4f: the proportion of distinguished points must be in (0, 1) -- at 1 "
			     "every point is distinguished and no trail can start.  Use --engine direct", theta);
		threshold = std::pow(2, m) * theta;
		dp_max_it = 20 / theta;
		points_per_version = beta * w;
		if (points_per_version == 0)
			errx(1, "--beta %.2f: a round collecting no distinguished point never ends", beta);

		capped = (max_versions != 0xffffffffffffffffull);

		/* report by volume too: on a timer alone a short round overshoots points_per_version */
		report_points = points_per_version / ((u64) REPORTS_PER_ROUND * n_nodes);
		if (report_points < 1)
			report_points = 1;
	}
};


/*********************************** header **********************************/

/* the round's function: every node draws it from its own copy of the PRNG, so it travels on no wire */
struct Header {
	u64 i = 0;                             /* mixing function version */
	u64 root_seed = 0;                     /* chain j starts at root_seed + j * multiplier */
};


/********************************** counters **********************************/

/* every tally of the engine; one u64[N_COUNTERS] per thread, written by its owner, read by thread 0 */
enum counter {
	/* walkers */
	N_EVAL = 0,             /* evaluations of the mixing function, by walking or by resolving */
	N_DP,                   /* distinguished points found */
	N_POINTS_TRAILS,        /* sum of the lengths of the trails that reached a distinguished point */
	N_COLLISIONS,           /* collisions located */
	COLLIDING_LEN_MIN,      /* sum of the shorter length of each colliding pair */
	COLLIDING_LEN_MAX,      /* ... and of the longer one */
	N_MEASURE,              /* trails re-walked because their length had saturated */
	BAD_DP,                 /* trail gave up before a distinguished point, walked or re-walked to be measured */
	BAD_COLLISION,          /* the two trails "collide" on the same value */
	BAD_WALK_ROBINHOOD,     /* one trail is a suffix of the other */
	BAD_WALK_NONCOLLIDING,  /* dictionary false positive: the trails never meet */
	/* dict threads */
	N_PROBE,                /* dictionary probes retired */
	BAD_PROBE,              /* dictionary slot was empty, or held a different key */
	DROP_COLL,              /* candidate dropped: the collision queue was full */
	N_COUNTERS
};

/*
 * The record a node contributes to the epilogue's MPI_Allgather.  Its first REC_FOUND words are also the
 * progress report a node sends the controller, as deltas: the tallies, then what the transport did.
 */
enum record {
	REC_ROUTER = N_COUNTERS,                        /* ROUTER_STATS_SIZE words: the node's Router_Stats */
	REC_FOUND = N_COUNTERS + ROUTER_STATS_SIZE,     /* 1 if the node holds a golden pair */
	REC_I,                                          /* the version it was found in */
	REC_X0,                                         /* and the two colliding points */
	REC_X1,
	REC_WORDS
};

}
#endif
