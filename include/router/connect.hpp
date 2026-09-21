#ifndef MITM_ROUTER_CONNECT
#define MITM_ROUTER_CONNECT

#include "common.hpp"
#include "workers.hpp"                /* Router_Finalize asks a consumer the same question Router_Test_drained does */

namespace mitm {

/*
 * A producer gets its lines and its cache of blocks, a consumer its inbox, the service thread neither (an unused
 * inbox has one slot).  The object registers itself with the node, then holds the team's last barrier: this is
 * Router_Init's last stage, and the constructor runs nowhere else.  Past the barrier the service may read every
 * object, and thread 0 prints the measured layout.
 */
inline Router_thread::Router_thread(int role_, int index_, int cpu_, Router_node &rn)
	: swc(role_ == ROUTER_PRODUCER ? (Point *) router_alloc((size_t) rn.F * rn.swc_linesize * sizeof(Point)) : NULL),
	  swc_linesize(rn.swc_linesize), role(role_), index(index_),
	  cpu(cpu_), node(rn), closed(0),
	  inbox(role_ == ROUTER_CONSUMER ? (size_t) rn.opt.inbox_blocks : 0)
{
	if (role == ROUTER_PRODUCER) {
		rn.producers[index] = this;
		rn.refill(*this);
	} else if (role == ROUTER_CONSUMER)
		rn.consumers[index] = this;
	#pragma omp barrier
	if (role == ROUTER_SERVICE && rn.rank == 0 && rn.opt.verbose)
		rn.plan.report_measured();
}

/* Thread 0's object owns the node.  The team's objects die in any order at the end of the region -- a worker's
 * never touches the node here, and the node never touches a worker's -- which is sound only because every thread
 * has left the Router by then: a handle destroyed without Router_Finalize halts the process, since the thread
 * that skipped it may still be reading a node another thread's destructor has deleted. */
inline Router_thread::~Router_thread()
{
	if (not finalized) {
		const char *what = role == ROUTER_SERVICE ? "service" : (role == ROUTER_PRODUCER ? "producer" : "consumer");
		errx(1, "Router: the %s thread's handle was destroyed without Router_Finalize", what);
	}
	::free(swc);
	if (role == ROUTER_SERVICE)
		delete &node;
}

/*
 * The shell, no MPI: the caller's communicator as is, `tag` for every message of ours -- the caller keeps both
 * and sends nothing else with that tag --, the options (NULL: the defaults) and room for the team's roles.
 * Reached from Router_Init's single, on any thread of the team; connect() does the rest on thread 0.
 */
inline Router_node::Router_node(int n_threads, MPI_Comm mpi_comm, int tag_, const Router_Opts *options)
	: comm(mpi_comm), tag(tag_), rank(-1), n_nodes(0), opt(options ? *options : Router_Opts()),
	  input_closed(0)
{
	roles.assign(n_threads, -1);
	masks.assign(n_threads, cpu_set_t());
}

/* Reached from thread 0's ~Router_thread, at the end of the region, and only past Router_Finalize, which checked
 * that nothing of ours was in flight.  The workers' objects may be gone by now: nothing of theirs is read. */
inline Router_node::~Router_node()
{
	for (size_t k = 0; k < in_req.size(); k++)
		if (in_req[k] != MPI_REQUEST_NULL) {
			MPI_Cancel(&in_req[k]);
			MPI_Wait(&in_req[k], MPI_STATUS_IGNORE);
		}
	free(dest);
	free(pool);
	free(n_valid);
	free(partial);
	free(blk_link);
	free(free_cell);
}


/******************************** connection ********************************/

/*
 * Reached from Router_Init, thread 0 only, once every thread has declared its role and before any object is
 * built: the MPI checks, the agreement with the peers, the pool (reserved, not touched) and the shared area, one
 * block per destination, a stash kept, every other block onto the free ring in one run of tickets.
 */
inline void Router_node::connect()
{
	int provided;
	MPI_Query_thread(&provided);
	if (provided < MPI_THREAD_FUNNELED)
		errx(1, "Router: MPI_THREAD_FUNNELED is required (got %d)", provided);
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &n_nodes);
	P = 0;
	C = 0;
	producer_tid.clear();
	consumer_tid.clear();
	for (size_t t = 1; t < roles.size(); t++) {
		if (roles[t] == ROUTER_PRODUCER) {
			P += 1;
			producer_tid.push_back((int) t);
		} else {
			C += 1;
			consumer_tid.push_back((int) t);
		}
	}
	/* The windows follow the communicator, and every node derives the same ones from it.  Two messages in
	 * flight a peer is what saturates a link, measured on raw MPI and on the Router alike; the same count
	 * sizes the send slots, so a credit that wants more blocks in flight than there are slots collapses the
	 * rate -- 24 GB/s a node at 28 in flight of 32 slots, 12.4 at 56, 7.2 at 224. */
	if (opt.n_recv == 0)
		opt.n_recv = n_nodes > 1 ? std::max(16, 2 * (n_nodes - 1)) : 16;
	if (n_nodes > 1 && opt.credit * (n_nodes - 1) > opt.n_recv && rank == 0)
		warnx("Router: credit %d over %d peers wants %d blocks in flight but n_recv gives %d send slots: "
		      "the rate collapses above n_recv / peers", opt.credit, n_nodes - 1,
		      opt.credit * (n_nodes - 1), opt.n_recv);

	if (P < 1 || C < 1)
		errx(1, "Router: a node needs at least one producer and one consumer (P=%d, C=%d)", P, C);

	/* every node must agree on the team's shape and on the options: a difference is fatal, and named */
	const int N_AGREE = 8;
	const char *name[N_AGREE] = {"producers", "consumers", "block_points", "swc_linesize", "n_recv", "inbox_blocks",
	                             "credit", "pin"};
	double mine[N_AGREE] = {(double) P, (double) C, (double) opt.block_points,
							(double) opt.swc_linesize, (double) opt.n_recv, (double) opt.inbox_blocks,
							(double) opt.credit, (double) opt.pin};
	double lo[N_AGREE], hi[N_AGREE];
	MPI_Allreduce(mine, lo, N_AGREE, MPI_DOUBLE, MPI_MIN, comm);
	MPI_Allreduce(mine, hi, N_AGREE, MPI_DOUBLE, MPI_MAX, comm);
	for (int k = 0; k < N_AGREE; k++)
		if (lo[k] != hi[k]) {
			if (rank == 0)
				warnx("Router: the nodes disagree on %s (from %g to %g); every node must pass the same", name[k], lo[k], hi[k]);
			MPI_Abort(comm, 1);
		}

	if (not router_pow2(opt.block_points) || opt.block_points < 4)
		errx(1, "Router: block_points must be a power of two >= 4");
	if (opt.swc_linesize != 0 && (not router_pow2(opt.swc_linesize) || opt.swc_linesize < 4
								  || opt.swc_linesize > opt.block_points))
		errx(1, "Router: swc_linesize must be 0 or a power of two, 4 <= swc_linesize <= block_points");
	if (opt.swc_linesize != 0 && opt.block_points / opt.swc_linesize > ROUTER_MAX_L)
		errx(1, "Router: block_points / swc_linesize must be <= %u", ROUTER_MAX_L);
	if (opt.n_recv < 1 || opt.inbox_blocks < 1 || opt.credit < 1)
		errx(1, "Router: n_recv, inbox_blocks and credit must be >= 1");

	/* the union of the threads' own affinity masks: under OMP_PROC_BIND thread 0's alone would be one core */
	int n_threads = (int) roles.size();
	cpu_set_t umask;
	CPU_ZERO(&umask);
	for (int t = 0; t < n_threads; t++)
		CPU_OR(&umask, &umask, &masks[t]);

	/* pinning must own the CPUs: refuse a mask too small for the team, or one shared with a co-hosted rank */
	if (opt.pin) {
		if (CPU_COUNT(&umask) < n_threads)
			errx(1, "Router: rank %d has %d CPUs in its affinity mask for %d threads; use --bind-to none,"
			        " --map-by slot:PE=n --bind-to core, or pin = false", rank, CPU_COUNT(&umask), n_threads);
		MPI_Comm host;
		MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &host);
		int hn = 0;
		int hr = 0;
		MPI_Comm_size(host, &hn);
		MPI_Comm_rank(host, &hr);
		if (hn > 1) {
			std::vector<cpu_set_t> all(hn);
			MPI_Allgather(&umask, sizeof(cpu_set_t), MPI_BYTE, all.data(), sizeof(cpu_set_t), MPI_BYTE, host);
			for (int o = 0; o < hn; o++) {
				if (o == hr)
					continue;
				cpu_set_t both;
				CPU_AND(&both, &umask, &all[o]);
				if (CPU_COUNT(&both) > 0) {
					warnx("Router: rank %d shares CPUs with another rank on its host; use --map-by numa"
					      " --bind-to numa, --map-by slot:PE=n, or pin = false", rank);
					MPI_Abort(comm, 1);
				}
			}
		}
		MPI_Comm_free(&host);
	}

	plan = RouterPlacement(umask, roles, opt.pin, rank);
	if (rank == 0 && opt.verbose)
		plan.report();
	if (opt.pin && plan.thread_cpu[0] >= 0 && pin_to_cpu(plan.thread_cpu[0]) < 0)
		warn("Router: rank %d: cannot pin the service thread to CPU %d", rank, plan.thread_cpu[0]);

	int n_peers = n_nodes;
	credit_used.assign(n_peers, 0);
	seq_sent.assign(n_peers, 0);
	end_msg.assign(n_peers, RouterMsgHdr());
	end_req.assign(n_peers, MPI_REQUEST_NULL);
	end_sent.assign(n_peers, 0);
	n_data_recv.assign(n_peers, 0);
	end_seq.assign(n_peers, -1);

	F = C * n_nodes;

	size_t swc = opt.swc_linesize;
	if (swc == 0) {                       /* the lines fit L2, and a block switch is at most one line in 8 */
		size_t budget = (1024 << 10) / ((size_t) F * sizeof(Point));
		swc = 4;
		while (swc * 2 <= budget && swc * 2 <= opt.block_points / 8)
			swc *= 2;
	}
	while (opt.block_points / swc > ROUTER_MAX_L)
		swc *= 2;
	swc_linesize = swc;
	L = (u32) (opt.block_points / swc);
	nv_stride = (L + 63) & ~(u32) 63;
	block_bytes = ROUTER_HDR_BYTES + opt.block_points * sizeof(Point);

	/* installed, the stash, the slots, what the consumers may hold, the producers' caches, pending, and sealed or
	 * parked */
	size_t slack = 16 * ((size_t) F > 32 * (size_t) P ? (size_t) F : 32 * (size_t) P);   /* EXPERIMENT: x16 */
	size_t nb = (size_t) F + 2 * ROUTER_BATCH + 2 * (size_t) opt.n_recv
				+ (size_t) C * ((size_t) opt.inbox_blocks + 1) + (size_t) P * (ROUTER_BATCH + 1) + slack;
	if (nb > ((size_t) 1 << 30))          /* the ring's sequence arithmetic is 32-bit */
		errx(1, "Router: too many blocks (%zu)", nb);
	n_blocks = (u32) nb;
	dest = (Atomic<u64> *) router_alloc((size_t) F * ROUTER_DEST_WORDS * sizeof(u64));
	pool = (char *) aligned_alloc(64, (size_t) n_blocks * block_bytes);   /* untouched: every thread writes a slice */
	if (pool == NULL)
		err(1, "Router: aligned_alloc(%zu)", (size_t) n_blocks * block_bytes);
	n_valid = (Atomic<u8> *) router_alloc((size_t) n_blocks * nv_stride);
	partial_cap = ((size_t) P * (swc_linesize - 1) + 3) & ~(size_t) 3;
	partial = (Point *) router_alloc((size_t) F * partial_cap * sizeof(Point));
	blk_link = (Atomic<u64> *) router_alloc((size_t) n_blocks * sizeof(u64));
	free_mask = 1;
	while ((size_t) free_mask + 1 < n_blocks)   /* cells: the power of two >= n_blocks, so the ring is never full */
		free_mask = 2 * free_mask + 1;
	free_cell = (Atomic<u64> *) router_alloc(((size_t) free_mask + 1) * sizeof(u64));
	for (u64 i = 0; i <= free_mask; i++)
		free_cell[i] = (i << 32) | ROUTER_NONE;
	free_list.reserve(n_blocks);
	for (u32 b = 0; b < n_blocks; b++) {
		blk_link[b] = ROUTER_NONE;
		free_list.push_back(b);
	}
	blk_count.assign(n_blocks, 0);
	park_head.assign(C + n_nodes, ROUTER_NONE);
	park_tail.assign(C + n_nodes, ROUTER_NONE);
	pending.reserve(P + 1);
	inbox_pushed.assign(C, 0);

	out_req.assign(opt.n_recv, MPI_REQUEST_NULL);
	out_blk.assign(opt.n_recv, ROUTER_NONE);
	out_peer.assign(opt.n_recv, -1);
	out_free.clear();
	for (int k = 0; k < opt.n_recv; k++)
		out_free.push_back(k);
	in_req.assign(opt.n_recv, MPI_REQUEST_NULL);
	in_blk.assign(opt.n_recv, ROUTER_NONE);
	mpi_idx.assign(opt.n_recv, 0);
	mpi_st.assign(opt.n_recv, MPI_Status());
	producers.assign(P, NULL);
	consumers.assign(C, NULL);

	for (int d = 0; d < F; d++)
		dest[ROUTER_DEST_WORDS * d] = (u64) stash_pop();
	if (free_list.size() > ROUTER_BATCH)
		spill(free_list.size() - ROUTER_BATCH);
	if (rank == 0 && opt.verbose)
		banner();
}

/* This thread's slice of the pool, written once, pinned, so that its pages land on this thread's NUMA node
 * rather than all on the service's.  Whole blocks, T equal slices: no block is written by two threads.
 * Reached from Router_Init, every thread, between the pinning and the objects. */
inline void Router_node::touch_pool(int tid, int n_threads)
{
	size_t lo = (size_t) n_blocks * tid / n_threads;
	size_t hi = (size_t) n_blocks * (tid + 1) / n_threads;
	memset(pool + lo * block_bytes, 0, (hi - lo) * block_bytes);
}

/* the sizes, once.  Reached from connect(), on rank 0 only, when opt.verbose. */
inline void Router_node::banner() const
{
	double blocks = (double) n_blocks * block_bytes;
	double lines = (double) P * F * swc_linesize * sizeof(Point);
	double inboxes = (double) C * opt.inbox_blocks * sizeof(RouterBlockMsg);
	double closing = (double) F * partial_cap * sizeof(Point);
	printf("Router: %d node%s, %d producer%s and %d consumer%s per node, %d destinations\n",
	       n_nodes, n_nodes > 1 ? "s" : "", P, P > 1 ? "s" : "", C, C > 1 ? "s" : "", F);
	printf("Router: blocks of %zu points, lines of %zu (%u lines per block), %u blocks (%.1f MB, %u cached per"
	       " producer), messages of %zu bytes\n", opt.block_points, swc_linesize, L, n_blocks, blocks / 1e6,
	       ROUTER_BATCH, block_bytes);
	printf("Router: private lines %.1f MB (%.1f MB per producer), closing buffers %.1f MB (%zu points per destination)\n",
	       lines / 1e6, lines / 1e6 / P, closing / 1e6, partial_cap);
	printf("Router: inboxes of %d blocks (%.1f MB), %d receive and %d send slots, credit %d per peer, stash batches"
	       " of %u\n", opt.inbox_blocks, inboxes / 1e6, opt.n_recv, opt.n_recv, opt.credit, ROUTER_BATCH);
}

/*
 * Collective over every thread of the team and every node, the same arguments on every thread but `role`.  Thread
 * 0 is the service whatever `role` says.  Each thread records its role and its own affinity mask; thread 0
 * connects and builds the plan; then every thread pins itself, confirms the kernel obeyed, touches its slice of
 * the pool and builds its object, which every later call of this thread takes; the caller keeps it, named, for
 * the team's whole life.
 */
[[nodiscard]] inline Router_thread Router_Init(int role, MPI_Comm comm, int tag, const Router_Opts *opts)
{
	int tid = omp_get_thread_num();
	Router_node *rn = NULL;
	#pragma omp single copyprivate(rn)
	rn = new Router_node(omp_get_num_threads(), comm, tag, opts);
	if (tid == 0)
		rn->roles[0] = ROUTER_SERVICE;
	else if (role == ROUTER_PRODUCER || role == ROUTER_CONSUMER)
		rn->roles[tid] = role;
	else
		errx(1, "Router_Init: role must be ROUTER_PRODUCER or ROUTER_CONSUMER");
	if (sched_getaffinity(0, sizeof(cpu_set_t), &rn->masks[tid]) != 0)
		err(1, "Router_Init: sched_getaffinity");
	#pragma omp barrier
	if (tid == 0)
		rn->connect();
	#pragma omp barrier

	int want = rn->plan.thread_cpu[tid];  /* pin as the plan says, then confirm the kernel put us there */
	if (want >= 0 && pin_to_cpu(want) < 0) {
		warn("Router: rank %d: cannot pin thread %d to CPU %d", rn->rank, tid, want);
		#pragma omp atomic
		rn->misplaced += 1;
	}
	unsigned cpu = 0;
	if (syscall(SYS_getcpu, &cpu, NULL, NULL) != 0)
		err(1, "Router_Init: getcpu");
	if (want >= 0 && (int) cpu != want) {
		warnx("Router: rank %d: thread %d asked for CPU %d, runs on CPU %u", rn->rank, tid, want, cpu);
		#pragma omp atomic
		rn->misplaced += 1;
	}
	#pragma omp barrier                   /* everybody is pinned, or nobody builds an object */
	if (tid == 0 && rn->misplaced > 0)
		MPI_Abort(comm, 1);
	rn->touch_pool(tid, (int) rn->roles.size());

	int mine = rn->roles[tid];
	int index = 0;                        /* among the threads of my role: those before me */
	for (int t = 0; t < tid; t++)
		if (rn->roles[t] == mine)
			index += 1;
	return Router_thread(mine, index, (int) cpu, *rn);   /* registers itself, then holds the team's last barrier */
}

/*
 * Collective over the team, the last call every thread makes on its handle: each thread declares it is out of the
 * Router, and the barrier makes that true of every other before any of them returns, so that no thread reads the
 * node after the service thread's handle has deleted it -- which is what the caller has no other way of ensuring,
 * a consumer's Router_Test_drained reading the node.  Each thread also answers for its end of the round.  A
 * producer answers before the barrier, for its own flag: it closed what it pushed.  A consumer answers after it,
 * where the node is final and the question can be Router_Test_drained itself rather than a guess made while the
 * service still runs.  The service answers after a second barrier, once the workers have named themselves: the
 * node is quiescent, or was never used.  A consumer the service handed nothing this round is not asked at all,
 * which is what lets "reset, then finalize" pass: a fresh round has closed no input, so nobody is drained yet.
 * Nothing of the Router may be called on this thread afterwards; the handle may then die.
 */
inline void Router_Finalize(Router_thread &rt)
{
	Router_node &rn = rt.node;
	if (rt.finalized)
		errx(1, "Router_Finalize: called twice on the same thread");
	if (rt.role == ROUTER_PRODUCER && rt.ctr[ROUTER_PUSHED] > 0 && rt.closed.load_acquire() == 0)
		errx(1, "Router_Finalize: producer %d pushed %" PRIu64 " point(s) and never called Router_Close",
		     rt.index, rt.ctr[ROUTER_PUSHED]);
	#pragma omp barrier                   /* every thread is out of the Router: the node's state changes no more */
	if (rt.role == ROUTER_CONSUMER && rn.inbox_pushed[rt.index] > 0 && not Router_Test_drained(rt))
		errx(1, "Router_Finalize: consumer %d is not drained: of the %" PRIu64 " block(s) the service handed it,"
		        " it read through %" PRIu64 "%s", rt.index, rn.inbox_pushed[rt.index], rt.ctr[ROUTER_BLOCKS],
		     rn.input_closed.load_acquire() ? "" : ", and the round was still open");
	#pragma omp barrier                   /* every worker has answered for itself: what is left is the node's */
	if (rt.role == ROUTER_SERVICE) {
		bool idle = rn.phase == ROUTER_OPEN && rn.ctr[ROUTER_BLOCKS] == 0 && rn.ctr[ROUTER_MSGS_SENT] == 0;
		if (not rn.quiescent && not idle)
			errx(1, "Router_Finalize: the router is not quiescent");
	}
	rt.finalized = true;
}

/*
 * Any thread, about itself and the thread of `role` and local rank `index` on its node: the depth of the deepest
 * object of the machine's hierarchy holding both their CPUs, 0 the whole machine, larger closer, the caller with
 * itself the deepest of all; 0 for every pair when the threads are not pinned.  Ordinal only: hwloc merges the
 * levels a machine has one of, so no depth names a level.  The one thing the Router says about where it put the
 * threads.  The index is the caller's to keep in range, as for Router_size.
 */
inline int Router_closeness(const Router_thread &rt, int role, int index)
{
	const Router_node &rn = rt.node;
	int me = 0;
	if (rt.role == ROUTER_PRODUCER)
		me = rn.producer_tid[rt.index];
	else if (rt.role == ROUTER_CONSUMER)
		me = rn.consumer_tid[rt.index];
	int other = 0;
	if (role == ROUTER_PRODUCER)
		other = rn.producer_tid[index];
	else if (role == ROUTER_CONSUMER)
		other = rn.consumer_tid[index];
	return rn.plan.closeness(me, other);
}

/* Any thread.  The threads of `role` on this node (ROUTER_NODE) or over every node (ROUTER_GLOBAL). */
inline int Router_size(const Router_thread &rt, int role, int scope)
{
	int per_node = 1;                    /* the service: one per node, so its global size is the node count */
	if (role == ROUTER_PRODUCER)
		per_node = rt.node.P;
	else if (role == ROUTER_CONSUMER)
		per_node = rt.node.C;
	return (scope == ROUTER_GLOBAL) ? per_node * rt.node.n_nodes : per_node;
}

/* Any thread.  Its rank among the threads of its own role, on this node or over every node (node by node). */
inline int Router_rank(const Router_thread &rt, int scope)
{
	if (scope == ROUTER_NODE)
		return rt.index;
	return rt.node.rank * Router_size(rt, rt.role, ROUTER_NODE) + rt.index;
}

}
#endif
