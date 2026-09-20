#ifndef MITM_ROUTER_SERVICE
#define MITM_ROUTER_SERVICE

#include "common.hpp"

namespace mitm {

/******************************** the service thread ********************************/

/* a block from the stash, refilled from the free ring by the batch when empty; NONE if both are empty.  Reached
 * from Router_Init (the installs), Router_Progress (a repost, the F-scan) and Router_Reset (a repost). */
inline u32 Router_node::stash_pop()
{
	if (free_list.empty()) {
		u32 got[ROUTER_BATCH];
		u32 n = free_pop_many(ROUTER_BATCH, got, 0);   /* the service takes the last block: its receives come first */
		for (u32 i = 0; i < n; i++)
			free_list.push_back(got[i]);
		if (n == 0)
			return ROUTER_NONE;
	}
	u32 blk = free_list.back();
	free_list.pop_back();
	return blk;
}

/* the top n blocks of the stash onto the free ring, one run of tickets.  Reached from Router_Init, once the stash
 * exceeds a batch, and Router_Progress, once a release takes it past two. */
inline void Router_node::spill(size_t n)
{
	size_t keep = free_list.size() - n;
	free_push(free_list.data() + keep, (u32) n);
	free_list.resize(keep);
}

/*
 * Are all L lines of a sealed block written?  A zero n_valid byte means not yet, never a torn value, each
 * line being its own release/acquire pair.  A written line is full: a block in flight has no hole.
 * Reached from Router_Progress, for every sealed block the take or the pending list yields.
 */
inline bool Router_node::complete(u32 blk)
{
	for (u32 k = 0; k < L; k++)
		if (n_valid[(size_t) nv_stride * blk + k].load_acquire() == 0)
			return false;
	return true;
}

/*
 * A block to its destination's owner: the local consumer's inbox, or an Isend from block memory to its node.
 * True == the block changed hands, and the caller must forget it.  False == no room: the inbox is full, or
 * no send slot is free, or the peer has its credit in flight.
 * Reached from Router_Progress, for every block dispatched and every parked block retried.
 */
inline bool Router_node::place(int d, u32 blk, u32 count)
{
	int node = d / C;
	int local = d % C;
	if (node == rank) {
		RouterBlockMsg e = {blk, count};
		if (not consumers[local]->inbox.push(e))
			return false;
		inbox_pushed[local] += 1;
		ctr[ROUTER_LOCAL] += count;
		return true;
	}
	if (out_free.empty() || credit_used[node] >= opt.credit)
		return false;
	int k = out_free.back();
	out_free.pop_back();
	char *hdr = pool + (size_t) blk * block_bytes;
	RouterMsgHdr h = {ROUTER_DATA, round, seq_sent[node], (u32) local, count, 0};
	memcpy(hdr, &h, sizeof(h));
	int len = (int) (ROUTER_HDR_BYTES + (size_t) count * sizeof(Point));
	MPI_Isend(hdr, len, MPI_BYTE, node, tag, comm, &out_req[k]);
	out_blk[k] = blk;
	out_peer[k] = node;
	credit_used[node] += 1;
	seq_sent[node] += 1;
	ctr[ROUTER_SENT] += count;
	ctr[ROUTER_MSGS_SENT] += 1;
	ctr[ROUTER_BYTES_SENT] += len;
	return true;
}

/* a complete block to its destination: placed, or parked until it fits, never kept by the caller.
 * Reached from Router_Progress: a sealed block once complete, a DATA message received, the F-scan's blocks. */
inline void Router_node::dispatch(int d, u32 blk, u32 count)
{
	if (place(d, blk, count))
		return;
	park(d, blk, count);
}

/* a block the service is done with: into its stash, which spills a batch to the free ring when it grows.  Reached
 * from Router_Progress, for a send that completed. */
inline void Router_node::release(u32 blk)
{
	ctr[ROUTER_BLOCKS] += 1;
	free_list.push_back(blk);
	if (free_list.size() > 2 * ROUTER_BATCH)
		spill(ROUTER_BATCH);
}

/* keep a block whose target is full, in arrival order, until the target has room; parking rather than
 * refusing to pop is what keeps one slow consumer from stalling every other destination.  Reached from
 * Router_Progress, when a block finds its inbox full or its peer out of slots or credit. */
inline void Router_node::park(int d, u32 blk, u32 count)
{
	int node = d / C;
	int target = (node == rank) ? d % C : C + node;
	blk_count[blk] = count;
	blk_link[blk] = ((u64) d << 32) | ROUTER_NONE;
	if (park_head[target] == ROUTER_NONE)
		park_head[target] = blk;
	else {                                /* the tail keeps its own destination in the high word */
		Atomic<u64> &tail = blk_link[park_tail[target]];
		tail = (tail & ~(u64) ROUTER_NONE) | blk;
	}
	park_tail[target] = blk;
	ctr[ROUTER_STALL_OUT] += 1;
}

/* one target's parked blocks, in order, as far as they place.  Reached from Router_Progress, every turn for every
 * target; a no-op unless blocks are parked.  careful: the link is read before the block is placed,
 * since once placed the block is on its way round and a later seal may rewrite its link. */
inline void Router_node::retry_parked(int target)
{
	while (park_head[target] != ROUTER_NONE) {
		u32 blk = park_head[target];
		u64 link = blk_link[blk];
		u32 after = (u32) link;
		if (not place((int) (link >> 32), blk, blk_count[blk]))
			return;
		park_head[target] = after;
		if (after == ROUTER_NONE)
			park_tail[target] = ROUTER_NONE;
	}
}

/* a sealed block: dispatch it, or pend it until its last line is written.  Reached from Router_Progress, for
 * every block of the take and every block pended the turn before. */
inline void Router_node::handle_block(int d, u32 blk)
{
	if (not complete(blk)) {
		pending.push_back(((u64) d << 32) | blk);
		return;
	}
	dispatch(d, blk, (u32) opt.block_points);
}

/* the sends that completed: their blocks are the service's again, their slots and their peer's credit too.
 * Reached from Router_Progress, every turn, first. */
inline void Router_node::poll_out()
{
	int outcount = 0;
	MPI_Testsome(opt.n_recv, out_req.data(), &outcount, mpi_idx.data(), mpi_st.data());
	if (outcount == MPI_UNDEFINED)
		return;
	for (int t = 0; t < outcount; t++) {
		int k = mpi_idx[t];
		release(out_blk[k]);
		credit_used[out_peer[k]] -= 1;
		out_blk[k] = ROUTER_NONE;
		out_free.push_back(k);
	}
}

/* post a block from the stash on receive slot k; none == the slot idles, and repost_idle retries every turn.
 * Reached from Router_Progress (a slot whose DATA block went to its consumer, and the idle slots, every slot on
 * the first turn) and Router_Reset (the idle slots). */
inline void Router_node::repost(int k)
{
	u32 blk = stash_pop();
	if (blk == ROUTER_NONE) {
		ctr[ROUTER_STALL_IN] += 1;
		return;
	}
	in_blk[k] = blk;
	MPI_Irecv(pool + (size_t) blk * block_bytes, (int) block_bytes, MPI_BYTE, MPI_ANY_SOURCE, tag, comm, &in_req[k]);
}

/* a block on every idle receive slot, as far as the stash and the free ring allow.  Reached from
 * Router_Progress, every turn, and Router_Reset, once. */
inline void Router_node::repost_idle()
{
	for (int k = 0; k < opt.n_recv; k++)
		if (in_blk[k] == ROUTER_NONE) {
			repost(k);
			if (in_blk[k] == ROUTER_NONE)
				return;
		}
}

/*
 * Take delivery of what arrived: an END is recorded and its block reposted; a DATA block goes to its consumer
 * as it is -- parked if the inbox is full -- and its slot gets a fresh block.
 * Reached from Router_Progress, every turn.
 */
inline void Router_node::poll_in()
{
	int outcount = 0;
	MPI_Testsome(opt.n_recv, in_req.data(), &outcount, mpi_idx.data(), mpi_st.data());
	if (outcount == MPI_UNDEFINED)
		return;
	for (int t = 0; t < outcount; t++) {
		int k = mpi_idx[t];
		u32 blk = in_blk[k];
		int src = mpi_st[t].MPI_SOURCE;
		int len = 0;
		MPI_Get_count(&mpi_st[t], MPI_BYTE, &len);
		RouterMsgHdr h;
		memcpy(&h, pool + (size_t) blk * block_bytes, sizeof(h));
		if (h.round != round)
			errx(1, "Router: rank %d got a round %u message in round %u", rank, h.round, round);
		ctr[ROUTER_MSGS_RECV] += 1;
		ctr[ROUTER_BYTES_RECV] += len;
		if (h.kind == ROUTER_END) {
			end_seq[src] = (long long) h.seq;
			MPI_Irecv(pool + (size_t) blk * block_bytes, (int) block_bytes, MPI_BYTE, MPI_ANY_SOURCE, tag, comm,
			          &in_req[k]);
			continue;
		}
		n_data_recv[src] += 1;
		ctr[ROUTER_RECV] += h.count;
		in_blk[k] = ROUTER_NONE;
		dispatch(rank * C + (int) h.dest, blk, h.count);
		repost(k);
	}
}

/* the pending list, then the sealed blocks: the stack taken whole in one exchange and handled whole, newest
 * first; the take is bounded by the pool, and what is not complete yet is pended for the next turn.
 * Reached from Router_Progress, every turn. */
inline void Router_node::sweep()
{
	if (not pending.empty()) {
		std::vector<u64> again;
		again.swap(pending);
		for (size_t i = 0; i < again.size(); i++)
			handle_block((int) (again[i] >> 32), (u32) again[i]);
	}
	u32 blk = sealed_top.exchange(ROUTER_NONE, std::memory_order_acquire);
	while (blk != ROUTER_NONE) {
		u64 link = blk_link[blk];   /* careful: park or a later seal rewrites it */
		handle_block((int) (link >> 32), blk);
		blk = (u32) link;
	}
}

/*
 * Destination `d` at closure, once every producer is closed: what its installed block holds, and how many
 * points its closing buffer holds.
 * Reached from Router_Progress in the F-scan phase, once per destination and round.
 */
inline void Router_node::start_flush(int d)
{
	u64 v = dest[ROUTER_DEST_WORDS * d].load_acquire();
	u32 k = (u32) (v >> 32);
	u32 blk = (u32) v;
	if (blk == ROUTER_NONE)
		errx(1, "Router: destination %d has no block at closure", d);
	if (k > L)
		errx(1, "Router: destination %d has %u lines reserved at closure", d, k);
	for (u32 i = 0; i < k; i++)
		if (n_valid[(size_t) nv_stride * blk + i].load_acquire() == 0)
			errx(1, "Router: destination %d has an unwritten line at closure", d);
	flush_blk = blk;
	flush_k = k;
	flush_install = false;
	flush_len = (u32) dest[ROUTER_DEST_WORDS * d + 1];
	flush_off = 0;
	flush_built = true;
}

/*
 * The closing runs, one destination per step.  The installed block's full lines go as one short block, the
 * block itself, and a fresh one is installed in its place; then the closing buffer is copied into fresh
 * blocks, at most a block at a time, each dispatched -- the service's one copy, per round.  Both need free
 * blocks: it stops where there is none and resumes there next turn.
 * Reached from Router_Progress, every turn of the F-scan phase, until it returns true.
 */
inline bool Router_node::fscan()
{
	for (; flush_d < F; flush_d++) {
		if (not flush_built)
			start_flush(flush_d);
		if (flush_k > 0) {
			dispatch(flush_d, flush_blk, flush_k * (u32) swc_linesize);
			flush_k = 0;
			flush_install = true;
		}
		if (flush_install) {
			u32 fresh = stash_pop();
			if (fresh == ROUTER_NONE)
				return false;
			zero_valid(fresh);
			dest[ROUTER_DEST_WORDS * flush_d].store_release((u64) fresh);
			flush_install = false;
		}
		const Point *buf = partial + (size_t) flush_d * partial_cap;
		while (flush_off < flush_len) {
			u32 chunk = (u32) std::min((size_t) (flush_len - flush_off), opt.block_points);
			u32 blk = stash_pop();
			if (blk == ROUTER_NONE)
				return false;
			memcpy(pool + (size_t) blk * block_bytes + ROUTER_HDR_BYTES, buf + flush_off,
			       (size_t) chunk * sizeof(Point));
			dispatch(flush_d, blk, chunk);
			flush_off += chunk;
		}
		dest[ROUTER_DEST_WORDS * flush_d + 1] = 0;
		flush_built = false;
	}
	return true;
}

/*
 * Send the ENDs, test every request; true once nothing of ours is in flight.  A peer's END waits for its
 * parked blocks: they still have to go out, and the END must follow them.
 * Reached from Router_Progress, every turn of the flushing phase, until it returns true.
 */
inline bool Router_node::flush_all()
{
	bool done = true;
	for (int p = 0; p < n_nodes; p++) {
		if (p == rank)
			continue;
		if (park_head[C + p] != ROUTER_NONE) {
			done = false;
			continue;
		}
		if (not end_sent[p]) {
			end_msg[p] = RouterMsgHdr{ROUTER_END, round, seq_sent[p], 0, 0, 0};
			MPI_Isend(&end_msg[p], sizeof(RouterMsgHdr), MPI_BYTE, p, tag, comm, &end_req[p]);
			end_sent[p] = 1;
			ctr[ROUTER_MSGS_SENT] += 1;
			ctr[ROUTER_BYTES_SENT] += sizeof(RouterMsgHdr);
		}
		if (end_req[p] != MPI_REQUEST_NULL) {
			int flag = 0;
			MPI_Test(&end_req[p], &flag, MPI_STATUS_IGNORE);
			if (not flag)
				done = false;
		}
	}
	if (out_free.size() != (size_t) opt.n_recv)
		done = false;
	return done;
}

/* nothing can enter an inbox any more: local flush done, no block parked on a consumer, every peer's END and
 * all its DATA in.  The END's seq is compared with the DATA count because MPI orders matching, not
 * completion, across posted receives.  Reached from Router_Progress, every turn; a no-op before the flushing
 * phase. */
inline void Router_node::check_input_closed()
{
	if (phase < ROUTER_FLUSHING)
		return;
	for (int r = 0; r < C; r++)
		if (park_head[r] != ROUTER_NONE)
			return;
	for (int p = 0; p < n_nodes; p++)
		if (p != rank && (end_seq[p] < 0 || (u64) end_seq[p] != n_data_recv[p]))
			return;
	input_closed.store_release(1);
}

/* out closed, input closed, and every block handed to a consumer given back: nobody holds anything.  The
 * consumer's count is plain and read racily; it is written after the push it counts, so it is never early.
 * Reached from Router_Progress, every turn; a no-op until the output side is closed and the input closed. */
inline void Router_node::check_quiescent()
{
	if (phase != ROUTER_OUT_CLOSED || input_closed == 0)
		return;
	for (int r = 0; r < C; r++)
		if (inbox_pushed[r] != consumers[r]->ctr[ROUTER_BLOCKS])
			return;
	quiescent = true;
}

/* the output side's phases, one step per turn at most.  Reached from Router_Progress, every turn; a phase
 * advances only once its condition holds, the first being every producer closed. */
inline void Router_node::closure()
{
	switch (phase) {
	case ROUTER_OPEN:
		for (int s = 0; s < P; s++)
			if (producers[s]->closed.load_acquire() == 0)
				return;
		phase = ROUTER_COLLECTING;
		return;
	case ROUTER_COLLECTING:
		if (not pending.empty() || sealed_top.load_acquire() != ROUTER_NONE)
			return;
		phase = ROUTER_FSCAN;
		flush_d = 0;
		return;
	case ROUTER_FSCAN:
		if (fscan())
			phase = ROUTER_FLUSHING;
		return;
	case ROUTER_FLUSHING:
		if (flush_all())
			phase = ROUTER_OUT_CLOSED;
		return;
	default:
		return;
	}
}

/* Service.  One non-blocking turn: every part runs whatever the phase, so no stall couples two; the sends that
 * completed come first, so that this turn's receive slots have their blocks.  The time since the previous turn
 * ended is this turn's, BUSY when it moved anything -- a block handed over, parked, delivered or recycled, a
 * message either way -- so that what the caller does between two turns counts as the service's. */
inline void Router_Progress(Router_thread &rt)
{
	if (rt.role != ROUTER_SERVICE)
		errx(1, "Router_Progress: not the service thread");
	Router_node &rn = rt.node;
	if (rn.quiescent)
		return;
	u64 before = rn.ctr[ROUTER_BLOCKS] + rn.ctr[ROUTER_MSGS_SENT] + rn.ctr[ROUTER_MSGS_RECV] + rn.ctr[ROUTER_RECV]
	           + rn.ctr[ROUTER_LOCAL] + rn.ctr[ROUTER_STALL_OUT];
	rn.poll_out();
	for (int t = 0; t < rn.C + rn.n_nodes; t++)
		rn.retry_parked(t);
	rn.poll_in();
	rn.repost_idle();
	rn.sweep();
	rn.closure();
	rn.check_input_closed();
	rn.check_quiescent();
	u64 after = rn.ctr[ROUTER_BLOCKS] + rn.ctr[ROUTER_MSGS_SENT] + rn.ctr[ROUTER_MSGS_RECV] + rn.ctr[ROUTER_RECV]
	          + rn.ctr[ROUTER_LOCAL] + rn.ctr[ROUTER_STALL_OUT];
	u64 now = (u64) (wtime() * 1e9);
	if (rn.ctr[ROUTER_TURNS] == 0)       /* the round's first turn: its clock starts here */
		rn.t_round = now;
	else if (after != before)
		rn.ctr[ROUTER_BUSY] += now - rn.t_turn;
	rn.t_turn = now;
	rn.ctr[ROUTER_TURNS] += 1;
	if (after == before)
		rn.ctr[ROUTER_IDLE_TURNS] += 1;
}

/* Service.  Every piece of the node's state, for a run that does not end; the reads are racy on purpose. */
inline void Router_Dump(FILE *f, const Router_thread &rt)
{
	if (rt.role != ROUTER_SERVICE)
		errx(1, "Router_Dump: not the service thread");
	Router_node &rn = rt.node;
	u64 in = rn.free_in.load();
	u64 out = rn.free_out.load();
	fprintf(f, "rank %d round %u phase %d input_closed %u quiescent %d flush_d %d run %u/%u pending %zu"
	        " sealed top %u stash %zu ring %" PRIu64 " (in %" PRIu64 " out %" PRIu64 ")"
	        " send slots free %zu\n", rn.rank, rn.round, rn.phase, rn.input_closed.load(), (int) rn.quiescent,
	        rn.flush_d, rn.flush_off, rn.flush_len, rn.pending.size(), rn.sealed_top.load(),
	        rn.free_list.size(), in - out, in, out, rn.out_free.size());
	for (int s = 0; s < rn.P; s++) {
		Router_thread &sd = *rn.producers[s];
		fprintf(f, "  producer %d (cpu %d): closed %u pushed %" PRIu64 " cached %u\n", s,
		        sd.cpu, sd.closed.load(), sd.ctr[ROUTER_PUSHED], sd.cache_n);
	}
	for (int r = 0; r < rn.C; r++) {
		Router_thread &rr = *rn.consumers[r];
		fprintf(f, "  consumer %d (cpu %d): inbox %zu/%zu cur %u (%u/%u) pushed %" PRIu64 " given back %"
		        PRIu64 " parked %u\n", r, rr.cpu, (size_t) rr.inbox.head.load(),
		        (size_t) rr.inbox.tail.load(), rr.cur_blk, rr.cur_off, rr.cur_count, rn.inbox_pushed[r],
		        rr.ctr[ROUTER_BLOCKS], rn.park_head[r]);
	}
	for (int d = 0; d < rn.F; d++) {
		u64 v = rn.dest[ROUTER_DEST_WORDS * d].load();
		fprintf(f, "  dest %d: block %u lines %u", d, (u32) v, (u32) (v >> 32));
		if ((u32) v != ROUTER_NONE)
			for (u32 k = 0; k < rn.L && k < 64; k++)
				fprintf(f, " %u", rn.n_valid[(size_t) rn.nv_stride * (u32) v + k].load());
		fprintf(f, "\n");
	}
	for (int p = 0; p < rn.n_nodes; p++) {
		if (p == rn.rank)
			continue;
		fprintf(f, "  peer %d: seq_sent %" PRIu64 " n_data_recv %" PRIu64 " end_seq %lld end_sent %d end_req %d"
		        " credit_used %d parked %u\n", p, rn.seq_sent[p], rn.n_data_recv[p], rn.end_seq[p],
		        (int) rn.end_sent[p], rn.end_req[p] != MPI_REQUEST_NULL, rn.credit_used[p], rn.park_head[rn.C + p]);
	}
	for (int k = 0; k < rn.opt.n_recv; k++) {
		if (rn.out_blk[k] != ROUTER_NONE)
			fprintf(f, "  send slot %d: block %u to peer %d\n", k, rn.out_blk[k], rn.out_peer[k]);
		if (rn.in_blk[k] == ROUTER_NONE)
			fprintf(f, "  receive slot %d: idle\n", k);
	}
	fflush(f);
}

/* Service.  This node has sent and received everything of the round and its consumers hold nothing. */
inline bool Router_Test_quiescent(const Router_thread &rt)
{
	return rt.node.quiescent;
}

/*
 * Service.  The node's tallies; exact once quiescent and the caller's barrier has passed, a snapshot before.
 * A worker's wait still under way goes in, so that a snapshot sees a thread starved since the round began: its
 * tallies are read before its stamp, so a wait it closes in between is missed once rather than counted twice,
 * and a stamp later than the clock read here adds nothing rather than wrapping.
 */
inline void Router_Stats(u64 *stats, const Router_thread &rt)
{
	if (rt.role != ROUTER_SERVICE)
		errx(1, "Router_Stats: not the service thread");
	Router_node &rn = rt.node;            /* the handle is const, the node it refers to is not: ELAPSED is written */
	u64 now = (u64) (wtime() * 1e9);
	rn.ctr[ROUTER_ELAPSED] = rn.ctr[ROUTER_TURNS] ? now - rn.t_round : 0;
	for (int k = 0; k < ROUTER_STATS_SIZE; k++)
		stats[k] = rn.ctr[k];
	for (int s = 0; s < rn.P; s++) {
		for (int k = 0; k < ROUTER_STATS_SIZE; k++)
			stats[k] += rn.producers[s]->ctr[k];
		u64 since = rn.producers[s]->wait_since;
		if (since != 0 && (int64_t) (now - since) > 0)
			stats[ROUTER_WAIT_SEND] += now - since;
	}
	for (int r = 0; r < rn.C; r++) {
		for (int k = 0; k < ROUTER_STATS_SIZE; k++)
			stats[k] += rn.consumers[r]->ctr[k];
		u64 since = rn.consumers[r]->wait_since;
		if (since != 0 && (int64_t) (now - since) > 0)
			stats[ROUTER_WAIT_RECV] += now - since;
	}
}

/*
 * Any thread.  What limits the flow, read off a Router_Stats array -- one node's, or the sum of several nodes',
 * ELAPSED being then summed as many times, so that the per-node counts below divide right: the share of the round
 * the producers spent without pushing -- waiting for a free block, or closed --, the consumers with nothing to pop
 * and the service in turns that moved something, each in [0, 1]; and the verdict.  "producers" when they never waited and the consumers did:
 * the points are produced too slowly.  "consumers" the other way round: they are consumed too slowly.
 * "service/network" when both waited for the Router: the blocks are in flight or in the service's hands, and
 * *busy says which -- near 1 the service's CPU, well below it the wire or the credit.  "balanced" when neither
 * waited.  Meaningful for a round long against its closure: the drain is consumer wait.
 */
inline const char *Router_Bottleneck(const u64 *stats, const Router_thread &rt, double *send_wait,
                                     double *recv_wait, double *busy)
{
	const Router_node &rn = rt.node;
	double elapsed = (double) stats[ROUTER_ELAPSED];
	*send_wait = 0;
	*recv_wait = 0;
	*busy = 0;
	if (elapsed > 0) {
		*send_wait = std::min(1., (double) stats[ROUTER_WAIT_SEND] / (rn.P * elapsed));
		*recv_wait = std::min(1., (double) stats[ROUTER_WAIT_RECV] / (rn.C * elapsed));
		*busy = std::min(1., (double) stats[ROUTER_BUSY] / elapsed);
	}
	bool producers_wait = *send_wait >= ROUTER_WAIT_THRESHOLD;
	bool consumers_wait = *recv_wait >= ROUTER_WAIT_THRESHOLD;
	if (producers_wait && consumers_wait)
		return "service/network";
	if (producers_wait)
		return "consumers";
	if (consumers_wait)
		return "producers";
	return "balanced";
}

/*
 * The node itself, back to a fresh round: the service thread's own state, its peers' bookkeeping and the round
 * counter, then an MPI_Barrier so that no peer starts the next round before this node has left this one.
 * Reached from Router_Reset, on the service thread, between the team's two barriers.
 */
inline void Router_node::reset_round()
{
	if (not quiescent)
		errx(1, "Router_Reset: not quiescent");
	repost_idle();
#ifdef ROUTER_PARANOID
	size_t accounted = F + free_list.size();
	for (int k = 0; k < opt.n_recv; k++)
		if (in_blk[k] != ROUTER_NONE)
			accounted += 1;
	for (int s = 0; s < P; s++)
		accounted += producers[s]->cache_n;
	u64 out = free_out.load();         /* the ring, nobody else touching it now: exact, and every cell ready */
	u64 in = free_in.load();
	for (u64 p = out; p < in; p++)
		if ((u32) (free_cell[p & free_mask].load() >> 32) != (u32) (p + 1))
			errx(1, "Router_Reset: a free cell is not ready");
	accounted += in - out;
	if (accounted != n_blocks)
		errx(1, "Router_Reset: %zu of %u blocks accounted for (%" PRIu64 " in the ring)", accounted, n_blocks,
		     in - out);
	if (out_free.size() != (size_t) opt.n_recv)
		errx(1, "Router_Reset: a send is still in flight");
	for (int p = 0; p < n_nodes; p++)
		if (credit_used[p] != 0)
			errx(1, "Router_Reset: credit in use");
	if (not pending.empty())
		errx(1, "Router_Reset: pending work left");
	if (sealed_top.load() != ROUTER_NONE)
		errx(1, "Router_Reset: sealed blocks left");
	for (int d = 0; d < F; d++)
		if (dest[ROUTER_DEST_WORDS * d + 1] != 0)
			errx(1, "Router_Reset: a closing buffer was left behind");
#endif
	for (int k = 0; k < ROUTER_STATS_SIZE; k++)
		ctr[k] = 0;
	for (int r = 0; r < C; r++)
		inbox_pushed[r] = 0;
	for (int p = 0; p < n_nodes; p++) {
		seq_sent[p] = 0;
		n_data_recv[p] = 0;
		end_seq[p] = -1;
		end_sent[p] = 0;
	}
	input_closed = 0;
	quiescent = false;
	phase = ROUTER_OPEN;
	flush_d = 0;
	flush_built = false;
	flush_blk = ROUTER_NONE;
	flush_k = 0;
	flush_install = false;
	flush_len = 0;
	flush_off = 0;
	round += 1;
	MPI_Barrier(comm);
}

/*
 * Collective over the team.  Back to a fresh round: once every worker is out of the round, each thread clears
 * its own tallies and flags while the service thread resets the node, and nobody pushes or pops before all of
 * that is done, here and on every peer.  Both barriers are the team's, so every thread meets the same two.  The
 * tallies are gone past this call: Router_Stats reads them before it.
 */
inline void Router_Reset(Router_thread &rt)
{
	Router_node &rn = rt.node;
	if (rt.role == ROUTER_CONSUMER && rt.cur_blk != ROUTER_NONE)
		errx(1, "Router_Reset: a consumer stopped popping before it was drained");
	#pragma omp barrier                   /* every worker is out of the round: nothing cleared below is read */
	if (rt.role != ROUTER_SERVICE) {
		for (int k = 0; k < ROUTER_STATS_SIZE; k++)
			rt.ctr[k] = 0;
		rt.closed = 0;
		rt.wait_since = 0;
	} else
		rn.reset_round();
	#pragma omp barrier                   /* the node is fresh, here and on every peer, before anybody pushes or pops */
}

}
#endif
