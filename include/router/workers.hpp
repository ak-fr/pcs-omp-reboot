#ifndef MITM_ROUTER_WORKERS
#define MITM_ROUTER_WORKERS

#include "common.hpp"

namespace mitm {

/******************************** the free ring ********************************/

/*
 * Put n blocks onto the free ring, a run of tickets then their cells, each once the popper of its previous
 * element has stored it free -- at most that popper's last stores away, the ring being never full.  The cell's
 * store releases everything the pusher did to the block.
 * Reached from Router_Pop (the block it read through) and from spill, in Router_Init and Router_Progress (a batch).
 */
inline void Router_node::free_push(const u32 *blks, u32 n)
{
	u64 t = free_in.fetch_add(n, std::memory_order_relaxed);
	for (u32 i = 0; i < n; i++) {
		Atomic<u64> &cell = free_cell[(t + i) & free_mask];
		while ((u32) (cell.load_acquire() >> 32) != (u32) (t + i))
			cpu_relax();
		cell.store_release(((u64) (u32) (t + i + 1) << 32) | blks[i]);
	}
}

/*
 * Pop up to k blocks off the free ring: the ready prefix at free_out, claimed whole by one CAS; zero when the
 * element at free_out is not there (nothing free, or a push under way), never a wait.  A ready cell changes only
 * when claimed, and a claim moves free_out, which fails the CAS: the ids read before it are the blocks.  The
 * acquire loads pair with the pushers' release stores: what they did to a block is visible to whoever pops it.
 * `floor` blocks are left in the ring for the caller who passes zero, the service: a node that cannot post a
 * receive stalls every peer that sends to it, and a peer's send holds a block until it completes, so producers
 * taking the last block deadlock the whole run.  The consumers' releases refill the ring without the network.
 * Nothing is held back with one node: nothing arrives, so a posted receive never needs replacing.
 * Reached from Router_Push (a producer's refill, floor n_recv) and from the service whenever its stash runs empty.
 */
inline u32 Router_node::free_pop_many(u32 k, u32 *out, u32 floor)
{
	u64 pos = free_out;
	u32 n = 0;
	while (n == 0) {
		u64 avail = free_in - pos;
		if (avail <= (u64) floor)
			return 0;
		if ((u64) k > avail - floor)
			k = (u32) (avail - floor);
		u64 c = free_cell[pos & free_mask].load_acquire();
		int32_t dif = (int32_t) ((u32) (c >> 32) - (u32) (pos + 1));
		if (dif < 0)
			return 0;                     /* element pos is not there yet */
		if (dif > 0) {                    /* claimed already: my view of free_out is stale */
			pos = free_out;
			continue;
		}
		out[0] = (u32) c;
		n = 1;
		while (n < k) {                   /* the ready prefix behind it */
			c = free_cell[(pos + n) & free_mask].load_acquire();
			if ((u32) (c >> 32) != (u32) (pos + n + 1))
				break;
			out[n] = (u32) c;
			n += 1;
		}
		if (not free_out.compare_exchange_weak(pos, pos + n, std::memory_order_relaxed))
			n = 0;                        /* lost the race; pos now holds the value that won */
	}
	for (u32 i = 0; i < n; i++)           /* the cells, free for their next round */
		free_cell[(pos + i) & free_mask].store_release(((u64) (u32) (pos + i + free_mask + 1) << 32) | ROUTER_NONE);
	return n;
}


/******************************** the producer path ********************************/

/*
 * A producer's cache filled off the free ring, a batch in one CAS, every block zeroed here rather than at its
 * install: it waits for at least one block, holding nothing meanwhile, so that no point is ever lost.  The wait
 * is timed into WAIT_SEND: a producer starved of blocks is the one way the Router holds a producer back.
 * Reached from Router_Init (the producer's constructor) and Router_Push, by the sealer whose install emptied it.
 */
inline void Router_node::refill(Router_thread &s)
{
	u32 n = free_pop_many(ROUTER_BATCH, s.cache, n_nodes > 1 ? (u32) opt.n_recv : 0);
	if (n == 0) {
		s.wait_since = (u64) (wtime() * 1e9);
		while (n == 0) {
			cpu_relax();
			n = free_pop_many(ROUTER_BATCH, s.cache, n_nodes > 1 ? (u32) opt.n_recv : 0);
		}
		s.ctr[ROUTER_WAIT_SEND] += (u64) (wtime() * 1e9) - s.wait_since;
		s.wait_since = 0;
	}
	for (u32 i = 0; i < n; i++)
		zero_valid(s.cache[i]);
	s.cache_n = n;
}

/*
 * No line of a block about to be installed is written yet.
 * Reached from Router_Init and Router_Push (a producer's refill), and Router_Progress (the F-scan's fresh block).
 */
inline void Router_node::zero_valid(u32 blk)
{
	for (u32 k = 0; k < L; k++)
		n_valid[(size_t) nv_stride * blk + k] = 0;
}

/* Put a sealed block onto the sealed stack, its destination in the link's high word. */
inline void Router_node::seal(int d, u32 blk)
{
	u64 hi = (u64) d << 32;
	u32 top = sealed_top;
	for (;;) {
		blk_link[blk] = hi | top;
		if (sealed_top.compare_exchange_weak(top, blk, std::memory_order_release, std::memory_order_relaxed))
			return;
	}
}

/*
 * Reserve one line slot in the block of destination `d` and write the full line there, so that a block in
 * flight never has a hole; seal the block when the reservation lands on its end, the fresh one coming out of
 * the producer's cache, refilled after the seal once empty; wait for that install when another producer is the
 * sealer.  A destination always names a block: connect installs one and every seal replaces it.
 * Reached from Router_Push, by the push that fills the producer's private line for `d`.
 */
inline void Router_node::stage_line(Router_thread &s, int d, const Point *line)
{
	Atomic<u64> &next = dest[ROUTER_DEST_WORDS * d];
	int n = (int) swc_linesize;
	for (;;) {
		u64 v = next.fetch_add(1ull << 32, std::memory_order_acq_rel);
		u32 k = (u32) (v >> 32);
		u32 blk = (u32) v;
		if (blk == ROUTER_NONE)
			errx(1, "Router: destination %d has no block", d);
		if (k < L) {
			Point *pts = (Point *) (pool + (size_t) blk * block_bytes + ROUTER_HDR_BYTES);
			router_stream_copy(pts + (size_t) k * swc_linesize, line, (size_t) n * sizeof(Point));
			n_valid[(size_t) nv_stride * blk + k].store_release(1);
			return;
		}
		if (k == L) {                     /* the sealer; refill waits, so its cache is never empty here */
			u32 fresh = s.cache[--s.cache_n];
			next.store_release((u64) fresh);
			seal(d, blk);
			if (s.cache_n == 0)
				refill(s);
			continue;
		}
		for (;;) {                        /* k > L: at exactly L nobody is installing, and I may become the sealer */
			u64 w = next.load_acquire();
			if ((u32) (w >> 32) <= L)
				break;
			cpu_relax();
		}
	}
}

/* Producer.  May spin until the Router has room; the point is delivered exactly once. */
inline void Router_Push(u64 a, u64 b, int d, Router_thread &rt)
{
	Point *line = rt.swc + (size_t) d * rt.swc_linesize;
	u64 &count = line[rt.swc_linesize - 1].val;
	u64 n = count;
	line[n].key = a;
	line[n].val = b;                  /* the line's last point lands on the count, read already */
	n += 1;
	rt.ctr[ROUTER_PUSHED] += 1;
	if (n < rt.swc_linesize) {
		count = n;
		return;
	}
	rt.node.stage_line(rt, d, line);
	count = 0;                        /* after the copy: the word held a point until now */
}

/*
 * Producer.  Append every partial line's points to its destination's closing buffer, contiguous with the other
 * producers' thanks to the offset a fetch_add hands out, then announce; nothing more from this producer until
 * Router_Reset.  Nothing partial ever enters a block before the F-scan copies the closing buffers into blocks.
 * From here to the round's end the producer is idle, and that is its wait too: a round that goes on once every
 * producer has closed is the consumers' to finish.
 */
inline void Router_Close(Router_thread &rt)
{
	Router_node &rn = rt.node;
	for (int d = 0; d < rn.F; d++) {
		Point *line = rt.swc + (size_t) d * rt.swc_linesize;
		u64 &count = line[rt.swc_linesize - 1].val;
		if (count == 0)
			continue;
		u64 off = rn.dest[ROUTER_DEST_WORDS * d + 1].fetch_add(count, std::memory_order_relaxed);
		memcpy(rn.partial + (size_t) d * rn.partial_cap + off, line, (size_t) count * sizeof(Point));
		count = 0;
	}
	rt.closed.store_release(1);
	rt.wait_since = (u64) (wtime() * 1e9);   /* closed while the round goes on: Router_Stats counts it up to its call */
}


/******************************** the consumer's calls ********************************/

/*
 * The next block the consumer's inbox names becomes the one it reads, in place: where its points are and how
 * many.  False when nothing is there, and the time from that to the next block is the consumer's wait,
 * WAIT_RECV.  Reached from Router_Pop, with no block being read.
 */
inline bool Router_node::grab(Router_thread &r)
{
	RouterBlockMsg e;
	if (not r.inbox.pop(e)) {
		if (r.wait_since == 0)
			r.wait_since = (u64) (wtime() * 1e9);
		return false;
	}
	if (r.wait_since != 0) {
		r.ctr[ROUTER_WAIT_RECV] += (u64) (wtime() * 1e9) - r.wait_since;
		r.wait_since = 0;
	}
	r.cur_blk = e.blk;
	r.cur_count = e.count;
	r.cur_pts = (const u64 *) (pool + (size_t) r.cur_blk * block_bytes + ROUTER_HDR_BYTES);
	r.cur_off = 0;
	return true;
}

/* The block being read is read through: onto the free ring, where a producer may install it at once, so nothing
 * of it is read past this call.  The push orders those reads before the block's reuse; the count comes after
 * the push so that it never runs ahead of it: the service compares it with what it handed out.  Reached from
 * Router_Pop, by the pop that takes the block's last point. */
inline void Router_node::release(Router_thread &r)
{
	free_push(&r.cur_blk, 1);
	r.ctr[ROUTER_BLOCKS] += 1;
	r.cur_blk = ROUTER_NONE;
}

/* Consumer.  One point, out of the block being read or the next one; false when nothing is there now.  The
 * last point of a block is copied out before the release it triggers: the block may be someone else's right
 * after. */
inline bool Router_Pop(u64 *a, u64 *b, Router_thread &rt)
{
	if (rt.cur_blk == ROUTER_NONE && not rt.node.grab(rt))
		return false;
	*a = rt.cur_pts[2 * rt.cur_off];
	*b = rt.cur_pts[2 * rt.cur_off + 1];
	rt.cur_off += 1;
	rt.ctr[ROUTER_POPPED] += 1;
	if (rt.cur_off == rt.cur_count)
		rt.node.release(rt);
	return true;
}

/* Consumer.  Nothing can arrive any more and nothing is left to read, the block being read included: the flag
 * first, then emptiness, so that a block pushed between the two reads is still popped.  The round's last wait
 * ends here, no block coming to end it. */
inline bool Router_Test_drained(Router_thread &rt)
{
	if (rt.node.input_closed.load_acquire() == 0)
		return false;
	if (rt.cur_blk != ROUTER_NONE || not rt.inbox.empty())
		return false;
	if (rt.wait_since != 0) {
		rt.ctr[ROUTER_WAIT_RECV] += (u64) (wtime() * 1e9) - rt.wait_since;
		rt.wait_since = 0;
	}
	return true;
}

}
#endif
