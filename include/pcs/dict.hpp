#ifndef MITM_PCS_DICT
#define MITM_PCS_DICT

#include <cassert>

#include "tools.hpp"
#include "router/router.hpp"
#include "pcs/params.hpp"
#include "pcs/shared.hpp"

/* One shard of the distributed dictionary of trail endpoints, and the thread that owns it. */

namespace mitm::pcs {

/******************************* dictionary shard *****************************/

/*
 * Probed with a trail's endpoint, it returns the chain index -- and the length, if it fit -- of an
 * earlier trail that ended there.  One slot per endpoint, no probing: a hit is only ever a hint, and
 * the walker that resolves it re-walks both trails anyway.  Built by its dict thread once it is pinned.
 */
class PcsDict {
public:
	u64 jbits;             /* low bits of a slot: the chain index */
	u64 lbits;             /* jbits + 8: the trail length sits above the chain index */
	u64 jmask;             /* the low jbits */
	u64 lmask;             /* 0xff: lengths saturate at 255 */
	u64 key_mask;          /* the bits above lbits */
	const u64 n_slots;     /* size of A */

	vector<u64> A;         /* A[i][0:jbits] == j.  A[i][jbits:lbits] == len1.  A[lbits:64] == key bits */

	PcsDict(u64 jbits, u64 w) : jbits(jbits), n_slots(w)
	{
		assert(jbits <= 56);
		jmask = make_mask(jbits);
		lmask = make_mask(8);
		lbits = jbits + 8;
		key_mask = (lbits == 64) ? 0 : 0xffffffffffffffff << lbits;
		A.resize(n_slots);     /* zero-filled by the calling thread: the first touch of every page */
	}

	/* zero the shard between rounds; the owner alone today */
	void flush()
	{
		for (u64 i = 0; i < n_slots; i++)
			A[i] = 0;
	}

	/*
	 * Probe-and-insert: the (chain index, length) already at this endpoint's slot, if its key matches;
	 * length 0 == it had saturated.  A slot is overwritten only by a trail at least as long.  `len0`
	 * arrives saturated by the wire format already, so the clamp below only narrows 8 bits further.
	 */
	optional<pair<u64, u64>> pop_insert(u64 end, u64 start, u64 len0)
	{
		u64 idx = end % n_slots;
		u64 key = (end / n_slots) << lbits;

		u64 e = A[idx];
		u64 ekey = e & key_mask;
		u64 elen = (e >> jbits) & lmask;

		if (e == 0 || len0 >= elen) {
			if (len0 > lmask)
				len0 = lmask;
			A[idx] = start ^ (len0 << jbits) ^ key;
		}

		if (ekey != key || e == 0)
			return nullopt;

		if (elen == lmask)
			elen = 0;

		return optional(pair(e & jmask, elen));
	}
};


/******************************* the dict thread ******************************/

/*
 * A dict thread's round: probe every distinguished point the Router delivers into its shard and hand the
 * hits to its own walkers, over its own queue and in runs.  A run goes over as soon as there is nothing
 * left to probe, so that a partial one never waits on the next hit.  Nothing is resolved here: a walk of
 * two trails costs far more than a probe, and the walkers own the mixing function.  `r` is this dict
 * thread's local rank, which names its channel.
 */
static void dict_round(Router_thread &rt, const Params &params, Shared &shared, PcsDict &dict, u64 *ctr, int r)
{
	CollisionQueue &coll_q = *shared.chan[r].q;
	const u64 i = shared.header.i;
	const u64 jmask = make_mask(params.jbits);
	const u64 n_recv = (u64) Router_size(rt, ROUTER_CONSUMER, ROUTER_GLOBAL);   /* the routing modulus */
	static const size_t BATCH = 64;
	CollisionCandidate pending[BATCH];     /* hits waiting for the run to be handed over */
	size_t n_pending = 0;

	for (;;) {
		u64 key;                       /* the endpoint the walker routed on */
		u64 val;                       /* its chain index, and its trail length above that */
		if (Router_Pop(&key, &val, rt)) {
			u64 seed0 = val & jmask;
			u64 len = (val >> params.jbits) & params.len_sat;
			u64 len0_maybe = (len == params.len_sat) ? 0 : len;
			/* unknown on the wire stays unknown in the slot, whatever the 8-bit field could hold */
			u64 dict_len = len0_maybe ? len0_maybe : 0xffffffffffffffffull;
			ctr[N_PROBE] += 1;
			auto hit = dict.pop_insert(key / n_recv, seed0, dict_len);
			if (not hit) {
				ctr[BAD_PROBE] += 1;
				continue;
			}
			auto [seed1, len1_maybe] = *hit;
			pending[n_pending] = {i, seed0, len0_maybe, key, seed1, len1_maybe};
			n_pending += 1;
			if (n_pending == BATCH) {
				ctr[DROP_COLL] += n_pending - coll_q.push_bulk(pending, n_pending);
				n_pending = 0;
			}
			continue;
		}

		if (n_pending > 0) {
			ctr[DROP_COLL] += n_pending - coll_q.push_bulk(pending, n_pending);
			n_pending = 0;
		}
		if (Router_Test_drained(rt))
			break;
		cpu_relax();                   /* no CAS in Router_Pop --> no need for backoff */
	}

	/* the queue is final before the shard is emptied, so a walker's drain overlaps the flush */
	shared.chan[r].done.store_release(1);
	dict.flush();
}

}
#endif
