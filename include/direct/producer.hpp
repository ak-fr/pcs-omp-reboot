#ifndef MITM_DIRECT_PRODUCER
#define MITM_DIRECT_PRODUCER

#include <algorithm>

#include "tools.hpp"
#include "router/router.hpp"
#include "direct/params.hpp"

namespace mitm::direct {

/******************************** the producer thread ********************************/

/* push every image as(murmur64(image), preimage) to the consumer the hash's names */
template <class Wrapper>
void producer_round(Router_thread &rt, const Wrapper &wrapper, const Params &params, u64 *ctr, u64 round, int phase)
{
	constexpr int vlen = Wrapper::vlen;
	const u64 n_recv = (u64) Router_size(rt, ROUTER_CONSUMER, ROUTER_GLOBAL);
	u64 lo = 0;
	u64 hi = params.domain;
	if (phase == FILL) {
		lo = round * params.per_round;
		hi = std::min(lo + params.per_round, params.domain);
	}
	u64 p = (u64) Router_rank(rt, ROUTER_GLOBAL); 
	u64 n_pieces = (u64) Router_size(rt, ROUTER_PRODUCER, ROUTER_GLOBAL);
	u64 span = hi - lo;
	u64 piece = span / n_pieces;
	u64 extra = span % n_pieces;
	u64 my_lo = lo + piece * p + std::min(p, extra);
	u64 my_hi = my_lo + piece + ((p < extra) ? 1 : 0);

	u64 x[vlen] __attribute__ ((aligned(sizeof(u64) * vlen)));
	u64 y[vlen] __attribute__ ((aligned(sizeof(u64) * vlen)));
	for (u64 base = my_lo; base < my_hi; base += vlen) {
		int valid = (my_hi - base < (u64) vlen) ? (int) (my_hi - base) : vlen;
		for (int k = 0; k < vlen; k++)
			x[k] = (k < valid) ? base + k : base;   /* a lane past the end still gets an input in the domain */
		wrapper.veval(phase, x, y);
		for (int k = 0; k < valid; k++) {
			u64 h = murmur64(y[k]);
		    int dest = (int) (((unsigned __int128) h * n_recv) >> 64);
			Router_Push(h, x[k], dest, rt);
		}
		ctr[N_EVAL] += valid;
	}
	Router_Close(rt);
}

}
#endif
