#ifndef MITM_DIRECT_DICT
#define MITM_DIRECT_DICT

#include <cassert>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <cinttypes>
#include <algorithm>
#include <sys/mman.h>

#include "tools.hpp"
#include "router/router.hpp"
#include "direct/params.hpp"
#include "direct/shared.hpp"

/* One shard of the distributed dictionary of preimages, and the thread that owns it. */

namespace mitm::direct {

using std::vector;

/******************************* the shard's pages ****************************/

/* the page these mappings ask for: the kernel rounds a MAP_HUGETLB length up to it, munmap does not */
static constexpr size_t HUGE_PAGE = 2 * 1024 * 1024;

/* glibc < 2.36 defines MAP_HUGE_SHIFT but not the sizes, which only <linux/mman.h> has: log2(2 MB) == 21 */
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

/*
 * Map `nbytes` of zeroed memory for a shard: 2 MB pages when the kernel has some reserved, ordinary ones
 * otherwise, `why` saying why MAP_HUGETLB was refused, 0 if it was not.  No MAP_LOCKED: the kernel clears
 * VM_LOCKED on a hugetlb mapping, whose pages cannot be swapped anyway, so the flag locks nothing and its
 * RLIMIT_MEMLOCK check only refuses shards larger than the limit.
 */
static u64 *map_shard(size_t nbytes, int *why)
{
	int flags = MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | MAP_HUGE_2MB;
	void *ptr = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, flags, -1, 0);
	if (ptr != MAP_FAILED)
		return (u64 *) ptr;
	*why = errno;
	ptr = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	if (ptr == MAP_FAILED)
		err(1, "direct: cannot map %zu bytes for a dictionary shard", nbytes);
	madvise(ptr, nbytes, MADV_HUGEPAGE);   /* transparent huge pages, all the kernel has left to offer */
	return (u64 *) ptr;
}

/*
 * How much of [ptr, ptr + nbytes) the kernel really backs with 2 MB pages, read from /proc/self/smaps once
 * every page has been touched.  Exact for MAP_HUGETLB, whose mapping stands alone; an upper bound for
 * transparent pages, which the kernel reports over a region it may have merged with the neighbouring ones.
 */
static u64 huge_bytes(const void *ptr, size_t nbytes)
{
	FILE *f = fopen("/proc/self/smaps", "r");
	if (f == NULL)
		return 0;
	uintptr_t base = (uintptr_t) ptr;
	uintptr_t end = base + nbytes;
	u64 overlap = 0;          /* how much of the region being read falls inside the shard */
	u64 total = 0;            /* the bytes on 2 MB pages so far */
	char line[256];
	while (fgets(line, sizeof(line), f) != NULL) {
		uintptr_t lo, hi;
		int kb;
		if (sscanf(line, "%" SCNxPTR "-%" SCNxPTR, &lo, &hi) == 2) {
			overlap = (lo < end && base < hi) ? std::min(end, hi) - std::max(base, lo) : 0;
			continue;
		}
		if (overlap == 0)
			continue;
		if (sscanf(line, "AnonHugePages: %d kB", &kb) == 1)
			total += (u64) kb * 1024;
		if (sscanf(line, "MMUPageSize: %d kB", &kb) == 1 && kb >= 2048)
			total += overlap;      /* a hugetlb region: every page of it is a huge one */
	}
	fclose(f);
	return std::min(total, (u64) nbytes);
}


/******************************* dictionary shard *****************************/

/*
 * One shard of the distributed dictionary: linear probing over 64-bit slots.
 * A slot is the preimage in its low n bits, the key's check bits above, and
 * bit 63 set; zero is empty.  Built by its dict thread once it is pinned.
 */
class DirectDict {
public:
	static constexpr u64 OCCUPIED = 1ull << 63;   /* occupancy bit */

	const u64 n_slots;     /* size of A */
	const int n;           /* preimage bits */
	const u64 xmask;       /* the low n bits: the preimage */
	const u64 cmask;       /* the check bits, 63 - n of them, before their shift */
	u64 *A;                /* A[i] == OCCUPIED | check << n | preimage, or 0 */
	size_t nbytes;         /* the mapping: the slots rounded up to a whole 2 MB page, so up to that much over */
	u64 huge;              /* of those bytes, what the kernel really backs with 2 MB pages */
	int hugetlb_errno;     /* why MAP_HUGETLB was refused; 0 if it was not */

	DirectDict(u64 n_slots, int n) : n_slots(n_slots), n(n), xmask(make_mask(n)), cmask(make_mask(63 - n)),
	                                 A(nullptr), nbytes(0), huge(0), hugetlb_errno(0)
	{
		assert(n <= 63);
		if (n_slots == 0)
			return;
		nbytes = (n_slots * sizeof(*A) + HUGE_PAGE - 1) & ~(HUGE_PAGE - 1);
		A = map_shard(nbytes, &hugetlb_errno);
		memset(A, 0, nbytes);      /* first touch (NUMA awareness) + claiming of transparent huge pages */
		huge = huge_bytes(A, nbytes);
	}

	~DirectDict()
	{
		if (A != nullptr)
			munmap(A, nbytes);
	}

	DirectDict(const DirectDict &) = delete;
	DirectDict &operator=(const DirectDict &) = delete;

	/* assumption: h is a uniformly random u64 */
	void insert(u64 h, u64 x)
	{
		u64 i = (u64) (((unsigned __int128) h * n_slots) >> 64);
		while (A[i] != 0) {
			i += 1;
			if (i == n_slots)
				i = 0;
		}
		A[i] = OCCUPIED | ((h & cmask) << n) | x;
	}

	/* 
	 * The preimages the shard holds under `key`: every slot of its run carrying the key's tag, check-bit
	 * false positives included; `out` is cleared first. 
	 */
	void probe(u64 h, vector<u64> &out) const
	{
	    out.clear();
	    u64 tag = OCCUPIED | ((h & cmask) << n);
	    u64 i = (u64) (((unsigned __int128) h * n_slots) >> 64);
		while (A[i] != 0) {
	        if ((A[i] & ~xmask) == tag)
	            out.push_back(A[i] & xmask);
			i += 1;
			if (i == n_slots)
				i = 0;
		}
	}

	/* Pull the first slot of the key's run into cache ahead of its probe (read). */
	void prefetch(u64 h) const
	{
		u64 i = (u64) (((unsigned __int128) h * n_slots) >> 64);
		__builtin_prefetch(&A[i], 0);
	}

	/* empty the shard after a PROBE phase */
	void clear()
	{
		memset(A, 0, n_slots * sizeof(*A));
	}
};


/******************************* the dict thread ******************************/

/*
 * A dict thread's phase: take the points the Router delivers, one at a time, until nothing more can arrive.
 * While the round fills, a point is an entry, (hashed image, preimage), inserted; while it probes, a point is a
 * probe, resolved on the spot: a match costs about what a probe does, so nothing is handed to anybody.  A point
 * is not retired as it comes: its slot is prefetched and it waits in a ring of D points, retired when the point
 * D pops later arrives, so that the slot's cache miss overlaps the pops behind it instead of being sat through
 * alone.  The ring spans blocks and is drained, oldest first, once the Router is; D == 0 retires every point as
 * it comes.  After a PROBE phase the shard is emptied.
 */

template <class Wrapper>
inline void retire(const Wrapper &wrapper, Shared &shared, DirectDict &dict, u64 *ctr, int phase, vector<u64> &candidates, u64 h, u64 y)
{
	if (phase == FILL) {
		dict.insert(h, y);
		ctr[N_INSERT] += 1;
	} else {
		dict.probe(h, candidates);
		ctr[N_PROBE] += 1;
		for (u64 x : candidates) {
			ctr[N_COLLISIONS] += 1;
			if (not wrapper.good(x, y))
				continue;
			shared.set_golden(x, y);
		}
	}	
}


template <class Wrapper>
void dict_round(Router_thread &rt, const Wrapper &wrapper, Shared &shared, DirectDict &dict, u64 *ctr,
                u64 round, int phase, int D)
{
	pair<u64,u64> ring[MAX_PREFETCH];  /* the points popped and prefetched but not yet retired */
	int head = 0;                      /* the oldest of them, the next one retired */
	int n_pending = 0;                 /* how many wait: D once the ring is full */
	vector<u64> candidates;

	for (;;) {
		u64 h, x;
		/* get (h, xi), with h = murmur64(f(xi)), but h is skewed (destination picked from high bits) */
		if (not Router_Pop(&h, &x, rt)) {
			if (Router_Test_drained(rt))
				break;
			cpu_relax();               /* no CAS in Router_Pop --> no need for backoff */
			continue;
		}
		u64 hh = murmur64(h);          /* re-randomize the hash, to get rid of the skew */
		if (D == 0) {
			retire(wrapper, shared, dict, ctr, phase, candidates, hh, x);
			continue;
		}
        /* prefetching ring */
		dict.prefetch(hh);
		if (n_pending < D) {       /* the ring is still filling */
			ring[n_pending] = pair(hh, x);
			n_pending += 1;
			continue;
		}
		auto [hh_old, y_old] = ring[head];
		retire(wrapper, shared, dict, ctr, phase, candidates, hh_old, y_old);
		ring[head] = pair(hh, x);
		head += 1;
		if (head == D)
			head = 0;
	}

	for (int i = 0; i < n_pending; i++) {   /* drain the prefect ring */
		auto [hh, y] = ring[head];
		retire(wrapper, shared, dict, ctr, phase, candidates, hh, y);
		head += 1;
		if (head == D)
			head = 0;
	}
	if (phase == PROBE)
		dict.clear();
}

}
#endif
