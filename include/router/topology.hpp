#ifndef MITM_ROUTER_TOPOLOGY
#define MITM_ROUTER_TOPOLOGY

#include <err.h>
#include <sched.h>
#include <pthread.h>
#include <cerrno>
#include <cstdint>
#include <vector>
#include <hwloc.h>
#include <hwloc/glibc-sched.h>

/*
 * The Router's hwloc primitives: pinning, and the machine's hierarchy as the tree the placement descends.  The one
 * file that includes hwloc: RouterTree names objects by a dense id and answers in ids, CPUs and depths, so nothing
 * of hwloc's leaves it.
 */

static_assert(HWLOC_API_VERSION >= 0x00020000, "hwloc 2.x is required (NUMA nodes as memory children)");

namespace mitm {

/* pin the calling thread to `cpu`.  Returns cpu, or -1 with errno set: a failure, or cpu < 0 */
static inline int pin_to_cpu(int cpu)
{
	if (cpu < 0)
		return -1;
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
	if (rc != 0) {
		errno = rc;
		return -1;
	}
	return cpu;
}

/*
 * The machine's hierarchy restricted to an affinity mask: the machine, then whatever hwloc reports as normal objects
 * under it -- packages, groups, caches, cores -- down to the PUs, in child order.  NUMA nodes are memory children
 * and never on a path; a PU's nodeset names its own.  Per object, the PUs under it the placement may use and the
 * threads placed under it so far, which is what the descent compares.  Objects are named by a dense id, the root 0,
 * a subtree contiguous after its root.
 */
struct RouterTree {
	hwloc_topology_t topo;             /* the loaded topology, restricted to the mask */
	std::vector<hwloc_obj_t> obj;      /* by id */
	std::vector<int> avail;            /* per object, the PUs under it the placement may use */
	std::vector<int> placed;           /* per object, the threads placed under it so far */

	/* load, restrict to the mask -- an object with no PU of it is gone --, number what is left */
	RouterTree(const cpu_set_t &mask)
	{
		if (hwloc_topology_init(&topo) != 0)
			err(1, "hwloc_topology_init");
		if (hwloc_topology_load(topo) != 0)         /* no flag: RESTRICT_TO_CPUBINDING would defeat HWLOC_SYNTHETIC */
			err(1, "hwloc_topology_load");
		hwloc_cpuset_t set = hwloc_bitmap_alloc();
		if (hwloc_cpuset_from_glibc_sched_affinity(topo, set, &mask, sizeof(mask)) != 0)
			err(1, "hwloc_cpuset_from_glibc_sched_affinity");
		if (hwloc_topology_restrict(topo, set, 0) != 0)
			err(1, "hwloc_topology_restrict");
		hwloc_bitmap_free(set);
		number(hwloc_get_root_obj(topo));
	}

	~RouterTree()
	{
		hwloc_topology_destroy(topo);
	}

	RouterTree(const RouterTree &) = delete;
	RouterTree &operator=(const RouterTree &) = delete;

	/* `o` and everything under it get their ids, depth first in child order.  Reached from the constructor. */
	void number(hwloc_obj_t o)
	{
		o->userdata = (void *) (uintptr_t) obj.size();
		obj.push_back(o);
		avail.push_back(hwloc_bitmap_weight(o->cpuset));
		placed.push_back(0);
		for (unsigned k = 0; k < o->arity; k++)
			number(o->children[k]);
	}

	/* the first PU in child order: the first core of the first package, the first NUMA node's */
	int leftmost() const
	{
		hwloc_obj_t o = obj[0];
		while (o->arity > 0)
			o = o->children[0];
		return (int) (uintptr_t) o->userdata;
	}

	/* the core holding PU `pu` (the PU itself when hwloc reports no core level) leaves the tree: nothing is placed
	 * under it from now on, and no count sees its PUs.  Returns how many PUs went with it. */
	int remove_core(int pu)
	{
		hwloc_obj_t core = obj[pu];
		while (core->parent != NULL && core->type != HWLOC_OBJ_CORE)
			core = core->parent;
		if (core->type != HWLOC_OBJ_CORE)
			core = obj[pu];
		int n = avail[(int) (uintptr_t) core->userdata];
		for (hwloc_obj_t o = core; o != NULL; o = o->parent)
			avail[(int) (uintptr_t) o->userdata] -= n;
		return n;
	}

	/*
	 * One more thread: from the root, at every level the least-filled child -- placed over avail, ties to the
	 * first -- among those with a PU left, down to a PU, whose id it returns; every object on the path counts one
	 * more.  The root must have a PU left.
	 */
	int descend()
	{
		hwloc_obj_t o = obj[0];
		while (o->arity > 0) {
			hwloc_obj_t best = NULL;
			int ib = -1;
			for (unsigned k = 0; k < o->arity; k++) {
				hwloc_obj_t c = o->children[k];
				int ic = (int) (uintptr_t) c->userdata;
				if (avail[ic] <= placed[ic])
					continue;
				if (best == NULL || (long) placed[ic] * avail[ib] < (long) placed[ib] * avail[ic]) {
					best = c;
					ib = ic;
				}
			}
			if (best == NULL)
				errx(1, "Router: the placement ran out of PUs");
			o = best;
		}
		for (hwloc_obj_t a = o; a != NULL; a = a->parent)
			placed[(int) (uintptr_t) a->userdata] += 1;
		return (int) (uintptr_t) o->userdata;
	}

	/* what the placement records of PU `pu`: its CPU number, its core's id, its NUMA node, and its path -- the
	 * logical index of its ancestor at every depth, the machine's first */
	void locate(int pu, int *cpu, int *core, int *numa, std::vector<int> &path) const
	{
		hwloc_obj_t o = obj[pu];
		*cpu = (int) o->os_index;
		*numa = hwloc_bitmap_first(o->nodeset);
		path.assign(o->depth + 1, -1);
		*core = pu;
		for (hwloc_obj_t a = o; a != NULL; a = a->parent) {
			path[a->depth] = (int) a->logical_index;
			if (a->type == HWLOC_OBJ_CORE)
				*core = (int) (uintptr_t) a->userdata;
		}
	}
};

}
#endif
