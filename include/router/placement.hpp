#ifndef MITM_ROUTER_PLACEMENT
#define MITM_ROUTER_PLACEMENT

#include <sched.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <err.h>

#include "topology.hpp"

/*
 * Where the Router puts a node's threads.  The service thread alone on the leftmost core of the machine, which
 * then no longer exists for anyone; then every consumer, then every producer, each down the hierarchy to the
 * least-filled child at every level, so that both roles spread over the NUMA nodes, the caches and the cores in
 * proportion to their PUs, one thread per core while cores last.  No groups: the caller who pairs threads asks
 * Router_closeness.  The role enum lives here, the lowest Router file, so both this and common.hpp see it.
 */

namespace mitm {

enum router_role { ROUTER_SERVICE = 0, ROUTER_PRODUCER = 1, ROUTER_CONSUMER = 2 };


/*
 * The plan, derived once from the union of the threads' affinity masks and const from then on.  Every per-thread
 * vector is indexed by OpenMP thread id.
 */
struct RouterPlacement {
	bool pin = false;                      /* pin the threads, or leave the CPUs to the caller */
	int n_threads = 0;                     /* the team: 1 service + P producers + C consumers */
	int n_avail_cpu = 0;                   /* CPUs in the union mask */
	int n_numa_nodes = 0;                  /* NUMA nodes the threads land on; 0 when not pinned */
	int n_shared_cores = 0;                /* cores holding two threads or more; 0 when not pinned */

	std::vector<int> thread_role;          /* per thread, its router_role: what the reports read */
	std::vector<int> thread_cpu;           /* per thread, the CPU to pin it to; -1 == do not pin */
	std::vector<int> thread_core;          /* per thread, its core (an id of the tree it was placed in); -1 when not pinned */
	std::vector<int> thread_numa;          /* per thread, the NUMA node it lands on; -1 when not pinned */
	std::vector<std::vector<int>> thread_path;   /* per thread, its PU's ancestors' logical indices by depth, the machine's first; empty when not pinned */

	RouterPlacement() = default;

	/*
	 * From the union `mask`, the team's `roles` and the rank (for the messages).  Not pinned: nothing is decided.
	 * Pinned: the service on the first PU of the leftmost core, that core taken out; a mask then too small for
	 * the workers is refused; the consumers, then the producers, each down the tree.
	 */
	RouterPlacement(const cpu_set_t &mask, const std::vector<int> &roles, bool pin_, int rank)
		: pin(pin_), n_threads((int) roles.size()), n_avail_cpu(CPU_COUNT(&mask)), thread_role(roles)
	{
		thread_cpu.assign(n_threads, -1);
		thread_core.assign(n_threads, -1);
		thread_numa.assign(n_threads, -1);
		thread_path.assign(n_threads, std::vector<int>());
		if (not pin)
			return;

		RouterTree tree(mask);
		int pu = tree.leftmost();
		tree.locate(pu, &thread_cpu[0], &thread_core[0], &thread_numa[0], thread_path[0]);
		int taken = tree.remove_core(pu);
		if (tree.avail[0] < n_threads - 1)
			errx(1, "Router: rank %d: %d CPUs in the affinity mask, %d of them the service thread's core, %d left for"
			        " %d workers; use fewer threads, a larger mask (--bind-to none, --map-by slot:PE=n) or pin = false",
			     rank, n_avail_cpu, taken, tree.avail[0], n_threads - 1);

		for (int pass = 0; pass < 2; pass++) {      /* the consumers, then the producers */
			int want = (pass == 0) ? ROUTER_CONSUMER : ROUTER_PRODUCER;
			for (int t = 1; t < n_threads; t++) {
				if (thread_role[t] != want)
					continue;
				pu = tree.descend();
				tree.locate(pu, &thread_cpu[t], &thread_core[t], &thread_numa[t], thread_path[t]);
			}
		}

		std::vector<int> numa_ids;                  /* for the reports: the distinct NUMA nodes, and the shared cores */
		for (int t = 0; t < n_threads; t++)
			if (std::find(numa_ids.begin(), numa_ids.end(), thread_numa[t]) == numa_ids.end())
				numa_ids.push_back(thread_numa[t]);
		n_numa_nodes = (int) numa_ids.size();
		std::vector<int> per_core(tree.obj.size(), 0);
		for (int t = 0; t < n_threads; t++)
			per_core[thread_core[t]] += 1;
		for (size_t k = 0; k < per_core.size(); k++)
			n_shared_cores += per_core[k] >= 2;
	}

	/*
	 * How close threads `a` and `b` sit: the depth of the deepest object of the hierarchy holding both, 0 the
	 * machine, a thread with itself the deepest of all; 0 for every pair when not pinned.  Ordinal only: hwloc
	 * merges the levels a machine has one of, so no depth names a level.
	 */
	int closeness(int a, int b) const
	{
		const std::vector<int> &pa = thread_path[a];
		const std::vector<int> &pb = thread_path[b];
		size_t n = std::min(pa.size(), pb.size());
		int d = 0;
		while ((size_t) d + 1 < n && pa[d + 1] == pb[d + 1])
			d += 1;
		return d;
	}

	/* the intended layout, once.  Reached from connect() before the sizes banner, on rank 0 when verbose. */
	void report() const
	{
		if (not pin) {
			printf("Router: threads NOT pinned (pin = false)\n");
			return;
		}
		printf("Router: %d CPUs over %d NUMA node(s); threads pinned, the service alone on CPU %d\n", n_avail_cpu,
		       n_numa_nodes, thread_cpu[0]);
	}

	/*
	 * What the kernel honoured, once the team is pinned and every thread's getcpu has confirmed it: per NUMA node
	 * the counts by role, then the cores two threads share.  Reached from thread 0's Router_thread constructor,
	 * rank 0 when verbose.
	 */
	void report_measured() const
	{
		if (not pin)
			return;
		std::vector<int> ids;
		for (int t = 0; t < n_threads; t++)
			if (std::find(ids.begin(), ids.end(), thread_numa[t]) == ids.end())
				ids.push_back(thread_numa[t]);
		std::sort(ids.begin(), ids.end());
		for (size_t j = 0; j < ids.size(); j++) {
			int n_comm = 0;
			int n_prod = 0;
			int n_cons = 0;
			for (int t = 0; t < n_threads; t++) {
				if (thread_numa[t] != ids[j])
					continue;
				if (thread_role[t] == ROUTER_SERVICE)
					n_comm += 1;
				else if (thread_role[t] == ROUTER_PRODUCER)
					n_prod += 1;
				else
					n_cons += 1;
			}
			printf("NUMA node %d: %s%d producers + %d consumers\n", ids[j], n_comm ? "service + " : "", n_prod, n_cons);
		}
		printf("%d core(s) run two threads\n", n_shared_cores);
		fflush(stdout);
	}
};

}
#endif
