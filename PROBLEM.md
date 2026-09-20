# Routing is the bottleneck on a large node

Measured on `grdix-4` (2 sockets, 256 cores / 512 PUs, 2 NUMA nodes of 256 PUs each,
1 TB) in September 2026, on `double_speck64_demo`.  The L3 layout below is inferred
from the placement the banner reports, not from `lstopo`; check it before relying on
it.  Everything below is from
`--n 40 --ram 4G`, so `w = 5.0e8` slots (`2^28.90`) and the auto-tuned trail length is
`1/theta == 18.76`.

**The engine delivers about 4 M distinguished points per second to its dictionary, on
a machine whose walkers can produce 760 M/s.  Nothing tried so far moved that number,
because every point on a node passes through one thread.**

Re-measured since on a second, smaller machine, where the same 4 M/s ceiling appeared and
**one MPI rank per L3 domain moved it by 11x** (40x on collisions per second): **§8**.
Read that section before quoting any number here.

## 1. What was measured

`-O3 -DNDEBUG` throughout; the first `-O0` run is left out, its rates mean nothing.
"offered" is `walkers * (f/s) / L`; "absorbed" is `inserters * (probe/s)`, both as the
round report prints them.

| walkers / inserters | threads | offered | walker-queue drops | absorbed | dict occupancy | collisions | coll/s |
|---|---|---|---|---|---|---|---|
| 507 / 4 | 512 | 757 M/s | 95.9% | 3.9 M/s | 3.1% | 2^16.19 | 13.8 K |
| 250 / 4 | 255 | 456 M/s | 92.4% | 4.4 M/s | 7.3% | 2^17.88 | 27.0 K |
| 24 / 32 | 57 | 48 M/s | 90.4% | 4.6 M/s | 76.7% | 2^24.17 | 219 K |
| 4 / 4 | 9 | 2.8 M/s | 0.88% | 2.8 M/s | 8x overfilled | 2^29.33 | **477 K** |

Two readings that are easy to get wrong:

- **Wall-clock per round is not the metric.**  A round closes on distinguished points
  *found* (`controller.hpp:96`, against `N_DP`), and in the first three rows over 90%
  of those were dropped before reaching a shard.  The 512-thread run finished a round
  in 5.4 s having inserted 3% of `beta*w`; the 9-thread run took 1436 s and filled the
  dictionary eight times over.  Collisions per second is the objective, and on it the
  9-thread configuration beats the 512-thread one by **34x**.
- **The comm thread's apparent 31-34 M DP/s in rows 1 and 2 was 87% *failed* pushes** --
  a `head.load` and a return, no write -- because the inserter queues were permanently
  full.  Its real delivery has been 2.8-4.6 M DP/s in every configuration ever run.

## 2. Why this is the whole game

```
rounds        = 1.8 * 2^n / w
DPs per round = beta * w
total DPs     = 1.8 * beta * 2^n          <- independent of w, theta, alpha
runtime       = 1.8 * beta * 2^n / D_total
```

The number of distinguished points the whole job must route is fixed by `n` alone.
**Routing throughput is the attack's speed, one for one.**  At `--n 40` that is
`1.58e13` points: 41 days at 4.5 M/s, 5.8 hours at the 760 M/s the walkers already
produce.

Three consequences:

- **Difficulty cannot buy anything.**  `total f-evals = 1.8 * beta * 2^n * L` with
  `L = sqrt(2^n / w) / alpha`, so lowering `theta` cuts the DP rate and raises the
  f-work by the same factor.  `runtime * DP rate` is a constant.
- **Memory makes it worse.**  DP work is flat in `w`; f-work falls as `1/sqrt(w)`.  A
  node with 1 TB and 256 cores is by construction one whose walkers are nearly free
  and whose router is everything.  Using the full 1 TB instead of 4 GB multiplies the
  DP rate by `sqrt(250) = 16`.
- **Scale makes it worse too.**  `w = ram_per_node * N / 8`, so
  `D_node = R_f * alpha * sqrt(w / 2^n)` grows as `sqrt(N)`.  The funnel tightens as
  the machine gets bigger, in both directions at once.

Where one comm thread at `C = 4.5 M DP/s` is enough, with `alpha = 2.5` and
`R_f = 1.4e10 f/s` per node (507 walkers x 28 M, SMT on, dictionary still empty), from
`w <= 2^n * (C / (alpha * R_f))^2`:

| RAM/node | dictionary usable from | nodes before the router saturates, at n = 64 |
|---|---|---|
| 4 GB | n >= 55 | ~600 |
| 1 TB | n >= 63 | **~2** |

That last cell is the finding.  With the memory this machine actually has, the
one-comm-thread design runs out at about two nodes.

## 3. Where the comm thread's time goes

`route_walker_queues` (`engine.hpp:82`) sweeps every walker queue with `pop_bulk(64)`,
then `deliver_local` (`engine.hpp:74`) writes each 24-byte point into the ring of the
shard that owns it.  Measured cost per *delivered* point, 32 inserters: **220 ns**.

- **Distance.**  One thread pulls from up to 507 queues spread over ~16 L3 domains and
  both sockets, then scatters into rings whose consumers are equally spread.  Nearly
  every line it touches arrives from another die.
- **True sharing on the ring data.**  With 32 inserters the queues are never full
  (`DROP_INSERTERQ == 0`), so the consumer sits right behind each write; 24-byte points
  put 2-3 of them per 64-byte line, and half the consumers are across the socket.
- **One `tail.store(release)` per point** (`spsc.hpp:44`), which each idle inserter
  re-reads from its `in.empty()` spin.  `SPSCQueue` has `pop_bulk` and no `push_bulk`;
  that asymmetry is the bug.
- **Two runtime 64-bit divisions per point** in `deliver_local` -- `x % n_nodes` and
  `x / n_nodes % inserters_per_node` -- with `n_nodes == 1`, because both are runtime
  members.  The inserter adds three more (`inserter.hpp:53`, `:54`, `:101`).

All but the last scale *with* the inserter count, which is why 4 -> 32 inserters made
the comm thread 7.5x slower instead of faster.

Two smaller effects, for completeness.  Thread placement (`parameters.hpp:143`) is
NUMA-aware and SMT-blind: it takes the next free CPU of a NUMA node, so once it passes
that node's primary threads it hands walkers the siblings of the comm thread and of
every inserter.  Measured by leaving the siblings idle (`--walkers-per-node 250`):
worth ~10% on those threads, while SMT is worth +66% to the walkers -- so the fix is
SMT-aware placement, not fewer threads.  And an inserter probe was believed to cost
~900 ns against a 1 GB shard, against ~150 ns for one dependent random access with a page
walk, which made huge pages and software pipelining (the TODO at `inserter.hpp:82`) look
worth trying.  **That figure was wrong**: measured on its own, a probe costs **58 ns** and
a shard retires 17.2 M/s (§8.1).  The ~900 ns was the pipeline around the probe, not the
probe.

## 4. What is not the answer

- **More inserters.**  Two targets, both unreachable.  Matching what the walkers
  produce needs `I/W = alpha * (R_f/R_p) * sqrt(w / 2^n) = 1.66` -- more inserters than
  walkers, 158 of the 254 workers.  Matching only what the comm thread delivers needs
  `31M / 1.1M = 28`.  Going from 4 to 32 made things worse, per section 3: the balance
  is right, the queue implementation punishes it.
- **Fewer walkers.**  The 9-thread run wins only because it stops overwhelming the
  router.  It is a diagnosis, not a configuration to ship.
- **Walkers delivering into the shards themselves** (MPSC, bypassing the comm thread).
  That removes the `1/N` local fraction and leaves `(N-1)/N` exactly where it was.  A
  single-node trick, worthless at scale, not worth a protocol change.
- **Turning `alpha` down.**  Section 2: it trades one for one.

## 5. Ideas worth trying

### A. One MPI rank per L3 domain -- zero code

`R` ranks per node is `R` comm threads, and the routing work splits `R` ways; whether
a point's shard is on this node stops mattering.  Binding a rank to an L3 domain makes
its walkers, its staging buffers and its shards all domain-local, so the per-router
rate should rise as well as the count.

```bash
mpirun -np 16 --map-by l3cache --bind-to l3cache \
    build/examples/double_speck64_demo --n 40 --ram 250M \
    --inserters-per-node 4 --walkers-per-node 8
```

`--ram 250M x 16` keeps `w` at the same 4 GB total, so the insertion rate is directly
comparable.  Check the banner reports 32 CPUs in the affinity mask, not 512: nothing
coordinates placement across ranks, so ranks that inherit the same mask will silently
pin to the same CPUs.

Cost: send buffers grow as `O(ranks^2)` (16 ranks x 1000 nodes is 1.15 GB per rank),
a round barrier over `16N` ranks instead of `N`, and `16N` progress reports converging
on rank 0.  Fine to ~10-20k ranks.  This contradicts the "one MPI rank per node"
convention in `CLAUDE.md`, which is exactly what makes it interesting.

### B. Router threads, one per L3 domain -- the in-process version of A

A router owns the walker queues, the staging buffers and the inserters of its domain;
thread 0 keeps every MPI call (`THREAD_FUNNELED` is why the funnel exists) but does
**no per-point work**: it posts an `Isend` for a full buffer, and hands an arriving
buffer to the router that owns it.

Two things this has to get right:

- **The destination key must name a single consumer.**  Today `dst = x % n_nodes`
  (`engine.hpp:92`), so an arriving buffer holds points for every shard on the target
  node and someone must scan and scatter it -- `O(D)` on one thread, on the *receiving*
  side too, which is half the problem and would survive the change untouched.  Key on
  `(node, domain)` or on the global shard index instead.
- **Buffer memory sets `buffer_capacity`.**  `routers * destinations * capacity`, and
  it wants to be L3-resident: ~32 MB per router, so ~850 points per buffer at 1600
  global shards.  At 16k shards that falls to 85, too small to amortise a message, and
  the key has to become two-level (`node`, then domain within it) to keep the
  destination count at `N * R`.

Over A this buys: no shared-memory copy for intra-node traffic, one dictionary and one
round barrier per node instead of 16, and `O(R^2 N)` buffers instead of `O((16N)^2)`.
It costs a real protocol rewrite -- a `ROUTER` role, new channels, the `TAG_POINTS`
addressing and wire layout, and a new stage in the drain (routers must quiesce after
the walkers and before the buffers flush).

**Measure A first.**  It is the same architecture implemented by the MPI runtime, and
it bounds what B is worth before anyone spends a protocol rewrite on it.

**Measured: §8.4.  A works** -- 3.5x at one rank per socket, 11.1x at 16 ranks on a
64-PU node, where 97.7% of the points found reach a shard and the funnel disappears.
B's remaining value is the shared-memory copy and the buffer count, not the bottleneck.

### C. Constants, worth having either way

Each multiplies whatever parallel routing gives.

- `push_bulk` on `SPSCQueue`: bucket a whole sweep by destination and push each bucket
  once -- ~48 points per `tail.store` instead of one, and contiguous writes per ring.
  **Done, and worth nothing: §8.3.**  It moves where the points are lost, not how many
  arrive.
- Multiply-shift reciprocals for the five runtime divisions of section 3; all five
  divisors are fixed at construction.
- `MADV_HUGEPAGE` on the shard (which means replacing `vector<u64> A`, since its
  zero-fill is also the NUMA first touch), and the software-pipelined probe.
  **Abandon both: §8.1 measures the probe at 58 ns, 17.2 M/s per shard, and the loop
  already overlaps its misses.**

## 6. Accounting that hides all this

- **A round closes on points found, not inserted** (`controller.hpp:96`).  A starved
  run burns rounds against an almost-empty dictionary and reports them complete.  The
  round report now prints a `ROUTED` line -- points found per second, points *inserted*
  per second, the percentage that reached a shard, and the dictionary's load in
  insertions per slot -- so a starved round is visible at a glance.  The closing rule
  itself is unchanged.
- ~~**`controller.hpp:179` divides `BAD_PROBE` by `N_DP`**~~ -- **fixed**: the probe
  failure percentage is now `BAD_PROBE / N_PROBE`.  It had been reading 3.4% where the
  truth was 97%.  The other four percentages on that line are walker-side, so `ndp` is
  right for them.
- ~~There is still **no way to measure a shard probe in isolation**~~ -- **there is
  now**: `probe_benchmark()` in `benchmark.hpp`, called by `double_speck64_bench` when it
  is given a `--ram` (optional there; two lines to wire into another driver).  It builds
  `inserters_per_node` shards of the size and at the
  placement the engine would give them and hammers each with pseudo-random endpoints,
  reporting probes/s per quarter-`w` of load.  `--beta` bounds how far it fills.  It
  does not exercise the hit path, so it measures the probe, not the collision queue.
- **And no way to price routing before rewriting it** -- **there is now**:
  `staging_benchmark()` in `benchmark.hpp`, next to the probe benchmark and asked for by the
  same `--ram`.  It prices §5's producer-side routing: the rank's walker threads, pinned
  where the engine puts them, each staging pseudo-random points into a shared buffer per
  destination, swept over the fan-out such a router would face (`n_nodes *
  inserters_per_node` destinations, in powers of four until the buffers no longer fit the
  budget).  Three ways per row -- no fan-out at all (the floor), one atomic reservation and
  one scattered store per point, and the same through one private cache line per destination
  -- and two point sizes, three words and two, since the destination index implies the low
  bits of `x` and a two-word point puts four in a write-combining line instead of two.  A row
  yields two different numbers: the node rate, which is what replaces one comm thread's
  4.2 M DP/s, and the per-thread nanoseconds, which are the tax on a walker.  It never seals a
  buffer nor sends one, so it prices the steady state and not the handoff to the funnel.

## 7. Numbers to re-measure

**Re-measured on a different node in §8**, which supersedes section 1 for anything but
`grdix-4` itself.  The original note stands:

The resolver is now batched, so walkers stay near their
empty-dictionary rate much further into a round than they did when these runs were
taken -- which means they will offer *more* points and reach the router's ceiling
sooner, not later.  Everything in section 1 should be re-run before it is quoted.

Worth having per node, once: `double_speck64_bench` (the scalar/vector ratio -- 5.9x
on a laptop), the comm thread's delivered rate at 1, 4 and 16 inserters, and a probe
microbenchmark once one exists.

## 8. Re-measured on grvingt-52, and the funnel bracketed on both sides

`grvingt-52` (Grid'5000, Nancy): 2 x Xeon Gold 6130 (Skylake-SP), 32 cores / 64 PUs, 2
NUMA nodes, **2** L3 domains of 22 MB, 192 GB -- a quarter of `grdix-4`.  `-O3 -DNDEBUG`,
September 2026, `double_speck64_demo --n 40 --ram 4G --seed 1337 --nrounds 1` throughout:
the same operating point as section 1, `w = 5.0e8` slots, auto-tuned `1/theta == 18.76`,
`beta*w == 4.0e9` points per round.  "inserted" is the `ROUTED` line of the round report
(§6), i.e. `N_PROBE`: what actually reached a shard.

### 8.1 The three legs, each measured on its own

| leg | capacity | how |
|---|---|---|
| what the walkers offer | **98 M DP/s** (61 walkers) | round report, `--inserters-per-node 2` |
| what one comm thread delivers | **4.2 M DP/s** | round report, `--inserters-per-node 8` |
| what 8 shards can absorb | **142 M probe/s** | `double_speck64_bench --ram 4G --inserters-per-node 8` |

One walker thread evaluates `vfg` at 79 M f/s with every PU busy (184 M/s alone --
Skylake drops its clock hard under AVX-512), and SMT is nearly free here: 79 M/s per
thread at 32 *and* at 64 processes, so the node's walkers are worth **5.0 G f/s**.

The probe benchmark (new, §6) is what changes the picture: **17.2 M probe/s per shard,
58 ns a probe, flat from an empty shard to one insertion per slot, and linear in the
shard count** -- 142 M/s at 8 shards, 277 M/s at 16, on 4 GB total either way.  Section 3
guessed ~900 ns against a 1 GB shard and was 15x pessimistic: the probes of a batch are
independent, so the hardware already keeps a dozen misses in flight.  **Huge pages and
the software-pipelined probe (§5.C, third bullet) have nothing left to win.**

The middle leg is 23x below what is offered to it and 34x below what waits behind it.

### 8.2 The inserter sweep, re-run

Walkers fill the mask (`63 - I`), so the thread count is 64 in every row:

| inserters | round | offered | inserted | coll/s | dict load |
|---|---|---|---|---|---|
| 1 | 52.0 s | 77 M/s | 2.7 M/s | 59 K | 0.28 |
| 2 | 40.9 s | 98 M/s | 3.3 M/s | 69 K | 0.27 |
| 4 | 64.3 s | 62 M/s | 3.5 M/s | 118 K | 0.45 |
| 8 | 80.4 s | 50 M/s | 4.2 M/s | 205 K | 0.68 |
| 16 | 127.5 s | 31 M/s | 4.2 M/s | 274 K | 1.08 |
| 32 | 759.5 s | 5 M/s | 3.2 M/s | 462 K | 4.84 |

**The same 4 M DP/s ceiling as `grdix-4`, on a machine with a quarter of the cores and a
fifth of the RAM.**  A ceiling that does not move with the machine is not a fact about
the machine: it is one thread's ~250 ns per point, times one thread.

The rows still disagree about the best `I`, and for the reason section 1 gives: a round
closes on points *found*, so a starved run buys its collisions by running long enough to
overfill an almost-empty dictionary.  Read §8.4 instead.

### 8.3 `push_bulk`: implemented, and worth nothing

§5.C's first bullet is in the tree (`SPSCQueue::push_bulk`, per-shard buckets of 128 in
`CommThread`).  Same seed, same configuration, before -> after:

| inserters | inserted | walker-queue drops | inserter-queue drops |
|---|---|---|---|
| 2 | 3.26 -> 3.4 M/s | 3.04 G -> 3.41 G | 0.83 G -> 0.40 G |
| 8 | 4.24 -> 4.2 M/s | 2.58 G -> 2.29 G | 1.09 G -> 1.35 G |
| 16 | 4.24 -> 4.0 M/s | **2.30 G -> 0.098 G** | **1.16 G -> 3.19 G** |

It does exactly what it was meant to do and it buys nothing.  At 16 shards it moves the
loss wholesale from the walker queues to the inserter queues -- the comm thread now
drains 47 walker queues nearly losslessly -- and the delivered rate does not move.  **The
release store per point was never the cost.**  Kept in the tree so that nobody spends the
afternoon on it twice.

### 8.4 One rank per socket, then per 16 PUs, then per 4: idea A is the answer

Zero code.  64 PUs, 4 GB of dictionary, 4.0e9 points per round and the same `theta` in
every row; only the number of ranks -- hence of comm threads -- changes.  Every row of
this table, the `R = 1` baseline included, is the current tree (§8.3).  Launched with
`--map-by ppr:R:node:PE=(64/R) --bind-to hwthread --ram (4G/R)`, and every rank's banner
confirms a disjoint affinity mask over exactly one NUMA node.

| ranks/node | per rank | walkers | offered | **inserted** | reached | coll/s | dict load | round |
|---|---|---|---|---|---|---|---|---|
| 1 | 1+8+55 | 55 | 47.7 M/s | 4.2 M/s | 8.9% | 197 K | 0.71 | 83.9 s |
| 2 | 1+4+27 | 54 | 95.7 M/s | **14.8 M/s** | 15.4% | 1.05 M | 1.24 | 41.9 s |
| 4 | 1+2+13 | 52 | 102.9 M/s | **26.4 M/s** | 25.7% | 2.60 M | 2.05 | 38.9 s |
| 8 | 1+1+6 | 48 | 93.1 M/s | **38.8 M/s** | 41.7% | 4.83 M | 3.34 | 43.0 s |
| 16 | 1+1+2 | 32 | 47.9 M/s | **46.8 M/s** | **97.7%** | 7.96 M | 7.82 | 83.5 s |

**3.5x, 6.3x, 9.2x, 11.1x on the delivered rate; 5x, 13x, 25x, 40x on collisions per
second**, which is the objective.  Per comm thread that is 4.2, 7.4, 6.6, 4.8, 2.9 M DP/s
-- a router congested at its operating point is worth ~5 M DP/s, an uncongested one 8 to
12 (§8.5), and `R` of them are worth about `R` times that.  The walkers gain as well,
16.4 -> 40.5 M f/s each, because they stop pushing into a saturated queue.

The last two rows are the interesting ones.  At `R = 16` the funnel is **gone**: 97.7% of
every point found reaches a shard, the round fills the dictionary to the `beta = 8` it was
asked for, and it does so in the same 83.5 s that `R = 1` spent filling 9% of it.  `R = 16`
also gives up 23 of the 55 walkers to comm threads and still wins 40x, but it is the end
of the road on this node: offered (47.9 M/s) and inserted (46.8 M/s) have met, so the
walkers are the constraint again and a 17th rank would only take another walker away.

Intra-node traffic goes through MPI shared memory for `(R-1)/R` of the points and costs
nothing measurable: `DROP_OUT` stays below 6e5 against 4.0e9 points, at every `R`.

**This is the configuration to run today**, and it contradicts the "one MPI rank per node"
convention in `CLAUDE.md`, as section 5 warned it would.  **Idea B (router threads) is now
worth its protocol rewrite only where one rank per L3 domain is impossible**: it saves the
shared-memory copy, one dictionary and one barrier per node, and `O((RN)^2)` buffers --
none of which is the bottleneck A already removes.

### 8.5 What one comm thread can really do, and what makes it collapse

Turn the *walker count* down and leave everything else at what the auto-tuning wants, so
that the search's physics is identical in every row and only the pressure on one comm
thread changes.  One rank, 8 shards, 4 GB, `--beta 1` (a round of `w == 5.0e8` points):

| walkers | offered | inserted | reached a shard | probe/s per shard | dict load | round |
|---|---|---|---|---|---|---|
| 1 | 3.6 M/s | 3.6 M/s | **99.9%** | 445 K | 1.00 | 140.2 s |
| 2 | 5.6 M/s | 5.6 M/s | **99.8%** | 703 K | 1.00 | 88.8 s |
| 4 | 7.6 M/s | 7.5 M/s | **99.7%** | 943 K | 1.00 | 66.2 s |
| 8 | 8.0 M/s | 8.0 M/s | **99.7%** | 994 K | 1.00 | 62.7 s |
| 16 | 12.6 M/s | 9.0 M/s | 71.5% | 1.1 M | 0.72 | 39.7 s |
| 32 | 40.7 M/s | 12.1 M/s | 29.8% | 1.5 M | 0.30 | 12.4 s |
| 55 | 60.6 M/s | 12.1 M/s | 19.9% | 1.5 M | 0.20 | 8.3 s |
| 55, `--beta 8` | 47.7 M/s | **4.2 M/s** | 8.9% | 528 K | 0.71 | 83.9 s |

**One comm thread is lossless to 8 M DP/s and peaks at 12 M/s.**  A single walker feeds it
3.6 M DP/s and loses one point in 700; 55 walkers feed it 60 M/s and lose four points in
five.  Nothing is gained by the flood: the round ends with a fifth of the dictionary
filled instead of all of it.

The last row is the one to stare at.  Same rank, same 55 walkers, same 8 shards, same
difficulty -- only `beta`, i.e. how full the dictionary gets before the round closes -- and
delivery falls from **12.1 to 4.2 M DP/s** while the load rises from 0.20 to 0.71.  A
fuller shard means more probes hit, and every hit is a mutex acquisition on
`shared.coll_q` plus a resolution a walker has to pay for.  **A third of what one router
can do is being spent on the collision path, at the operating point that matters.**

The same collapse seen through `--difficulty` instead, all else fixed (55 walkers, 8
shards, 4 GB, `--beta 0.25`):

| trail length | offered | inserted | reached a shard | probe/s per shard | dict load |
|---|---|---|---|---|---|
| 1375 | 292 K/s | 288 K/s | **98.6%** | 36 K | 0.25 |
| 688 | 811 K/s | 332 K/s | 40.9% | 41 K | 0.10 |
| 344 | 2.4 M/s | 541 K/s | 22.8% | 68 K | 0.06 |
| 172 | 4.9 M/s | 1.0 M/s | 21.2% | 129 K | 0.05 |
| 86 | 11.3 M/s | 2.7 M/s | 24.0% | 340 K | 0.06 |
| 43 | 24.2 M/s | **7.2 M/s** | 29.7% | 899 K | 0.07 |

One warning about this second table: `theta` is not a free knob for a measurement.  At
`alpha = 2.5` the auto-tuning wants `L == 18.76`, and by `L == 688` the trails merge so
thoroughly that **74% of probes hit** -- against 5% at the auto value -- although the
shard is a tenth full.  Those rows measure trail merging as much as routing, which is why
the walker sweep above is the one to quote.

### 8.6 Where that leaves the diagnosis

Section 1's headline survives -- routing is the bottleneck -- and it is now bracketed
rather than inferred.  What is new:

- **It is not the outbound store** (§8.3), **not the probe** (§8.1: 58 ns, and it
  pipelines itself), and **not memory bandwidth** (277 M probe/s over 16 shards, linear
  in the count).  It is one thread standing in the path of every point.
- **`R` comm threads are worth about `R` times one**, measured to 16 ranks on one node
  (§8.4), and the fix needs no code at all.  Section 2's arithmetic can be re-read with
  `C = 5 M DP/s per rank` rather than per node.
- **After that, the collision path -- not the dictionary.**  §8.5 prices it: the same 55
  walkers and 8 shards deliver 12.1 M DP/s over a round that fills a fifth of the
  dictionary and 4.2 M DP/s over one that fills 0.71 of it, **a factor of 3 lost to the
  hit path alone**.  In-engine inserters run 30x below the 17.2 M probe/s they are capable
  of, and the only structure they share is `shared.coll_q`: one mutex, every inserter
  pushing, every walker polling and popping.  The `I = 32` row of §8.2 (walkers down to
  3.8 M f/s each, 0.70*w collisions in one round) is the same effect from the other side.
  A per-inserter collision queue is the obvious next experiment, and it is cheap.
- Numbers worth having that this run did not take: the collision queue with per-inserter
  rings; `R` ranks per node at `--n 48` or above, where a round is long enough that
  startup and dictionary zeroing vanish; and idea A on a node with more than two L3
  domains, where `R` can grow without taking the last walkers away.  The staging sweep of
  §6 has since been run on `grdix-14`: **§9**.

## 9. Producer-side staging measured: the funnel is not a hard problem

`grdix-14`, 256 cores / 512 PUs, 503 walker threads, `-O3 -DNDEBUG`, September 2026:
`double_speck64_bench --n 40 --ram 4G --beta 0.25 --inserters-per-node 8`, the
`staging_benchmark()` of §6.  Rates are for the whole node; `dest` is the fan-out
`n_nodes * inserters_per_node` a producer-side router would face.  Three runs: the node's
full walker count (§9.1), the thread count of a 40-core node (§9.2), and the buffer capacity
turned down (§9.3), which is where the best figure of all of them is -- **5.1 G points/s,
1200x one comm thread.**

### 9.1 503 walkers: the ten-node machine's shape

| dest | shared buf | private/thread | local | direct 3w | line 3w | direct 2w | **line 2w** |
|---|---|---|---|---|---|---|---|
| 1 | 0.0 MB | 0.1 kB | 98 G/s | 40 M/s | 81 M/s | 40 M/s | 157 M/s |
| 4 | 0.2 MB | 0.3 kB | 115 G/s | 66 M/s | 158 M/s | 62 M/s | 311 M/s |
| 16 | 0.8 MB | 1.0 kB | 106 G/s | 112 M/s | 264 M/s | 95 M/s | 520 M/s |
| 64 | 3.1 MB | 4.1 kB | 105 G/s | 173 M/s | 432 M/s | 149 M/s | 976 M/s |
| 256 | 12.6 MB | 16 kB | 98 G/s | 412 M/s | 1.2 G/s | 409 M/s | **2.6 G/s** |
| 1024 | 50 MB | 66 kB | 112 G/s | 926 M/s | 2.2 G/s | 924 M/s | **4.1 G/s** |
| 4096 | 201 MB | 262 kB | 96 G/s | 841 M/s | 1.1 G/s | 1.2 G/s | 2.5 G/s |
| 16384 | 805 MB | 1.0 MB | 94 G/s | 934 M/s | 1.1 G/s | 1.2 G/s | 1.8 G/s |

**Against 4.2 M DP/s through one comm thread (§8), the walkers stage their own points at
1.8 to 4.1 G/s: two to three orders of magnitude.**  Sections 1 and 2 said routing
throughput is the attack's speed one for one; this says routing stops being the thing that
sets it.  What the walkers of this node can *offer* is 760 M DP/s (§1), so staging has 2.4x
to 5x of headroom over the producers and **the constraint goes back to the walkers**, which
is where it belongs.

Four readings:

- **The per-thread column is a tax, not a ceiling.**  At the 320 destinations of ten
  256-core nodes, a point costs its walker 192 ns against the ~670 ns it spends walking the
  trail that produced it: a 29% tax on walker throughput, paid by every core instead of
  serialised through one.  That is the whole difference from the funnel, and it is why the
  node rate and not the per-thread cost is what compares to §8's 4.2 M DP/s.
- **The two-word point is worth 2x, not a third.**  1.2 -> 2.6 G/s at 256 destinations,
  1.1 -> 1.8 G/s at 16384: a write-combining line holds four compressed points instead of
  two, which halves both the reservations and the shared-buffer traffic.  The destination
  index implies the low `log2(dest)` bits of `x` and a length saturates at 8 bits, so this
  costs nothing but a wire format.  **It is part of the design, not an optimisation of it.**
- **`direct` -- one `fetch_add` and one scattered store per point, no private state at all
  -- is nearly as good at large fan-out and flat in it**: 0.9 to 1.2 G/s from 1024
  destinations to 16384, where the write-combining line only wins 1.5x.  The line earns its
  keep at *small* fan-out, where it is worth 6.4x (409 M/s -> 2.6 G/s at 256).  So the line
  is what the ten-node machine needs and the thousand-node machine can take or leave.
- **The knee past 1024 destinations is not fan-out** -- it is these buffers leaving L3, and
  §9.3 removes it by making them smaller.  §9.2 rules fan-out out on its own: the same sweep
  with 36 walkers, over a *larger* footprint, has no knee at all.

### 9.2 36 walkers: the thousand-node machine's cell

Same node and command, `--walkers-per-node 36`.  A thousand nodes of 40 cores is ~8000
destinations reached by ~36 walkers, so this is that cell's thread count and fan-out on
borrowed hardware -- not that machine, which has roughly 3x this one's memory bandwidth per
core.  The smaller private side lets the sweep reach 65536 destinations.

| dest | local | direct 3w | line 3w | direct 2w | **line 2w** | ns/point |
|---|---|---|---|---|---|---|
| 1 | 15.2 G/s | 22 M/s | 45 M/s | 22 M/s | 90 M/s | 399 |
| 4 | 13.9 G/s | 30 M/s | 68 M/s | 30 M/s | 133 M/s | 271 |
| 16 | 14.3 G/s | 45 M/s | 106 M/s | 43 M/s | 220 M/s | 164 |
| 64 | 17.3 G/s | 79 M/s | 185 M/s | 72 M/s | 386 M/s | 93 |
| 256 | 17.3 G/s | 170 M/s | 340 M/s | 141 M/s | 642 M/s | 56 |
| 1024 | 17.1 G/s | 347 M/s | 523 M/s | 357 M/s | **902 M/s** | 40 |
| 4096 | 17.3 G/s | 517 M/s | 636 M/s | 540 M/s | **992 M/s** | 36 |
| 16384 | 17.3 G/s | 607 M/s | 688 M/s | 631 M/s | **1.0 G/s** | 36 |
| 65536 | 15.4 G/s | 632 M/s | 660 M/s | 662 M/s | **1.0 G/s** | 36 |

**Flat at 1.0 G/s and 36 ns a point from 1024 destinations to 65536**, over a footprint that
grows 64x across those four rows (2.1 GB of shared buffers and 151 MB of private lines at
65536).  Fan-out is free once there is enough of it to spread the reservations; the whole
cost of routing a point, at any scale either target machine will reach, is 36 ns of the
producing walker.

That cell has to route 150 M DP/s -- §2 at `n = 51` with 40 TB of dictionary, where
`D_node = R_f / L` is at once what the walkers produce and what has to be routed.  Staging
delivers 1.0 G/s: **6.7x the whole node's production, 240x one comm thread.**  `direct`
alone -- no private lines, one `fetch_add` and one store per point -- delivers 662 M/s
there, so even the crudest version of §5 clears this
machine; the write-combining line is worth 1.6x at that fan-out and 4.6x at 256, which is
what earns it on the ten-node machine.

The staging tax on a walker lands near 30% of its throughput in both configurations, from
opposite directions: 192 ns against 600 ns of walking at `L = 16.9` on ten large nodes, 36 ns
against ~104 ns at `L = 7.4` on a thousand small ones.  The tax scales as `1/L`, so it is
worst exactly where the dictionary is largest -- the same shape as §2's "memory makes it
worse", now costing 30% instead of two orders of magnitude.

### 9.3 Capacity: the staging set wants to fit the caches, and then 5 G/s

Same node and command as §9.1 -- 503 walkers -- with `--buffer 128` instead of the default
1500, so a destination's buffer holds 128 points rather than 2048.  Footprints are the
two-word ones; `line 2w @2048` repeats §9.1 for comparison.

| dest | shared buf | private/thread | direct 3w | line 3w | direct 2w | **line 2w** | line 2w @2048 |
|---|---|---|---|---|---|---|---|
| 256 | 0.5 MB | 16 kB | 410 M/s | 1.1 G/s | 351 M/s | 2.3 G/s | 2.6 G/s |
| 1024 | 2.1 MB | 66 kB | 919 M/s | 2.0 G/s | 680 M/s | 3.2 G/s | 4.1 G/s |
| 4096 | 8.4 MB | 262 kB | 1.3 G/s | 2.5 G/s | 1.1 G/s | 3.8 G/s | 2.5 G/s |
| 16384 | 33.6 MB | 1.0 MB | 1.7 G/s | 2.8 G/s | 1.8 G/s | **5.1 G/s** | 1.8 G/s |
| 65536 | 134 MB | 4.2 MB | 991 M/s | 1.2 G/s | 1.4 G/s | 1.8 G/s | -- |

**1.8 -> 5.1 G/s at 16384 destinations from shrinking the shared buffers 16x**, so §9.1's
collapse was those buffers leaving L3 (537 MB at the old capacity, 34 MB at this one) and not
memory bandwidth.  The peak of every run so far is that cell: **5.1 G points/s, 98 ns of one
walker per point, 1200x what one comm thread delivers.**

- **A small capacity is not universally better**: at 256 destinations it costs 2.6 -> 2.3
  G/s, because those buffers already fitted and shrinking them only shortens the messages.
  The rule is `dest * capacity * point_bytes` around the size of L3 -- which is exactly
  §5.B's arithmetic, against a *cache* budget rather than a memory one.
- **The new knee at 65536 is the private side**: one line per destination is `dest * 64 B`
  per thread, and 1.0 MB at 16384 destinations still runs at 5.1 G/s while 4.2 MB at 65536
  falls to 1.8.  So **one write-combining line per destination tops out around
  `dest = L2 / 64`, some 16K destinations on this hardware** -- and both target machines are
  under it, at 320 and ~8000.  Nothing needs a two-level key.
- **The capacity is bounded from below by the funnel, and that is what §5.B was really
  afraid of.**  A node's message rate is its point rate over the capacity, whatever the
  destination count: 760 M DP/s at 128 points a message is 5.9 M messages/s, far past what
  one thread can post.  Taking ~1 M `MPI_Isend`/s as that thread's budget -- an estimate, and
  the next thing worth its own microbenchmark -- the floor is `capacity >= 760` on ten large
  nodes and `>= 150` on a thousand small ones.  **The two constraints leave a wide window**:
  at `capacity = 1024` the ten-node machine stages into 5 MB and at 256 the thousand-node
  machine into 33 MB, both far inside the tens of MB this sweep runs at full speed.  128
  points is where the sweep peaks, not what to ship.
- **The ten-node machine is contention-limited, not fan-out-limited**, and the fix is a knob
  it already has.  At 320 destinations 503 walkers reserve from 1.6 lines each and get 2.6
  G/s; at 1024 they get 4.1.  The destination count *is* `n_nodes * inserters_per_node`, so
  raising `--inserters-per-node` from 32 to ~100 buys the fan-out that removes the
  contention -- and §8.1 measured the probe rate linear in the shard count, so the shards do
  not mind.

### 9.4 What this settles

This retires §5.B's second worry.  It feared that buffer memory would force
`buffer_capacity` down as the shard count grew, until a message was too small to amortise;
the capacity does come down as the shard count grows, exactly as §5.B computed, and §9.3
shows there is a wide window where it satisfies both ends -- large enough that the funnel
posts ~1 M messages/s rather than 6 M, small enough that the buffers stay tens of MB and
staging runs at full speed.  Neither target machine is near either edge.  What survives of
§5.B is only its first bullet -- the destination key must name a single consumer -- and that
is now the whole design: key the message on the global shard index, let every walker stage
into it with one cache line per destination, size `buffer_capacity` so the buffers fit L3,
and
leave the funnel nothing but per-message work.
