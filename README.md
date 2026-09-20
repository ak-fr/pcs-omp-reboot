# Parallel meet-in-the-middle

Given `f` and `g`, find a **claw** (`f(x0) == g(x1)`) or a **collision**
(`f(x0) == f(x1)`, `x0 != x1`) over a distributed dictionary.

The library is header-only C++17, in `include/`.  `examples/` holds one driver per
cipher: double-Speck64 (the current focus), double-DES, double-AES and SHA256.

**Two engines run today, both built on the Router.**  MPI + OpenMP, one rank per node.
Inside a rank the team is one service thread doing all the MPI, `--dicts-per-node`
threads each owning a shard of a distributed dictionary, and `--producers-per-node`
threads evaluating the functions.  `--engine` picks which one runs (`direct`, the
default, or `pcs`); every driver takes the flag, and neither engine shares a line of
transport with the other.

The **direct** engine is the exhaustive meet-in-the-middle: a round is two phases, `f`
fills the dictionary on a chunk of the domain, then `g` probes it on the whole domain;
`ceil(2^n / (fill * w))` rounds cover the domain, and finding nothing is a proof of
absence.

**PCS** is van Oorschot–Wiener parallel collision search over a dictionary of
distinguished points: the problem becomes one random function per round, and the
walkers' trails end at distinguished points that fill the dictionary.  A round ends on
a count no single node holds, so PCS keeps a controller on rank 0 and a control channel
to reach it; it is Monte-Carlo, so "not found" is never a proof.

The **Router** (`include/router/`) is the transport
underneath both engines: it carries pairs of `u64` from producer threads to globally
numbered consumer threads across ranks, and it owns thread placement: the service thread alone on
the leftmost core, then every consumer and every producer down hwloc's hierarchy to the least-filled
child at each level.  It forms no groups; the one thing it says about its plan is `Router_closeness`,
how deep in the hierarchy two threads sit together, which is what PCS pairs its walkers with its dict
threads by.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=release -DCMAKE_CXX_FLAGS="-mno-avx512f" && make -C build -j
```

Requires **MPI**, **OpenMP**, **OpenSSL** (headers; the DES example checks itself
against it) and **hwloc 2.x** (headers, `libhwloc-dev`): the Router places the threads
with it, and there is no fallback without it.  Binaries land in `build/examples/`.

Two things to know before trusting any number that comes out:

- No `CMAKE_BUILD_TYPE` is set, so there is no `-O` flag and `NDEBUG` is off: the
  default build is unoptimized with live `assert()`s.  Add `-DCMAKE_BUILD_TYPE=Release`
  to benchmark.
- Everything compiles with `-march=native`, so binaries are **not portable across CPU
  generations**.  On a cluster, build on the same architecture as the compute nodes.
  The SIMD path (AVX-512 / AVX2 / scalar) is picked by `#ifdef` guards in
  `include/types.h` that `-march=native` triggers; the probes in `config/*.cmake` set
  `HAVE_*` variables that nothing currently reads.

## Run

Every driver takes the same options (`--help` lists them all).  `--ram` is
**mandatory** for a search and accepts human units.  The exit status says whether the
golden pair was found.  `--benchmark` measures the problem's f/g rate and exits instead
of searching, so it needs no `--ram`.

```bash
mpirun -np 4 --bind-to none build/examples/double_speck64_demo \
       --n 32 --ram 4G --dicts-per-node 2 --producers-per-node 30
```

- `--n` problem size in bits (small == easy), `--seed` (0 == draw one and broadcast it)
- `--engine direct` (default) or `--engine pcs`
- `--ram` dictionary bytes **per node**; `--fill` is the direct engine's fill ratio, so
  `fill * w` entries per round; `--difficulty` / `--alpha` / `--beta` / `--dp-len-bits`
  are PCS's; `--nrounds` gives up after that many rounds (for PCS, which
  never exhausts anything, it is the only way to make it give up)
- `--producers-per-node` / `--dicts-per-node` size the team.  Producers default to
  filling whatever affinity mask the launcher handed the rank, which is why
  `--bind-to none` matters
- **the Router pins the threads itself**, and prints both the planned and the
  measured layout.  `--no-bind` turns pinning off, which is what two ranks sharing a
  host need (the Router refuses to pin overlapping masks)
- `--block`, `--swc`, `--n-recv`, `--inbox`, `--credit` tune the transport;
  `include/router/common.hpp` documents each of them

A smoke test that runs in well under a second:

```bash
mpirun -np 1 --bind-to none build/examples/double_speck64_demo --n 20 --ram 256K \
       --producers-per-node 4 --dicts-per-node 3
```

On a CEA/TGCC-style cluster the launcher is `ccc_mprun` under an `MSUB` batch script;
see `tgcc.sh`.

## Layout

```
include/
  types.h            SIMD vector types, selected by -march=native
  tools.hpp          PRNG (TRIVIUM), timing, hashing, human-readable numbers, Point
  problem.hpp        the interface a cipher implements: f, g, is_good_pair, vfg
  parameters.hpp     Options: every user knob, all defaulted, the Router's among them
  direct/            the direct engine: params, shared, wrappers, dict, producer, engine
                     (direct.hpp is the umbrella)
  pcs/               PCS: params, shared, wrappers, trail, dict, walker, control, engine
                     (pcs.hpp is the umbrella)
  router/            the Router: topology, ring, common, placement, connect, workers, service
                     (router.hpp is the umbrella)
examples/
  driver.hpp             command line + MPI startup, shared by every example
  benchmark.hpp          f/g throughput per rank and across ranks: every driver's --benchmark
  router_driver.hpp      the same for the Router's own two programs
  <cipher>_problem.hpp   f, g, and a planted golden pair
  <cipher>_demo.cpp      run the attack (--benchmark: measure f/s and stop there)
  router_test.cpp        the Router's test suite
  router_bench.cpp       the Router's throughput benchmark
```

A new cipher is a `*_problem.hpp` implementing `f`, `g` and the domain against
`AbstractClawProblem` (or `AbstractCollisionProblem`), plus a ~20-line driver.
Vectorization is opt-in per problem: set `vlen` and provide `vfg` using the `v32` /
`v64` types from `types.h`.  With `vlen == 1` the engine calls `f`/`g` directly and no
`vfg` is needed.

Recurring notation: `n` = domain bits, `m` = range bits, `w` = dictionary slots,
`theta` = proportion of distinguished points, DP = distinguished point.

## Documents

- `PROBLEM.md` — the measurements that showed the single comm thread was the
  bottleneck, which is why the Router exists.
- `FINDINGS.md` — one section per profiling session on Grid'5000.

## Testing

```bash
ctest --test-dir build
```

Eight tests: the Router's suite at one, two and four ranks; the direct engine at one and
two ranks; PCS at one and two ranks; and PCS's collision wrapper (`sha2_collision_demo`,
scalar path).  The engine tests plant a golden pair on a small `--n` and search for it
with `double_speck64_demo` -- the demos `assert` their way to it
(`assert(pb.f(x0) == pb.g(x1))`) and their exit status reports whether they found it.
