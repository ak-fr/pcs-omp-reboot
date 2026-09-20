# Every driver runs the same engine (MPI + OpenMP), so they all link the same way.  One driver per
# problem: --benchmark measures its f/g instead of searching, so no separate bench binary.
function(mitm_example name)
    add_executable(${name} ${name}.cpp ${ARGN})
    target_include_directories(${name} PRIVATE ../include)
    target_link_libraries(${name} PRIVATE MPI::MPI_C OpenMP::OpenMP_CXX Threads::Threads hwloc::hwloc fmt::fmt)
    # Keep <mpi.h> from pulling in the deprecated MPI-2 C++ bindings; we never use them.
    target_compile_definitions(${name} PRIVATE OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX _MPICC_H)
endfunction()


###### Speck

mitm_example(double_speck64_demo)


###### DES

mitm_example(double_DES_demo usuba_des.cpp)
target_link_libraries(double_DES_demo PRIVATE OpenSSL::Crypto)


###### AES

mitm_example(double_aes_demo aes.c)


###### SHA256

mitm_example(sha2_claw_demo sha256.c)
mitm_example(sha2_collision_demo sha256.c)



###### The Router: its test suite and its benchmark

mitm_example(router_test)
target_compile_definitions(router_test PRIVATE ROUTER_PARANOID)
mitm_example(router_bench)

# The Router: one rank, two ranks, four ranks -- 8 spinning threads in all on an 8-core laptop (12 at
# np 4, the minimum of one service thread, one producer and one consumer per rank).  np 1 pins (one rank
# owns the machine); np 2 and np 4 put several ranks on the one host, whose masks overlap, so they pass
# --no-bind (the Router refuses to pin overlapping masks) -- pinned multi-rank needs --map-by numa on a
# real cluster.
add_test(NAME router_np1 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         $<TARGET_FILE:router_test> --producers 4 --consumers 3 --quiet)
add_test(NAME router_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
         $<TARGET_FILE:router_test> --producers 2 --consumers 1 --no-bind --quiet)
add_test(NAME router_np4 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4 ${MPIEXEC_PREFLAGS}
         $<TARGET_FILE:router_test> --producers 1 --consumers 1 --no-bind --quiet)


###### The direct engine: a planted claw over 2^20 preimages, found or the exit status says so

add_test(NAME direct_np1 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         $<TARGET_FILE:double_speck64_demo> --n 20 --ram 256K --seed 1
         --producers-per-node 4 --dicts-per-node 3 --quiet)
add_test(NAME direct_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
         $<TARGET_FILE:double_speck64_demo> --n 20 --ram 256K --seed 1
         --producers-per-node 2 --dicts-per-node 1 --no-bind --quiet)


###### PCS: the same planted claw and a planted collision, over trails this time

# PCS needs ~2^n/w versions of the function to single out one claw among the 2^n the problem has, so
# these are sized for a difficulty worth having: the whole dictionary is 32000 slots either way, which
# puts 1/theta near 2.3 and the trails a couple of points long.  A larger dictionary over so small a
# domain drives theta towards 1, where every point is distinguished and the trails stop being trails.
# No --nrounds: the search runs until it finds the pair, and the exit status is whether it did.
add_test(NAME pcs_np1 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         $<TARGET_FILE:double_speck64_demo> --n 20 --ram 256K --seed 1 --engine pcs
         --producers-per-node 4 --dicts-per-node 3 --quiet)
add_test(NAME pcs_np2 COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS}
         $<TARGET_FILE:double_speck64_demo> --n 20 --ram 128K --seed 1 --engine pcs
         --producers-per-node 2 --dicts-per-node 1 --no-bind --quiet)

# the scalar path (vlen == 1, so the one-candidate-at-a-time resolver) and the collision wrapper
add_test(NAME pcs_collision COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
         $<TARGET_FILE:sha2_collision_demo> --n 20 --ram 256K --seed 1 --engine pcs
         --producers-per-node 4 --dicts-per-node 3 --quiet)
