# AVX512F + AVX512BW
message(STATUS "Testing whether AVX-512 code can be used")
set (HAVE_AVX512BW OFF)

# 1st try, no options
try_run(avx512_runs avx512_compiles
    ${PROJECT_BINARY_DIR}/config
    ${PROJECT_SOURCE_DIR}/config/avx512.c)
if(avx512_compiles)
    if (avx512_runs MATCHES FAILED_TO_RUN)
        message(STATUS "Testing whether AVX-512 code can be used -- No (compiles but does not run)")
        else()
            message(STATUS "Testing whether AVX-512 code can be used -- Yes")
            set (HAVE_AVX512BW ON)
        endif()
    else()
        # 2nd try with options
        try_run(avx512_runs avx512_compiles
            ${PROJECT_BINARY_DIR}/config
            ${PROJECT_SOURCE_DIR}/config/avx512.c
            COMPILE_DEFINITIONS -mavx512f -mavx512bw)
        if(avx512_compiles)
            if (avx512_runs MATCHES FAILED_TO_RUN)
                message(STATUS "Testing whether AVX-512 code can be used -- No (compiles with -mavx512f -mavx512bw but does not run)")
            else()
                message(STATUS "Testing whether AVX-512 code can be used -- Yes, with -mavx512f -mavx512bw")
                set (HAVE_AVX512BW ON)
                add_compile_options(-mavx512f -mavx512bw)
            endif()
        else()
            message(STATUS "Testing whether avx code can be used -- No (cannot compile)")
        endif()
    endif()