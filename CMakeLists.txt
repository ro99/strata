cmake_minimum_required(VERSION 3.20)

project(strata VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STRATA_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
option(STRATA_ENABLE_CUDA "Build the native CUDA backend when a CUDA compiler is available" ON)

include(CheckLanguage)
if(STRATA_ENABLE_CUDA)
    check_language(CUDA)
endif()

if(STRATA_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(STRATA_CUDA_SOURCE kernels/cuda/backend.cu)
    set(STRATA_HAS_CUDA ON)
else()
    set(STRATA_CUDA_SOURCE src/cuda_backend_stub.cpp)
    set(STRATA_HAS_CUDA OFF)
endif()

add_library(strata_core STATIC
    src/checkpoint.cpp
    src/compressed_tensors.cpp
    ${STRATA_CUDA_SOURCE}
    src/glm_ops.cpp
    src/glm_runtime.cpp
    src/glm_manifest.cpp
    src/json_cursor.cpp
    src/model.cpp
    src/safetensors.cpp
    src/tokenizer.cpp
    src/trace.cpp
    src/route_predictor.cpp
    src/residency.cpp
    src/simulator.cpp
    kernels/cpu/q4_reference.cpp
)

target_include_directories(strata_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(strata_core PUBLIC STRATA_VERSION="${PROJECT_VERSION}")
target_compile_options(strata_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
    $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>
)

if(STRATA_HAS_CUDA)
    set_property(TARGET strata_core PROPERTY CUDA_ARCHITECTURES 86 120)
    set_property(TARGET strata_core PROPERTY CUDA_STANDARD 20)
    set_property(TARGET strata_core PROPERTY CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(strata_core PUBLIC STRATA_HAS_CUDA=1)
    target_compile_options(strata_core PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-Wall,-Wextra>
    )
else()
    target_compile_definitions(strata_core PUBLIC STRATA_HAS_CUDA=0)
endif()

if(STRATA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(strata_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(strata-sim apps/strata_sim.cpp)
target_link_libraries(strata-sim PRIVATE strata_core)

add_executable(strata-inspect apps/strata_inspect.cpp)
target_link_libraries(strata-inspect PRIVATE strata_core)

add_executable(strata-tokenize apps/strata_tokenize.cpp)
target_link_libraries(strata-tokenize PRIVATE strata_core)

add_executable(strata-run apps/strata_run.cpp)
target_link_libraries(strata-run PRIVATE strata_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(strata-tests
        tests/test_main.cpp
        tests/test_compressed_tensors.cpp
        tests/test_cuda_backend.cpp
        tests/test_glm_ops.cpp
        tests/test_model.cpp
        tests/test_q4.cpp
        tests/test_trace.cpp
        tests/test_predictor.cpp
        tests/test_residency.cpp
        tests/test_safetensors.cpp
        tests/test_tokenizer.cpp
    )
    target_link_libraries(strata-tests PRIVATE strata_core)
    target_compile_definitions(strata-tests PRIVATE
        STRATA_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
    add_test(NAME strata-tests COMMAND strata-tests)
    add_test(
        NAME strata-sim-smoke
        COMMAND strata-sim
            --trace ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/route_trace_v1.txt
            --policy lease
            --ram 64M
            --vram 32M
            --expert-bytes 4M
            --prefetch 4
            --lease 8
            --json
    )
endif()

install(TARGETS strata-sim strata-inspect strata-tokenize strata-run RUNTIME DESTINATION bin)
