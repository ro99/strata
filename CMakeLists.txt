cmake_minimum_required(VERSION 3.20)

project(strata VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>"
    "$<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic;-Wconversion;-Wshadow>"
)

option(STRATA_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
option(STRATA_ENABLE_CUDA "Build the native CUDA backend when a CUDA compiler is available" ON)
# Production rank-local TP2 decode. OFF keeps every normal build supported and
# unlinked against NCCL; the runtime then rejects --rank-local-decode before
# model loading rather than falling back to centralized execution.
option(STRATA_ENABLE_NCCL "Build the rank-local TP2 decode topology against NCCL" OFF)
set(STRATA_NCCL_INCLUDE_DIR "" CACHE PATH
    "NCCL include directory used by the rank-local decode topology")
set(STRATA_NCCL_LIBRARY "" CACHE FILEPATH
    "NCCL shared library used by the rank-local decode topology")

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

# Phase 1 (docs/experiments/0121): strata_core split into six layered
# targets, dependencies pointing strictly downward:
#   platform -> device -> kernels -> engine -> models -> app
# Files are not moved -- every path below is exactly where it was in the
# single strata_core list. This is a link-graph and enforcement change only;
# scripts/check_layers.py is the boundary enforcement (there is no physical
# per-target include directory yet, so the compiler alone cannot catch a
# violation -- that is phase 3's job once files actually move).
#
# strata_core is kept as a thin alias so every existing
# `target_link_libraries(<binary> PRIVATE strata_core)` line below keeps
# working unchanged; it forwards to strata_app, which pulls in the other
# five PUBLICly through the chain.

add_library(strata_platform STATIC
    src/attention.cpp
    src/checkpoint_io.cpp
    src/compressed_tensors.cpp
    src/diagnostics.cpp
    src/diagnostics_json.cpp
    src/json_cursor.cpp
    src/numa_topology.cpp
    src/numerics.cpp
    src/safetensors.cpp
    src/trace.cpp
    src/worker_pool.cpp
)
target_include_directories(strata_platform PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(strata_platform PUBLIC STRATA_VERSION="${PROJECT_VERSION}")

add_library(strata_device STATIC
    ${STRATA_CUDA_SOURCE}
)
target_include_directories(strata_device PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_device PUBLIC strata_platform)

if(STRATA_HAS_CUDA)
    set_property(TARGET strata_device PROPERTY CUDA_ARCHITECTURES 86 120)
    set_property(TARGET strata_device PROPERTY CUDA_STANDARD 20)
    set_property(TARGET strata_device PROPERTY CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(strata_device PUBLIC STRATA_HAS_CUDA=1)
    target_compile_options(strata_device PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-Wall,-Wextra>
    )
else()
    target_compile_definitions(strata_device PUBLIC STRATA_HAS_CUDA=0)
endif()

add_library(strata_kernels STATIC
    kernels/cpu/q4_reference.cpp
    kernels/cpu/glm_int4.cpp
    kernels/cpu/attention_reference.cpp
)
target_include_directories(strata_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_kernels PUBLIC strata_device)

add_library(strata_engine STATIC
    src/chat_protocol.cpp
    src/placement.cpp
    src/placement_cache.cpp
    src/placement_model.cpp
    src/residency.cpp
    src/route_predictor.cpp
    src/runtime_support.cpp
    src/sampling.cpp
    src/simulator.cpp
    src/tokenizer.cpp
)
target_include_directories(strata_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_engine PUBLIC strata_kernels)

add_library(strata_models STATIC
    src/glm_ops.cpp
    src/gemma4_ops.cpp
    src/gemma4_checkpoint.cpp
    src/gemma4_image.cpp
    src/gemma4_runtime.cpp
    src/glm_runtime.cpp
    src/glm_manifest.cpp
    src/kimi_k3_ops.cpp
    src/kimi_k3_manifest.cpp
    src/kimi_k3_checkpoint.cpp
    src/kimi_k3_kv_cache.cpp
    src/kimi_k3_expert_arena.cpp
    src/kimi_k3_layer.cpp
    src/kimi_k3_runtime.cpp
    src/laguna_ops.cpp
    src/laguna_checkpoint.cpp
    src/laguna_runtime.cpp
    src/inkling_ops.cpp
    src/inkling_checkpoint.cpp
    src/inkling_device.cpp
    src/inkling_runtime.cpp
    src/deepseek_manifest.cpp
    src/deepseek_kv_cache.cpp
    src/dsv4_attention_kv.cpp
    src/deepseek_ops.cpp
    src/deepseek_checkpoint.cpp
    src/deepseek_rank_shard.cpp
    src/dsv4_host_moe_executor.cpp
    src/dsv4_rank_local_topology.cpp
    src/dsv4_rank_local_kv.cpp
    src/dsv4_rank_local_weights.cpp
    src/deepseek_admission.cpp
    src/deepseek_runtime.cpp
    src/model.cpp
    src/checkpoint.cpp
    kernels/cpu/dsv4_fp4_expert.cpp
)
target_include_directories(strata_models PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_models PUBLIC strata_engine)

add_library(strata_app STATIC
    src/runtime.cpp
    src/openai_protocol.cpp
)
target_include_directories(strata_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(strata_app PUBLIC strata_models)

add_library(strata_core INTERFACE)
target_link_libraries(strata_core INTERFACE strata_app)

if(STRATA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    foreach(strata_layer IN ITEMS strata_platform strata_device strata_kernels
                                  strata_engine strata_models strata_app)
        target_compile_options(${strata_layer} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
        target_link_options(${strata_layer} PUBLIC -fsanitize=address,undefined)
    endforeach()
endif()

add_executable(strata-sim apps/strata_sim.cpp)
target_link_libraries(strata-sim PRIVATE strata_core)

add_executable(strata-inspect apps/strata_inspect.cpp)
target_link_libraries(strata-inspect PRIVATE strata_core)

add_executable(strata-tokenize apps/strata_tokenize.cpp)
target_link_libraries(strata-tokenize PRIVATE strata_core)

add_executable(strata-run apps/strata_run.cpp)
target_link_libraries(strata-run PRIVATE strata_core)

add_executable(strata-deepseek-run apps/strata_deepseek_run.cpp)
target_link_libraries(strata-deepseek-run PRIVATE strata_core)

add_executable(strata-gemma4-run apps/strata_gemma4_run.cpp)
target_link_libraries(strata-gemma4-run PRIVATE strata_core)

add_executable(strata-chat apps/strata_chat.cpp)
target_link_libraries(strata-chat PRIVATE strata_core)

add_executable(strata-server apps/strata_server.cpp)
target_link_libraries(strata-server PRIVATE strata_core)

add_executable(strata-host-expert-probe apps/strata_host_expert_probe.cpp)
target_link_libraries(strata-host-expert-probe PRIVATE strata_core)

add_executable(strata-numa-probe apps/strata_numa_probe.cpp)
target_link_libraries(strata-numa-probe PRIVATE strata_core)

add_executable(strata-dsv4-host-expert-probe apps/strata_dsv4_host_expert_probe.cpp)
target_link_libraries(strata-dsv4-host-expert-probe PRIVATE strata_core)

add_executable(strata-dsv4-expert-row-batch-probe
    apps/strata_dsv4_expert_row_batch_probe.cpp)
target_link_libraries(strata-dsv4-expert-row-batch-probe PRIVATE strata_core)

add_executable(strata-mhc-prepack-probe apps/strata_mhc_prepack_probe.cpp)
target_link_libraries(strata-mhc-prepack-probe PRIVATE strata_core)

add_executable(strata-dsv4-index-probe apps/strata_dsv4_index_probe.cpp)
target_link_libraries(strata-dsv4-index-probe PRIVATE strata_core)

add_executable(strata-dsv4-attention-probe apps/strata_dsv4_attention_probe.cpp)
target_link_libraries(strata-dsv4-attention-probe PRIVATE strata_core)

add_executable(strata-kimi-expert-probe apps/strata_kimi_expert_probe.cpp)
target_link_libraries(strata-kimi-expert-probe PRIVATE strata_core)

add_executable(strata-kimi-decode-probe apps/strata_kimi_decode_probe.cpp)
target_link_libraries(strata-kimi-decode-probe PRIVATE strata_core)

add_executable(strata-kimi-compute-probe apps/strata_kimi_compute_probe.cpp)
find_library(NUMA_LIBRARY_PROBE numa REQUIRED)
target_link_libraries(strata-kimi-compute-probe PRIVATE strata_core ${NUMA_LIBRARY_PROBE})
add_executable(strata-inkling-probe apps/strata_inkling_probe.cpp)
target_link_libraries(strata-inkling-probe PRIVATE strata_core)

add_executable(strata-laguna-profile apps/strata_laguna_profile.cpp)
target_link_libraries(strata-laguna-profile PRIVATE strata_core)

add_executable(strata-laguna-stage-probe apps/strata_laguna_stage_probe.cpp)
target_link_libraries(strata-laguna-stage-probe PRIVATE strata_core)

if(STRATA_HAS_CUDA)
    find_library(NUMA_LIBRARY numa REQUIRED)
    add_executable(strata-topology-probe apps/strata_topology_probe.cu)
    target_link_libraries(strata-topology-probe PRIVATE strata_core ${NUMA_LIBRARY})
    set_property(TARGET strata-topology-probe PROPERTY CUDA_ARCHITECTURES 86 120)
    set_property(TARGET strata-topology-probe PROPERTY CUDA_STANDARD 20)
    set_property(TARGET strata-topology-probe PROPERTY CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strata-topology-probe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;-Xcompiler=-Wall,-Wextra>
    )

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(strata_device PUBLIC CUDA::cublas)

    add_executable(strata-dsv4-bf16-gemm-probe
        apps/strata_dsv4_bf16_gemm_probe.cu)
    target_link_libraries(strata-dsv4-bf16-gemm-probe PRIVATE
        CUDA::cublas CUDA::cudart)
    set_property(TARGET strata-dsv4-bf16-gemm-probe PROPERTY
        CUDA_ARCHITECTURES 86)
    set_property(TARGET strata-dsv4-bf16-gemm-probe PROPERTY CUDA_STANDARD 20)
    set_property(TARGET strata-dsv4-bf16-gemm-probe PROPERTY
        CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strata-dsv4-bf16-gemm-probe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )

    add_executable(strata-dsv4-fp8-tensor-gemm-probe
        apps/strata_dsv4_fp8_tensor_gemm_probe.cu)
    target_link_libraries(strata-dsv4-fp8-tensor-gemm-probe PRIVATE
        CUDA::cudart)
    set_property(TARGET strata-dsv4-fp8-tensor-gemm-probe PROPERTY
        CUDA_ARCHITECTURES 86)
    set_property(TARGET strata-dsv4-fp8-tensor-gemm-probe PROPERTY
        CUDA_STANDARD 20)
    set_property(TARGET strata-dsv4-fp8-tensor-gemm-probe PROPERTY
        CUDA_STANDARD_REQUIRED ON)
    target_compile_options(strata-dsv4-fp8-tensor-gemm-probe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )

    if(STRATA_ENABLE_NCCL)
        # NCCL is rarely installed system-wide. In practice it arrives with the
        # CUDA toolkit or as the nvidia-nccl-cu12 wheel inside a Python
        # environment, so search those before asking the user to locate it by
        # hand.
        #
        # Detection resolves a header and a library from the *same* root and
        # never mixes two installations: a glob over python* matches every
        # interpreter version present, and pairing one version's header with
        # another's library is an ABI mismatch that would surface as a link or
        # runtime failure rather than as a configure error.
        #
        # Auto-detection runs only when neither option is set. Supplying one
        # half is a user decision that detection must not complete, for the
        # same mismatch reason.
        if(STRATA_NCCL_INCLUDE_DIR AND NOT STRATA_NCCL_LIBRARY)
            message(FATAL_ERROR
                "STRATA_NCCL_INCLUDE_DIR is set but STRATA_NCCL_LIBRARY is not. "
                "Set both, from the same NCCL installation, or neither to "
                "auto-detect.")
        endif()
        if(STRATA_NCCL_LIBRARY AND NOT STRATA_NCCL_INCLUDE_DIR)
            message(FATAL_ERROR
                "STRATA_NCCL_LIBRARY is set but STRATA_NCCL_INCLUDE_DIR is not. "
                "Set both, from the same NCCL installation, or neither to "
                "auto-detect.")
        endif()
        if(NOT STRATA_NCCL_INCLUDE_DIR AND NOT STRATA_NCCL_LIBRARY)
            # Ordered most specific first: an environment the user has
            # deliberately activated outranks a system-wide install.
            #
            #   nvidia-nccl-cu12 wheel   <env>/lib/python*/site-packages/nvidia/nccl
            #   NCCL tar / "local installer"  NCCL_HOME, /usr/local/nccl*, /opt/nccl*
            #   apt or yum packages (libnccl2, libnccl-dev)   /usr
            #   CUDA toolkit bundle      CUDA_HOME, /usr/local/cuda*
            #
            # See the NCCL installation guide for the package and tar layouts.
            file(GLOB STRATA_NCCL_ROOTS
                "$ENV{VIRTUAL_ENV}/lib/python*/site-packages/nvidia/nccl"
                "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia/nccl")
            list(APPEND STRATA_NCCL_ROOTS "$ENV{NCCL_HOME}" "$ENV{CUDA_HOME}")
            file(GLOB STRATA_NCCL_TARBALL_ROOTS
                "/usr/local/nccl*" "/opt/nccl*")
            list(APPEND STRATA_NCCL_ROOTS ${STRATA_NCCL_TARBALL_ROOTS})
            list(APPEND STRATA_NCCL_ROOTS
                "${CUDAToolkit_LIBRARY_ROOT}" "/usr/local/cuda")
            file(GLOB STRATA_NCCL_CUDA_ROOTS "/usr/local/cuda-*")
            list(APPEND STRATA_NCCL_ROOTS ${STRATA_NCCL_CUDA_ROOTS})
            list(APPEND STRATA_NCCL_ROOTS "/usr/local" "/usr")
            foreach(nccl_root IN LISTS STRATA_NCCL_ROOTS)
                if(NOT nccl_root OR NOT EXISTS "${nccl_root}/include/nccl.h")
                    continue()
                endif()
                # The wheel ships only the versioned soname, with no
                # unversioned libnccl.so symlink, so the bare "nccl" name that
                # find_library derives is not enough on its own. The distro
                # packages do ship the symlink but place it in the multiarch
                # directory, which CMAKE_LIBRARY_ARCHITECTURE names for
                # whichever architecture this is -- x86_64 and aarch64 both
                # matter here, since NCCL ships for Grace and Jetson too.
                unset(STRATA_NCCL_ROOT_LIBRARY CACHE)
                find_library(STRATA_NCCL_ROOT_LIBRARY
                    NAMES nccl libnccl.so.2
                    HINTS "${nccl_root}/lib" "${nccl_root}/lib64"
                          "${nccl_root}/lib/${CMAKE_LIBRARY_ARCHITECTURE}"
                          "${nccl_root}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib"
                    NO_DEFAULT_PATH)
                if(STRATA_NCCL_ROOT_LIBRARY)
                    set(STRATA_NCCL_INCLUDE_DIR "${nccl_root}/include")
                    set(STRATA_NCCL_LIBRARY "${STRATA_NCCL_ROOT_LIBRARY}")
                    break()
                endif()
            endforeach()
            unset(STRATA_NCCL_ROOT_LIBRARY CACHE)
        endif()
        # One diagnosis for both halves, naming what was searched and how to
        # override it. Reporting only "requires STRATA_NCCL_INCLUDE_DIR" leaves
        # a reader who has never installed NCCL with nowhere to go.
        if(NOT STRATA_NCCL_INCLUDE_DIR OR NOT EXISTS "${STRATA_NCCL_INCLUDE_DIR}/nccl.h"
           OR NOT STRATA_NCCL_LIBRARY OR NOT EXISTS "${STRATA_NCCL_LIBRARY}")
            message(FATAL_ERROR
                "STRATA_ENABLE_NCCL=ON but NCCL was not found.\n"
                "  header: ${STRATA_NCCL_INCLUDE_DIR}\n"
                "  library: ${STRATA_NCCL_LIBRARY}\n"
                "Searched the active Python environment, the CUDA toolkit, "
                "NCCL_HOME, and CUDA_HOME.\n"
                "Install it with 'pip install nvidia-nccl-cu12', or point at an "
                "existing copy:\n"
                "  -DSTRATA_NCCL_INCLUDE_DIR=<dir containing nccl.h>\n"
                "  -DSTRATA_NCCL_LIBRARY=<path to libnccl.so>\n"
                "Both must come from the same NCCL installation. Configure "
                "without -DSTRATA_ENABLE_NCCL=ON to build everything except "
                "rank-local TP2 decode.")
        endif()
        message(STATUS "NCCL header: ${STRATA_NCCL_INCLUDE_DIR}")
        message(STATUS "NCCL library: ${STRATA_NCCL_LIBRARY}")
        get_filename_component(STRATA_NCCL_LIBRARY_DIR "${STRATA_NCCL_LIBRARY}" DIRECTORY)
        # Compiled into strata_models rather than a separate archive: the
        # executor needs CudaBackend (strata_device) and deepseek_runtime.cpp
        # (strata_models) needs the executor, so a distinct library between
        # them would be a cycle. Phase 1 note: this is itself a case worth
        # watching once headers move in phase 3 -- a device-tier kernel that
        # only a models-tier consumer uses is exactly the shape that makes a
        # clean device/models split hard, and it's already here today.
        target_sources(strata_models PRIVATE
            kernels/cuda/dsv4_rank_local_layer_executor.cu)
        target_include_directories(strata_models PRIVATE "${STRATA_NCCL_INCLUDE_DIR}")
        target_link_libraries(strata_models PUBLIC "${STRATA_NCCL_LIBRARY}" ${CMAKE_DL_LIBS} ${NUMA_LIBRARY})
        set_target_properties(strata_models PROPERTIES BUILD_RPATH "${STRATA_NCCL_LIBRARY_DIR}")
        # PUBLIC so the runtime, CLI, chat and server all agree on whether the
        # rank-local topology exists at all.
        target_compile_definitions(strata_models PUBLIC STRATA_HAS_NCCL=1)
        set_property(TARGET strata_models PROPERTY CUDA_ARCHITECTURES 86 120)
        set_property(TARGET strata_models PROPERTY CUDA_STANDARD 20)
        set_property(TARGET strata_models PROPERTY CUDA_STANDARD_REQUIRED ON)
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(strata-tests
        tests/test_main.cpp
        tests/test_attention.cpp
        tests/test_chat_protocol.cpp
        tests/test_compressed_tensors.cpp
        tests/test_cuda_backend.cpp
        tests/test_glm_ops.cpp
        tests/test_gemma4_ops.cpp
        tests/test_gemma4_checkpoint.cpp
        tests/test_gemma4_image.cpp
        tests/test_glm_int4.cpp
        tests/test_kimi_k3_ops.cpp
        tests/test_kimi_k3_manifest.cpp
        tests/test_kimi_k3_checkpoint.cpp
        tests/test_kimi_k3_kv_cache.cpp
        tests/test_kimi_k3_expert_arena.cpp
        tests/test_kimi_k3_backbone.cpp
        tests/test_kimi_k3_layer.cpp
        tests/test_kimi_k3_tokenizer.cpp
        tests/test_laguna_ops.cpp
        tests/test_laguna_checkpoint.cpp
        tests/test_inkling_ops.cpp
        tests/test_inkling_checkpoint.cpp
        tests/test_inkling_runtime.cpp
        tests/test_deepseek_manifest.cpp
        tests/test_deepseek_rank_shard.cpp
        tests/test_dsv4_rank_local_topology.cpp
        tests/test_dsv4_rank_local_kv.cpp
        tests/test_deepseek_kv_cache.cpp
        tests/test_deepseek_ops.cpp
        tests/test_deepseek_host_expert.cpp
        tests/test_deepseek_diagnostics.cpp
        tests/test_deepseek_runtime.cpp
        tests/test_model.cpp
        tests/test_openai_protocol.cpp
        tests/test_placement.cpp
        tests/test_q4.cpp
        tests/test_trace.cpp
        tests/test_predictor.cpp
        tests/test_residency.cpp
        tests/test_runtime.cpp
        tests/test_sampling.cpp
        tests/test_safetensors.cpp
        tests/test_tokenizer.cpp
        tests/test_worker_pool.cpp
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
    # Phase 0's equivalence oracle (brief 05, F2): re-runs the committed
    # Gemma 4 layer-hash fixture and diffs it against a fresh
    # strata-gemma4-run --layer-hash-trace capture. Before this test existed,
    # tests/fixtures/gemma4/layer-hash-trace.json was referenced by nothing
    # outside its own README -- re-verification was a human typing a diff by
    # hand, and nothing failed if it never happened. Guarded on the pinned
    # checkpoint's presence, same as every other real-checkpoint test in this
    # suite; SKIP_RETURN_CODE makes ctest report an absent checkpoint as
    # skipped, not passed.
    add_test(
        NAME strata-equivalence-gemma4
        COMMAND ${CMAKE_COMMAND} -E env python3
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_equivalence.py
    )
    set_tests_properties(strata-equivalence-gemma4 PROPERTIES SKIP_RETURN_CODE 77)
endif()

install(TARGETS strata-sim strata-inspect strata-tokenize strata-run strata-deepseek-run strata-gemma4-run strata-chat strata-server
        strata-host-expert-probe strata-kimi-expert-probe RUNTIME DESTINATION bin)
if(STRATA_HAS_CUDA)
    install(TARGETS strata-topology-probe RUNTIME DESTINATION bin)
endif()
