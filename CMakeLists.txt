cmake_minimum_required(VERSION 3.20)

project(strata VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(STRATA_ENABLE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)

add_library(strata_core STATIC
    src/model.cpp
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

if(STRATA_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(strata_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(strata_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(strata-sim apps/strata_sim.cpp)
target_link_libraries(strata-sim PRIVATE strata_core)

include(CTest)
if(BUILD_TESTING)
    add_executable(strata-tests
        tests/test_main.cpp
        tests/test_model.cpp
        tests/test_q4.cpp
        tests/test_trace.cpp
        tests/test_predictor.cpp
        tests/test_residency.cpp
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

install(TARGETS strata-sim RUNTIME DESTINATION bin)
