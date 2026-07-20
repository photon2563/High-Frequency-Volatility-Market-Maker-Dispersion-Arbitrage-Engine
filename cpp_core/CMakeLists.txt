cmake_minimum_required(VERSION 3.16)
project(davinci_vol_core VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─────────────────────────────────────────────────────────────────────────────
# Core static library
# ─────────────────────────────────────────────────────────────────────────────

add_library(davinci_core STATIC
    src/pricing/black_scholes.cpp
    src/pricing/implied_vol.cpp
)

target_include_directories(davinci_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(davinci_core PRIVATE
    -Wall -Wextra -Wpedantic -O2
)

# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (GoogleTest via FetchContent)
# ─────────────────────────────────────────────────────────────────────────────

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # Prevent GoogleTest from overriding our compiler/linker options
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Test targets
    set(TEST_SOURCES
        tests/test_pricing.cpp
        tests/test_market_making.cpp
        tests/test_hedging.cpp
        tests/test_dispersion.cpp
        tests/test_execution.cpp
    )

    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})
        target_link_libraries(${test_name} PRIVATE davinci_core GTest::gtest_main)
        target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# ─────────────────────────────────────────────────────────────────────────────
# Python bindings (pybind11 via FetchContent)
# ─────────────────────────────────────────────────────────────────────────────

option(BUILD_PYTHON_BINDINGS "Build pybind11 Python module" OFF)
if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(davinci_py bindings/pybind_module.cpp)
    target_link_libraries(davinci_py PRIVATE davinci_core)
    target_include_directories(davinci_py PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()
