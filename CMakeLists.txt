cmake_minimum_required(VERSION 3.24)

project(physicsProject LANGUAGES CXX)

include(CheckLanguage)

option(PHYSICS_ENABLE_OPENMP "Enable OpenMP backend when available" ON)
option(PHYSICS_ENABLE_CUDA "Enable CUDA backend when a CUDA toolkit is available" ON)
option(PHYSICS_ENABLE_FAST_MATH "Enable aggressive floating-point optimizations" ON)
option(PHYSICS_ENABLE_NATIVE_ARCH "Tune CPU code for the host microarchitecture" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PHYSICS_SOURCES
    src/benchmark.cpp
    src/config.cpp
    src/field.cpp
    src/solvers/naive_solver.cpp
    src/solvers/soa_solver.cpp
)

if(PHYSICS_ENABLE_CUDA)
    check_language(CUDA)
endif()

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    list(APPEND PHYSICS_SOURCES src/solvers/cuda_solver.cu)
else()
    list(APPEND PHYSICS_SOURCES src/solvers/cuda_stub.cpp)
endif()

add_library(physics_core ${PHYSICS_SOURCES})

target_include_directories(physics_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(physics_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(physics_core PRIVATE /W4 /permissive-)
    if(PHYSICS_ENABLE_FAST_MATH)
        target_compile_options(physics_core PRIVATE /fp:fast)
    endif()
else()
    target_compile_options(physics_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
    if(PHYSICS_ENABLE_FAST_MATH)
        target_compile_options(physics_core PRIVATE -ffast-math)
    endif()
    if(PHYSICS_ENABLE_NATIVE_ARCH)
        target_compile_options(physics_core PRIVATE -march=native)
    endif()
endif()

if(PHYSICS_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(physics_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(physics_core PUBLIC PHYSICS_HAS_OPENMP=1)
    endif()
endif()

if(CMAKE_CUDA_COMPILER)
    target_compile_definitions(physics_core PUBLIC PHYSICS_HAS_CUDA=1)
    set_target_properties(physics_core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

add_executable(physicsProject src/main.cpp)
target_link_libraries(physicsProject PRIVATE physics_core)
