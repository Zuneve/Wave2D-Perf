cmake_minimum_required(VERSION 3.20)

project(wave2d LANGUAGES CXX)

option(WAVE2D_FAST_MATH   "Enable aggressive floating-point optimizations" ON)
option(WAVE2D_NATIVE_ARCH "Tune CPU code for the host microarchitecture"   ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(wave2d_core
    src/benchmark.cpp
    src/config.cpp
    src/field.cpp
    src/solvers/naive_solver.cpp
)

target_include_directories(wave2d_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(wave2d_core PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(wave2d_core PRIVATE /W4 /permissive-)
    if(WAVE2D_FAST_MATH)
        target_compile_options(wave2d_core PRIVATE /fp:fast)
    endif()
else()
    target_compile_options(wave2d_core PRIVATE -O3 -Wall -Wextra -Wpedantic)
    if(WAVE2D_FAST_MATH)
        target_compile_options(wave2d_core PRIVATE -ffast-math)
    endif()
    if(WAVE2D_NATIVE_ARCH)
        target_compile_options(wave2d_core PRIVATE -march=native)
    endif()
endif()

add_executable(wave2d src/main.cpp)
target_link_libraries(wave2d PRIVATE wave2d_core)
