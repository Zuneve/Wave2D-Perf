cmake_minimum_required(VERSION 3.20)

project(wave2d LANGUAGES CXX)

option(WAVE2D_FAST_MATH   "Enable aggressive floating-point optimizations" ON)
option(WAVE2D_NATIVE_ARCH "Tune CPU code for the host microarchitecture"   ON)
option(WAVE2D_WITH_CUDA     "Build optional CUDA integrator (cuda-cn-adi)"          OFF)
option(WAVE2D_WITH_CUSPARSE "Build cuSPARSE batched-tridiagonal opponent (GPU)"     OFF)
option(WAVE2D_WITH_MAGMA    "Build MAGMA batched-dense-LU opponent (GPU)"           OFF)

# LAPACK-based reference integrator (lapack-cn-adi). On by default on Apple,
# where it links against the Accelerate framework's built-in LAPACK.
if(APPLE)
    option(WAVE2D_WITH_LAPACK "Build LAPACK reference integrator (lapack-cn-adi)" ON)
else()
    option(WAVE2D_WITH_LAPACK "Build LAPACK reference integrator (lapack-cn-adi)" OFF)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(wave2d_core
    src/benchmark.cpp
    src/config.cpp
    src/field.cpp
    src/solvers/cn_adi_solver.cpp
    src/solvers/cn_adi_cuda.cpp
    src/solvers/cn_adi_lapack.cpp
    src/solvers/cn_adi_cusparse.cpp
    src/solvers/cn_adi_magma.cpp
    src/solvers/euler_solver.cpp
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

# Any GPU opponent needs the CUDA language enabled.
if(WAVE2D_WITH_CUDA OR WAVE2D_WITH_CUSPARSE OR WAVE2D_WITH_MAGMA)
    enable_language(CUDA)
    set_target_properties(wave2d_core PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

if(WAVE2D_WITH_CUDA)
    target_sources(wave2d_core PRIVATE src/solvers/cn_adi_cuda.cu)
    target_compile_definitions(wave2d_core PRIVATE WAVE2D_ENABLE_CUDA=1)
endif()

# cuSPARSE ships with the CUDA Toolkit (CUDA::cusparse).
if(WAVE2D_WITH_CUSPARSE)
    find_package(CUDAToolkit REQUIRED)
    target_sources(wave2d_core PRIVATE src/solvers/cn_adi_cusparse.cu)
    target_compile_definitions(wave2d_core PRIVATE WAVE2D_ENABLE_CUSPARSE=1)
    target_link_libraries(wave2d_core PUBLIC CUDA::cusparse)
endif()

# MAGMA must be installed separately. Set MAGMA_ROOT (or have it on the prefix
# path) so the headers/library are found; it also needs CUDA + a BLAS/LAPACK.
if(WAVE2D_WITH_MAGMA)
    find_package(CUDAToolkit REQUIRED)
    find_path(MAGMA_INCLUDE_DIR magma_v2.h
        HINTS ${MAGMA_ROOT} ENV MAGMA_ROOT PATH_SUFFIXES include)
    find_library(MAGMA_LIBRARY NAMES magma
        HINTS ${MAGMA_ROOT} ENV MAGMA_ROOT PATH_SUFFIXES lib lib64)
    if(NOT MAGMA_INCLUDE_DIR OR NOT MAGMA_LIBRARY)
        message(FATAL_ERROR "MAGMA not found. Set -DMAGMA_ROOT=/path/to/magma")
    endif()
    target_sources(wave2d_core PRIVATE src/solvers/cn_adi_magma.cu)
    target_compile_definitions(wave2d_core PRIVATE WAVE2D_ENABLE_MAGMA=1)
    target_include_directories(wave2d_core PRIVATE ${MAGMA_INCLUDE_DIR})
    target_link_libraries(wave2d_core PUBLIC ${MAGMA_LIBRARY} CUDA::cudart CUDA::cublas CUDA::cusparse)
endif()

if(WAVE2D_WITH_LAPACK)
    target_compile_definitions(wave2d_core PRIVATE WAVE2D_ENABLE_LAPACK=1)
    if(APPLE)
        target_link_libraries(wave2d_core PUBLIC "-framework Accelerate")
    else()
        find_package(LAPACK REQUIRED)
        target_link_libraries(wave2d_core PUBLIC LAPACK::LAPACK)
    endif()
endif()

add_executable(wave2d src/main.cpp)
target_link_libraries(wave2d PRIVATE wave2d_core)
