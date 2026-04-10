#pragma once

#include "physics/benchmark.hpp"
#include "physics/field.hpp"

namespace physics {

BenchmarkResult run_naive_aos(const SimulationConfig& config, const InitialState& initial);
BenchmarkResult run_soa_simd(const SimulationConfig& config, const InitialState& initial, bool use_openmp);
BenchmarkResult run_soa_threads(const SimulationConfig& config, const InitialState& initial);
BenchmarkResult run_cuda_backend(const SimulationConfig& config, const InitialState& initial);

}  // namespace physics
