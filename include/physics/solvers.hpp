#pragma once

#include <string>

#include "physics/benchmark.hpp"
#include "physics/field.hpp"

namespace physics {

BenchmarkResult run_euler(const SimulationConfig& config, const InitialState& initial);
BenchmarkResult run_cn_adi(const SimulationConfig& config, const InitialState& initial);
BenchmarkResult run_cn_adi_cuda(const SimulationConfig& config, const InitialState& initial);

void run_euler_dump(const SimulationConfig& config, const InitialState& initial,
                    const std::string& output_path);
void run_cn_adi_dump(const SimulationConfig& config, const InitialState& initial,
                     const std::string& output_path);

}  // namespace physics
