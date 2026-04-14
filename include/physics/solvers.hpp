#pragma once

#include <string>

#include "physics/benchmark.hpp"
#include "physics/field.hpp"

namespace physics {

BenchmarkResult run_naive(const SimulationConfig& config, const InitialState& initial);

void run_naive_dump(const SimulationConfig& config, const InitialState& initial,
                    const std::string& output_path);

}  // namespace physics
