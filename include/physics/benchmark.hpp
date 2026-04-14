#pragma once

#include <cstddef>
#include <string>

#include "physics/config.hpp"

namespace physics {

struct BenchmarkResult {
    std::string name;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t steps = 0;
    double seconds = 0.0;
    double mlups = 0.0;
    double l2_norm = 0.0;
    float max_amplitude = 0.0f;
};

BenchmarkResult run_benchmark(const SimulationConfig& config);

}  // namespace physics
