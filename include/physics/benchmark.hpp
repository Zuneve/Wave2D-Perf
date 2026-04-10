#pragma once

#include <string>
#include <vector>

#include "physics/config.hpp"

namespace physics {

struct BenchmarkResult {
    std::string name;
    double seconds = 0.0;
    double mlups = 0.0;
    double l2_norm = 0.0;
    float max_amplitude = 0.0f;
};

std::vector<BenchmarkResult> run_requested_benchmarks(const SimulationConfig& config);
bool cuda_backend_available();

}  // namespace physics
