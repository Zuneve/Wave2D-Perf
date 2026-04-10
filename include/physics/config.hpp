#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace physics {

enum class SolverKind {
    all,
    naive_aos,
    soa_simd,
    threads,
    omp,
    cuda
};

struct SimulationConfig {
    std::size_t nx = 1024;
    std::size_t ny = 1024;
    std::size_t steps = 400;
    std::size_t warmup_steps = 5;
    float dt = 1.0e-4f;
    float dx = 0.1f;
    float dy = 0.1f;
    float mass = 1.0f;
    float potential_strength = 0.02f;
    float packet_sigma = 1.25f;
    float packet_kx = 8.0f;
    float packet_ky = 0.0f;
    std::size_t tile_x = 128;
    std::size_t tile_y = 32;
    int threads = 0;
    SolverKind solver = SolverKind::all;
};

SimulationConfig parse_args(int argc, char** argv);
std::string to_string(SolverKind kind);
std::vector<SolverKind> expand_requested_solvers(SolverKind requested);

}  // namespace physics
