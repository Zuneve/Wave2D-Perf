#pragma once

#include <cstddef>
#include <string>

namespace physics {

enum class IntegratorKind {
    cn_adi,
    explicit_euler,
    cuda_cn_adi,
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
    int threads = 0;
    IntegratorKind integrator = IntegratorKind::cn_adi;
    std::size_t dump_every = 0;  // 0 = disabled
    bool sweep = false;
    std::string output;
};

SimulationConfig parse_args(int argc, char** argv);
std::string to_string(IntegratorKind integrator);

}  // namespace physics
