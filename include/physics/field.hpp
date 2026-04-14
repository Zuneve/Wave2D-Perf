#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "physics/config.hpp"

namespace physics {

struct InitialState {
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::vector<std::complex<float>> psi;
    std::vector<float> potential;
};

InitialState make_initial_state(const SimulationConfig& config);
double l2_norm(const std::vector<std::complex<float>>& psi);
float max_amplitude(const std::vector<std::complex<float>>& psi);

}  // namespace physics
