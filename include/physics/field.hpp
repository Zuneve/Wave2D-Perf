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

struct SoAField {
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t pitch = 0;
    std::vector<float> real;
    std::vector<float> imag;
    std::vector<float> potential;
};

InitialState make_initial_state(const SimulationConfig& config);
SoAField make_soa_field(std::size_t nx, std::size_t ny);
void fill_soa_from_initial(const InitialState& initial, SoAField& field);
double l2_norm(const std::vector<std::complex<float>>& psi);
double l2_norm(const SoAField& field);
float max_amplitude(const std::vector<std::complex<float>>& psi);
float max_amplitude(const SoAField& field);

}  // namespace physics
