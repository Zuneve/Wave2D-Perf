#include "physics/field.hpp"

#include <cmath>

namespace physics {

InitialState make_initial_state(const SimulationConfig& config) {
    InitialState initial;
    initial.nx = config.nx;
    initial.ny = config.ny;
    initial.psi.resize(config.nx * config.ny, {0.0f, 0.0f});
    initial.potential.resize(config.nx * config.ny, 0.0f);

    const float center_x = 0.5f * static_cast<float>(config.nx - 1) * config.dx;
    const float center_y = 0.5f * static_cast<float>(config.ny - 1) * config.dy;
    const float packet_center_x = 0.25f * static_cast<float>(config.nx - 1) * config.dx;
    const float packet_center_y = center_y;
    const float barrier_x = 0.60f * static_cast<float>(config.nx - 1) * config.dx;
    const float barrier_half_width = 0.15f;
    const float slit_half_gap = 0.75f;
    const float sigma2 = config.packet_sigma * config.packet_sigma;

    for (std::size_t y = 0; y < config.ny; ++y) {
        const float fy = static_cast<float>(y) * config.dy;
        for (std::size_t x = 0; x < config.nx; ++x) {
            const float fx = static_cast<float>(x) * config.dx;
            const std::size_t idx = y * config.nx + x;

            const float rel_x = fx - center_x;
            const float rel_y = fy - center_y;
            const float harmonic = config.potential_strength * (rel_x * rel_x + rel_y * rel_y);

            float barrier = 0.0f;
            if (std::abs(fx - barrier_x) < barrier_half_width && std::abs(fy - center_y) > slit_half_gap) {
                barrier = 40.0f * config.potential_strength;
            }

            initial.potential[idx] = harmonic + barrier;

            if (x == 0 || x + 1 == config.nx || y == 0 || y + 1 == config.ny) {
                initial.psi[idx] = {0.0f, 0.0f};
                continue;
            }

            const float dx = fx - packet_center_x;
            const float dy = fy - packet_center_y;
            const float envelope = std::exp(-(dx * dx + dy * dy) / (2.0f * sigma2));
            const float phase = config.packet_kx * fx + config.packet_ky * fy;
            initial.psi[idx] = {envelope * std::cos(phase), envelope * std::sin(phase)};
        }
    }

    return initial;
}

double l2_norm(const std::vector<std::complex<float>>& psi) {
    double sum = 0.0;
    for (const auto& value : psi) {
        sum += static_cast<double>(std::norm(value));
    }
    return sum;
}

float max_amplitude(const std::vector<std::complex<float>>& psi) {
    float maximum = 0.0f;
    for (const auto& value : psi) {
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

}  // namespace physics
