#include "physics/field.hpp"

#include <algorithm>
#include <cmath>

namespace physics {

namespace {

std::size_t padded_pitch(std::size_t nx) {
    constexpr std::size_t alignment_floats = 16;
    return ((nx + alignment_floats - 1) / alignment_floats) * alignment_floats;
}

}  // namespace

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

SoAField make_soa_field(std::size_t nx, std::size_t ny) {
    SoAField field;
    field.nx = nx;
    field.ny = ny;
    field.pitch = padded_pitch(nx);
    field.real.assign(field.pitch * ny, 0.0f);
    field.imag.assign(field.pitch * ny, 0.0f);
    field.potential.assign(field.pitch * ny, 0.0f);
    return field;
}

void fill_soa_from_initial(const InitialState& initial, SoAField& field) {
    for (std::size_t y = 0; y < initial.ny; ++y) {
        const std::size_t src_row = y * initial.nx;
        const std::size_t dst_row = y * field.pitch;
        for (std::size_t x = 0; x < initial.nx; ++x) {
            const auto value = initial.psi[src_row + x];
            field.real[dst_row + x] = value.real();
            field.imag[dst_row + x] = value.imag();
            field.potential[dst_row + x] = initial.potential[src_row + x];
        }
    }
}

double l2_norm(const std::vector<std::complex<float>>& psi) {
    double sum = 0.0;
    for (const auto& value : psi) {
        sum += static_cast<double>(std::norm(value));
    }
    return sum;
}

double l2_norm(const SoAField& field) {
    double sum = 0.0;
    for (std::size_t y = 0; y < field.ny; ++y) {
        const std::size_t row = y * field.pitch;
        for (std::size_t x = 0; x < field.nx; ++x) {
            const double real = field.real[row + x];
            const double imag = field.imag[row + x];
            sum += real * real + imag * imag;
        }
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

float max_amplitude(const SoAField& field) {
    float maximum = 0.0f;
    for (std::size_t y = 0; y < field.ny; ++y) {
        const std::size_t row = y * field.pitch;
        for (std::size_t x = 0; x < field.nx; ++x) {
            const float real = field.real[row + x];
            const float imag = field.imag[row + x];
            maximum = std::max(maximum, std::sqrt(real * real + imag * imag));
        }
    }
    return maximum;
}

}  // namespace physics
