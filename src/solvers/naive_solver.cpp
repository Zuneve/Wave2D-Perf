#include "physics/solvers.hpp"

#include <algorithm>
#include <chrono>
#include <complex>
#include <vector>

namespace physics {
namespace {

using Complex = std::complex<float>;

void advance_steps(
    const SimulationConfig& config,
    const std::vector<float>& potential,
    std::vector<Complex>& current,
    std::vector<Complex>& next,
    std::size_t steps) {
    const float alpha = 0.5f / config.mass;
    const Complex laplacian_scale{0.0f, alpha};

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t y = 1; y + 1 < config.ny; ++y) {
            const std::size_t row = y * config.nx;
            const std::size_t north = row - config.nx;
            const std::size_t south = row + config.nx;
            next[row] = {0.0f, 0.0f};
            next[row + config.nx - 1] = {0.0f, 0.0f};

            for (std::size_t x = 1; x + 1 < config.nx; ++x) {
                const std::size_t idx = row + x;
                const Complex center = current[idx];
                const Complex laplacian =
                    current[idx - 1] + current[idx + 1] + current[north + x] + current[south + x] - 4.0f * center;
                const Complex derivative = laplacian_scale * laplacian + Complex{0.0f, -potential[idx]} * center;
                next[idx] = center + config.dt * derivative;
            }
        }

        current.swap(next);
    }
}

}  // namespace

BenchmarkResult run_naive_aos(const SimulationConfig& config, const InitialState& initial) {
    std::vector<Complex> warm_current = initial.psi;
    std::vector<Complex> warm_next(initial.psi.size(), {0.0f, 0.0f});
    advance_steps(config, initial.potential, warm_current, warm_next, config.warmup_steps);

    std::vector<Complex> current = initial.psi;
    std::vector<Complex> next(initial.psi.size(), {0.0f, 0.0f});

    const auto start = std::chrono::steady_clock::now();
    advance_steps(config, initial.potential, current, next, config.steps);
    const auto end = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(end - start).count();
    const double updated_cells = static_cast<double>((config.nx - 2) * (config.ny - 2) * config.steps);

    return {
        .name = "naive_aos",
        .seconds = seconds,
        .mlups = updated_cells / seconds / 1.0e6,
        .l2_norm = l2_norm(current),
        .max_amplitude = max_amplitude(current),
    };
}

}  // namespace physics
