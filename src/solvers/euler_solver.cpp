#include "physics/solvers.hpp"

#include <chrono>
#include <complex>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "physics/field.hpp"

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
    const float inv_dx2 = 1.0f / (config.dx * config.dx);
    const float inv_dy2 = 1.0f / (config.dy * config.dy);

    for (std::size_t step = 0; step < steps; ++step) {
        for (std::size_t y = 1; y + 1 < config.ny; ++y) {
            const std::size_t row   = y * config.nx;
            const std::size_t north = row - config.nx;
            const std::size_t south = row + config.nx;
            next[row] = {0.0f, 0.0f};
            next[row + config.nx - 1] = {0.0f, 0.0f};

            for (std::size_t x = 1; x + 1 < config.nx; ++x) {
                const std::size_t idx    = row + x;
                const Complex center     = current[idx];
                const Complex lap_x = (current[idx - 1] + current[idx + 1] - 2.0f * center) * inv_dx2;
                const Complex lap_y = (current[north + x] + current[south + x] - 2.0f * center) * inv_dy2;
                const Complex laplacian  = lap_x + lap_y;
                const Complex derivative = laplacian_scale * laplacian
                                         + Complex{0.0f, -potential[idx]} * center;
                next[idx] = center + config.dt * derivative;
            }
        }
        current.swap(next);
    }
}

}  // namespace

BenchmarkResult run_euler(const SimulationConfig& config, const InitialState& initial) {
    std::vector<Complex> warm_current = initial.psi;
    std::vector<Complex> warm_next(initial.psi.size(), {0.0f, 0.0f});
    advance_steps(config, initial.potential, warm_current, warm_next, config.warmup_steps);

    std::vector<Complex> current = initial.psi;
    std::vector<Complex> next(initial.psi.size(), {0.0f, 0.0f});

    const auto t0 = std::chrono::steady_clock::now();
    advance_steps(config, initial.potential, current, next, config.steps);
    const auto t1 = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double updated_cells =
        static_cast<double>((config.nx - 2) * (config.ny - 2)) * static_cast<double>(config.steps);

    return {
        .name          = "euler",
        .nx            = config.nx,
        .ny            = config.ny,
        .steps         = config.steps,
        .seconds       = seconds,
        .mlups         = updated_cells / seconds / 1.0e6,
        .l2_norm       = l2_norm(current),
        .max_amplitude = max_amplitude(current),
    };
}

void run_euler_dump(const SimulationConfig& config, const InitialState& initial,
                    const std::string& output_path) {
    if (config.dump_every == 0) {
        throw std::invalid_argument("dump_every must be > 0");
    }

    std::ofstream file(output_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open output file: " + output_path);
    }

    const std::size_t n_frames = config.steps / config.dump_every + 1;
    const uint64_t nx_u         = config.nx;
    const uint64_t ny_u         = config.ny;
    const uint64_t n_frames_u   = n_frames;
    const uint64_t dump_every_u = config.dump_every;
    const float pad             = 0.0f;

    const char magic[8] = {'W','A','V','E','2','D','\0','\0'};
    file.write(magic, 8);
    file.write(reinterpret_cast<const char*>(&nx_u),         8);
    file.write(reinterpret_cast<const char*>(&ny_u),         8);
    file.write(reinterpret_cast<const char*>(&n_frames_u),   8);
    file.write(reinterpret_cast<const char*>(&dump_every_u), 8);
    file.write(reinterpret_cast<const char*>(&config.dx),    4);
    file.write(reinterpret_cast<const char*>(&config.dy),    4);
    file.write(reinterpret_cast<const char*>(&config.dt),    4);
    file.write(reinterpret_cast<const char*>(&pad),          4);

    auto write_frame = [&](const std::vector<Complex>& psi) {
        for (const auto& c : psi) {
            const float prob = c.real() * c.real() + c.imag() * c.imag();
            file.write(reinterpret_cast<const char*>(&prob), sizeof(float));
        }
    };

    std::vector<Complex> current = initial.psi;
    std::vector<Complex> next(initial.psi.size(), {0.0f, 0.0f});

    write_frame(current);

    const std::size_t n_batches = config.steps / config.dump_every;
    for (std::size_t batch = 0; batch < n_batches; ++batch) {
        advance_steps(config, initial.potential, current, next, config.dump_every);
        write_frame(current);
    }
}

}  // namespace physics
