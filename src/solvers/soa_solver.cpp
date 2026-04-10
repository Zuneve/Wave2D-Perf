#include "physics/solvers.hpp"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "physics/simd.hpp"

#if defined(PHYSICS_HAS_OPENMP) && PHYSICS_HAS_OPENMP
#include <omp.h>
#endif

namespace physics {
namespace {

enum class ExecutionMode {
    serial,
    threads,
    openmp
};

[[maybe_unused]] void advance_row_scalar(
    const SimulationConfig& config,
    const SoAField& current,
    SoAField& next,
    std::size_t y,
    float alpha) {
    const std::size_t row = y * current.pitch;
    const std::size_t north = row - current.pitch;
    const std::size_t south = row + current.pitch;

    next.real[row] = 0.0f;
    next.imag[row] = 0.0f;
    next.real[row + current.nx - 1] = 0.0f;
    next.imag[row + current.nx - 1] = 0.0f;

    for (std::size_t block_x = 1; block_x < current.nx - 1; block_x += config.tile_x) {
        const std::size_t block_end = std::min(current.nx - 1, block_x + config.tile_x);
        for (std::size_t x = block_x; x < block_end; ++x) {
            const std::size_t idx = row + x;
            const float center_real = current.real[idx];
            const float center_imag = current.imag[idx];
            const float lap_real =
                current.real[idx - 1] + current.real[idx + 1] + current.real[north + x] + current.real[south + x] -
                4.0f * center_real;
            const float lap_imag =
                current.imag[idx - 1] + current.imag[idx + 1] + current.imag[north + x] + current.imag[south + x] -
                4.0f * center_imag;
            const float local_potential = current.potential[idx];

            next.real[idx] = center_real + config.dt * (local_potential * center_imag - alpha * lap_imag);
            next.imag[idx] = center_imag + config.dt * (alpha * lap_real - local_potential * center_real);
        }
    }
}

void advance_row_vectorized(
    const SimulationConfig& config,
    const SoAField& current,
    SoAField& next,
    std::size_t y,
    float alpha) {
    const std::size_t row = y * current.pitch;
    const std::size_t north = row - current.pitch;
    const std::size_t south = row + current.pitch;

    next.real[row] = 0.0f;
    next.imag[row] = 0.0f;
    next.real[row + current.nx - 1] = 0.0f;
    next.imag[row + current.nx - 1] = 0.0f;

#if defined(__ARM_NEON) || defined(__SSE__) || defined(__clang__) || defined(__GNUC__)
    if constexpr (simd::available) {
        const simd::Vec4f dt = simd::splat(config.dt);
        const simd::Vec4f alpha_v = simd::splat(alpha);
        const simd::Vec4f minus_four = simd::splat(-4.0f);

        for (std::size_t block_x = 1; block_x < current.nx - 1; block_x += config.tile_x) {
            const std::size_t block_end = std::min(current.nx - 1, block_x + config.tile_x);
            std::size_t x = block_x;

            while (x + simd::width - 1 < block_end) {
                const std::size_t idx = row + x;
                const simd::Vec4f center_real = simd::load_u(current.real.data() + idx);
                const simd::Vec4f center_imag = simd::load_u(current.imag.data() + idx);
                const simd::Vec4f lap_real =
                    simd::load_u(current.real.data() + idx - 1) +
                    simd::load_u(current.real.data() + idx + 1) +
                    simd::load_u(current.real.data() + north + x) +
                    simd::load_u(current.real.data() + south + x) +
                    minus_four * center_real;
                const simd::Vec4f lap_imag =
                    simd::load_u(current.imag.data() + idx - 1) +
                    simd::load_u(current.imag.data() + idx + 1) +
                    simd::load_u(current.imag.data() + north + x) +
                    simd::load_u(current.imag.data() + south + x) +
                    minus_four * center_imag;
                const simd::Vec4f potential = simd::load_u(current.potential.data() + idx);

                const simd::Vec4f next_real = center_real + dt * (potential * center_imag - alpha_v * lap_imag);
                const simd::Vec4f next_imag = center_imag + dt * (alpha_v * lap_real - potential * center_real);

                simd::store_u(next.real.data() + idx, next_real);
                simd::store_u(next.imag.data() + idx, next_imag);
                x += simd::width;
            }

            for (; x < block_end; ++x) {
                const std::size_t idx = row + x;
                const float center_real = current.real[idx];
                const float center_imag = current.imag[idx];
                const float lap_real =
                    current.real[idx - 1] + current.real[idx + 1] + current.real[north + x] + current.real[south + x] -
                    4.0f * center_real;
                const float lap_imag =
                    current.imag[idx - 1] + current.imag[idx + 1] + current.imag[north + x] + current.imag[south + x] -
                    4.0f * center_imag;
                const float local_potential = current.potential[idx];

                next.real[idx] = center_real + config.dt * (local_potential * center_imag - alpha * lap_imag);
                next.imag[idx] = center_imag + config.dt * (alpha * lap_real - local_potential * center_real);
            }
        }
        return;
    }
#endif

    advance_row_scalar(config, current, next, y, alpha);
}

void advance_rows_range(
    const SimulationConfig& config,
    const SoAField& current,
    SoAField& next,
    std::size_t row_begin,
    std::size_t row_end,
    float alpha) {
    for (std::size_t block_start = row_begin; block_start < row_end; block_start += config.tile_y) {
        const std::size_t block_end = std::min(row_end, block_start + config.tile_y);
        for (std::size_t y = block_start; y < block_end; ++y) {
            advance_row_vectorized(config, current, next, y, alpha);
        }
    }
}

void advance_steps_serial(
    const SimulationConfig& config,
    SoAField& current,
    SoAField& next) {
    const float alpha = 0.5f / config.mass;

    for (std::size_t step = 0; step < config.steps; ++step) {
        advance_rows_range(config, current, next, 1, current.ny - 1, alpha);
        current.real.swap(next.real);
        current.imag.swap(next.imag);
    }
}

void advance_steps_threads(
    const SimulationConfig& config,
    SoAField& current,
    SoAField& next) {
    const std::size_t interior_rows = current.ny - 2;
    const std::size_t requested_threads = config.threads > 0
        ? static_cast<std::size_t>(config.threads)
        : std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min(interior_rows, requested_threads);

    if (worker_count <= 1) {
        advance_steps_serial(config, current, next);
        return;
    }

    const float alpha = 0.5f / config.mass;
    const std::size_t rows_per_worker = (interior_rows + worker_count - 1) / worker_count;
    std::barrier step_barrier(
        static_cast<std::ptrdiff_t>(worker_count),
        [&]() {
            current.real.swap(next.real);
            current.imag.swap(next.imag);
        });
    std::vector<std::jthread> workers;
    workers.reserve(worker_count);

    for (std::size_t worker_id = 0; worker_id < worker_count; ++worker_id) {
        const std::size_t row_begin = 1 + worker_id * rows_per_worker;
        const std::size_t row_end = std::min(current.ny - 1, row_begin + rows_per_worker);

        workers.emplace_back([&, row_begin, row_end]() {
            for (std::size_t step = 0; step < config.steps; ++step) {
                advance_rows_range(config, current, next, row_begin, row_end, alpha);
                step_barrier.arrive_and_wait();
            }
        });
    }
}

#if defined(PHYSICS_HAS_OPENMP) && PHYSICS_HAS_OPENMP
void advance_steps_openmp(
    const SimulationConfig& config,
    SoAField& current,
    SoAField& next) {
    const float alpha = 0.5f / config.mass;

    if (config.threads > 0) {
        omp_set_num_threads(config.threads);
    }

    for (std::size_t step = 0; step < config.steps; ++step) {
#pragma omp parallel for schedule(static, 1)
        for (std::ptrdiff_t block_start = 1; block_start < static_cast<std::ptrdiff_t>(current.ny - 1);
             block_start += static_cast<std::ptrdiff_t>(config.tile_y)) {
            const auto row_begin = static_cast<std::size_t>(block_start);
            const std::size_t row_end = std::min(current.ny - 1, row_begin + config.tile_y);
            advance_rows_range(config, current, next, row_begin, row_end, alpha);
        }

        current.real.swap(next.real);
        current.imag.swap(next.imag);
    }
}
#endif

void advance_steps(
    const SimulationConfig& config,
    SoAField& current,
    SoAField& next,
    ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::serial:
            advance_steps_serial(config, current, next);
            break;
        case ExecutionMode::threads:
            advance_steps_threads(config, current, next);
            break;
        case ExecutionMode::openmp:
#if defined(PHYSICS_HAS_OPENMP) && PHYSICS_HAS_OPENMP
            advance_steps_openmp(config, current, next);
#else
            advance_steps_serial(config, current, next);
#endif
            break;
    }
}

std::string make_backend_name(ExecutionMode mode) {
    std::string name = "soa_";
    name += simd::backend_name;

    if (mode == ExecutionMode::threads) {
        name += "_threads";
    } else if (mode == ExecutionMode::openmp) {
        name += "_omp";
    }

    return name;
}

BenchmarkResult run_soa_impl(
    const SimulationConfig& config,
    const InitialState& initial,
    ExecutionMode mode) {
    SoAField warm_current = make_soa_field(initial.nx, initial.ny);
    SoAField warm_next = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, warm_current);
    warm_next.potential = warm_current.potential;

    if (config.warmup_steps > 0) {
        SimulationConfig warmup_config = config;
        warmup_config.steps = config.warmup_steps;
        advance_steps(warmup_config, warm_current, warm_next, mode);
    }

    SoAField current = make_soa_field(initial.nx, initial.ny);
    SoAField next = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, current);
    next.potential = current.potential;

    const auto start = std::chrono::steady_clock::now();
    advance_steps(config, current, next, mode);
    const auto end = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(end - start).count();
    const double updated_cells = static_cast<double>((config.nx - 2) * (config.ny - 2) * config.steps);

    return {
        .name = make_backend_name(mode),
        .seconds = seconds,
        .mlups = updated_cells / seconds / 1.0e6,
        .l2_norm = l2_norm(current),
        .max_amplitude = max_amplitude(current),
    };
}

}  // namespace

BenchmarkResult run_soa_simd(const SimulationConfig& config, const InitialState& initial, bool use_openmp) {
    return run_soa_impl(config, initial, use_openmp ? ExecutionMode::openmp : ExecutionMode::serial);
}

BenchmarkResult run_soa_threads(const SimulationConfig& config, const InitialState& initial) {
    return run_soa_impl(config, initial, ExecutionMode::threads);
}

}  // namespace physics
