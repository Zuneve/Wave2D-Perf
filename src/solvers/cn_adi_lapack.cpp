#include "physics/solvers.hpp"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <thread>
#include <vector>

#include "physics/field.hpp"

// "Standard library" opponent for the duel: the CN-ADI algorithm is kept
// identical to the hand-written cn-adi solver, but every tridiagonal solve is
// delegated to LAPACK's general complex tridiagonal routine cgtsv. On macOS
// this comes from the Accelerate framework; the legacy Fortran symbol cgtsv_
// is available without ACCELERATE_NEW_LAPACK.
//
// Falls back to the hand-written CPU solver when the build was configured
// without LAPACK, mirroring the CUDA fallback path.

namespace physics {

#if defined(WAVE2D_ENABLE_LAPACK)

namespace {

using Complex = std::complex<float>;

extern "C" void cgtsv_(const int* n, const int* nrhs,
                       Complex* dl, Complex* d, Complex* du,
                       Complex* b, const int* ldb, int* info);

// Per-worker scratch for one tridiagonal solve.
struct LapackScratch {
    std::vector<Complex> dl;  // sub-diagonal,   length n-1
    std::vector<Complex> d;   // main diagonal,  length n
    std::vector<Complex> du;  // super-diagonal, length n-1
    std::vector<Complex> b;   // RHS / solution, length n
};

void set_zero_boundaries(SoAField& field) {
    for (std::size_t x = 0; x < field.nx; ++x) {
        field.real[x] = 0.0f;
        field.imag[x] = 0.0f;
        const std::size_t bottom = (field.ny - 1) * field.pitch + x;
        field.real[bottom] = 0.0f;
        field.imag[bottom] = 0.0f;
    }
    for (std::size_t y = 1; y + 1 < field.ny; ++y) {
        const std::size_t row = y * field.pitch;
        field.real[row] = 0.0f;
        field.imag[row] = 0.0f;
        field.real[row + field.nx - 1] = 0.0f;
        field.imag[row + field.nx - 1] = 0.0f;
    }
}

template <typename Func>
void parallel_chunks(std::size_t begin, std::size_t end, std::size_t worker_count, Func&& func) {
    if (end <= begin) {
        return;
    }
    if (worker_count <= 1 || end - begin <= 1) {
        func(begin, end, 0);
        return;
    }
    const std::size_t total = end - begin;
    const std::size_t chunk = (total + worker_count - 1) / worker_count;
    std::vector<std::thread> workers;
    workers.reserve(worker_count - 1);
    for (std::size_t worker_id = 1; worker_id < worker_count; ++worker_id) {
        const std::size_t chunk_begin = begin + worker_id * chunk;
        if (chunk_begin >= end) {
            break;
        }
        const std::size_t chunk_end = std::min(end, chunk_begin + chunk);
        workers.emplace_back([&, chunk_begin, chunk_end, worker_id]() {
            func(chunk_begin, chunk_end, worker_id);
        });
    }
    func(begin, std::min(end, begin + chunk), 0);
    for (auto& worker : workers) {
        worker.join();
    }
}

std::size_t resolve_worker_count(const SimulationConfig& config) {
    if (config.threads > 0) {
        return static_cast<std::size_t>(config.threads);
    }
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<std::size_t>(hw);
}

// One implicit half-step along x: solve (1 + iβ_x) tridiagonal systems row by
// row with cgtsv. RHS uses the explicit operator (1 - iβ_x) - i*off_x*neighbours.
void apply_x_lapack(const SimulationConfig& config, const SoAField& input, SoAField& output,
                    const std::vector<float>& beta_x, float off,
                    std::vector<LapackScratch>& scratch,
                    std::size_t row_begin, std::size_t row_end, std::size_t worker_id) {
    const int n = static_cast<int>(config.nx) - 2;
    if (n <= 0) {
        return;
    }
    const Complex off_diag{0.0f, off};       // implicit super/sub-diagonal: i*off
    const Complex rhs_off{0.0f, -off};        // explicit off-diagonal:      -i*off
    LapackScratch& s = scratch[worker_id];

    for (std::size_t y = row_begin; y < row_end; ++y) {
        const std::size_t row = y * input.pitch;
        output.real[row] = 0.0f;
        output.imag[row] = 0.0f;
        output.real[row + config.nx - 1] = 0.0f;
        output.imag[row + config.nx - 1] = 0.0f;

        for (int k = 0; k < n; ++k) {
            const std::size_t idx = row + static_cast<std::size_t>(k) + 1;
            const float beta = beta_x[idx];
            const Complex psi{input.real[idx], input.imag[idx]};
            const Complex left  = (k > 0)     ? Complex{input.real[idx - 1], input.imag[idx - 1]} : Complex{0, 0};
            const Complex right = (k + 1 < n) ? Complex{input.real[idx + 1], input.imag[idx + 1]} : Complex{0, 0};

            s.d[k] = Complex{1.0f, beta};
            s.b[k] = Complex{1.0f, -beta} * psi + rhs_off * (left + right);
            if (k < n - 1) {
                s.dl[k] = off_diag;
                s.du[k] = off_diag;
            }
        }

        int nn = n, nrhs = 1, ldb = n, info = 0;
        cgtsv_(&nn, &nrhs, s.dl.data(), s.d.data(), s.du.data(), s.b.data(), &ldb, &info);

        for (int k = 0; k < n; ++k) {
            const std::size_t idx = row + static_cast<std::size_t>(k) + 1;
            output.real[idx] = s.b[k].real();
            output.imag[idx] = s.b[k].imag();
        }
    }
}

// One implicit half-step along y: same idea, marching down columns (stride = pitch).
void apply_y_lapack(const SimulationConfig& config, const SoAField& input, SoAField& output,
                    const std::vector<float>& beta_y, float off,
                    std::vector<LapackScratch>& scratch,
                    std::size_t col_begin, std::size_t col_end, std::size_t worker_id) {
    const int n = static_cast<int>(config.ny) - 2;
    if (n <= 0) {
        return;
    }
    const std::size_t pitch = input.pitch;
    const Complex off_diag{0.0f, off};
    const Complex rhs_off{0.0f, -off};
    LapackScratch& s = scratch[worker_id];

    for (std::size_t x = col_begin; x < col_end; ++x) {
        output.real[x] = 0.0f;
        output.imag[x] = 0.0f;
        output.real[(config.ny - 1) * pitch + x] = 0.0f;
        output.imag[(config.ny - 1) * pitch + x] = 0.0f;

        for (int k = 0; k < n; ++k) {
            const std::size_t idx = (static_cast<std::size_t>(k) + 1) * pitch + x;
            const float beta = beta_y[idx];
            const Complex psi{input.real[idx], input.imag[idx]};
            const Complex up   = (k > 0)     ? Complex{input.real[idx - pitch], input.imag[idx - pitch]} : Complex{0, 0};
            const Complex down = (k + 1 < n) ? Complex{input.real[idx + pitch], input.imag[idx + pitch]} : Complex{0, 0};

            s.d[k] = Complex{1.0f, beta};
            s.b[k] = Complex{1.0f, -beta} * psi + rhs_off * (up + down);
            if (k < n - 1) {
                s.dl[k] = off_diag;
                s.du[k] = off_diag;
            }
        }

        int nn = n, nrhs = 1, ldb = n, info = 0;
        cgtsv_(&nn, &nrhs, s.dl.data(), s.d.data(), s.du.data(), s.b.data(), &ldb, &info);

        for (int k = 0; k < n; ++k) {
            const std::size_t idx = (static_cast<std::size_t>(k) + 1) * pitch + x;
            output.real[idx] = s.b[k].real();
            output.imag[idx] = s.b[k].imag();
        }
    }
}

void compute_beta(const SimulationConfig& config, const SoAField& field,
                  std::vector<float>& beta_x, std::vector<float>& beta_y) {
    const std::size_t plane = field.pitch * field.ny;
    beta_x.assign(plane, 0.0f);
    beta_y.assign(plane, 0.0f);
    const float kinetic_x = 1.0f / (config.mass * config.dx * config.dx);
    const float kinetic_y = 1.0f / (config.mass * config.dy * config.dy);
    const float half_dt = 0.5f * config.dt;
    const float quarter_dt = 0.25f * config.dt;
    for (std::size_t y = 0; y < field.ny; ++y) {
        const std::size_t row = y * field.pitch;
        for (std::size_t x = 0; x < field.nx; ++x) {
            const std::size_t idx = row + x;
            const float potential_half = 0.5f * field.potential[idx];
            beta_x[idx] = quarter_dt * (kinetic_x + potential_half);
            beta_y[idx] = half_dt * (kinetic_y + potential_half);
        }
    }
}

void advance_steps(const SimulationConfig& config, SoAField& current, SoAField& next, SoAField& scratch,
                   const std::vector<float>& beta_x, const std::vector<float>& beta_y,
                   std::vector<LapackScratch>& work, std::size_t worker_count, std::size_t steps) {
    const float off_x = -0.25f * config.dt / (config.mass * config.dx * config.dx);
    const float off_y = -0.25f * config.dt / (config.mass * config.dy * config.dy);

    for (std::size_t step = 0; step < steps; ++step) {
        set_zero_boundaries(scratch);
        parallel_chunks(1, config.ny - 1, worker_count, [&](std::size_t b, std::size_t e, std::size_t w) {
            apply_x_lapack(config, current, scratch, beta_x, off_x, work, b, e, w);
        });

        set_zero_boundaries(next);
        parallel_chunks(1, config.nx - 1, worker_count, [&](std::size_t b, std::size_t e, std::size_t w) {
            apply_y_lapack(config, scratch, next, beta_y, off_y, work, b, e, w);
        });

        set_zero_boundaries(scratch);
        parallel_chunks(1, config.ny - 1, worker_count, [&](std::size_t b, std::size_t e, std::size_t w) {
            apply_x_lapack(config, next, scratch, beta_x, off_x, work, b, e, w);
        });

        current.real.swap(scratch.real);
        current.imag.swap(scratch.imag);
    }
}

}  // namespace

BenchmarkResult run_cn_adi_lapack(const SimulationConfig& config, const InitialState& initial) {
    const std::size_t worker_count = std::max<std::size_t>(1, resolve_worker_count(config));
    const std::size_t line = std::max(config.nx, config.ny);

    auto make_work = [&]() {
        std::vector<LapackScratch> work(worker_count);
        for (auto& s : work) {
            s.dl.assign(line, Complex{0, 0});
            s.d.assign(line, Complex{0, 0});
            s.du.assign(line, Complex{0, 0});
            s.b.assign(line, Complex{0, 0});
        }
        return work;
    };

    SoAField current = make_soa_field(initial.nx, initial.ny);
    SoAField next = make_soa_field(initial.nx, initial.ny);
    SoAField scratch = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, current);
    next.potential = current.potential;
    scratch.potential = current.potential;

    std::vector<float> beta_x;
    std::vector<float> beta_y;
    compute_beta(config, current, beta_x, beta_y);

    std::vector<LapackScratch> work = make_work();
    advance_steps(config, current, next, scratch, beta_x, beta_y, work, worker_count, config.warmup_steps);

    // Reset to the pristine initial state before the timed run.
    fill_soa_from_initial(initial, current);

    const auto t0 = std::chrono::steady_clock::now();
    advance_steps(config, current, next, scratch, beta_x, beta_y, work, worker_count, config.steps);
    const auto t1 = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double updated_cells =
        static_cast<double>((config.nx - 2) * (config.ny - 2)) * static_cast<double>(config.steps);

    return {
        .name = worker_count > 1 ? "cn_adi_lapack_threads" : "cn_adi_lapack",
        .nx = config.nx,
        .ny = config.ny,
        .steps = config.steps,
        .seconds = seconds,
        .mlups = updated_cells / seconds / 1.0e6,
        .l2_norm = l2_norm(current),
        .max_amplitude = max_amplitude(current),
    };
}

}  // namespace physics

#else  // !WAVE2D_ENABLE_LAPACK

#include <iostream>

namespace physics {

BenchmarkResult run_cn_adi_lapack(const SimulationConfig& config, const InitialState& initial) {
    std::cerr << "Warning: LAPACK support is not enabled in this build. "
                 "Falling back to CPU cn-adi.\n";
    return run_cn_adi(config, initial);
}

}  // namespace physics

#endif  // WAVE2D_ENABLE_LAPACK
