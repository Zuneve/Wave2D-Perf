#include "physics/solvers.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "physics/field.hpp"

namespace physics {
namespace {

#if defined(__clang__) || defined(__GNUC__)
constexpr bool kHasVec4 = true;
using Vec4f = float __attribute__((vector_size(16)));

inline Vec4f load_vec4(const float* ptr) {
    Vec4f value;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
}

inline void store_vec4(float* ptr, Vec4f value) {
    std::memcpy(ptr, &value, sizeof(value));
}

inline Vec4f splat(float value) {
    return Vec4f{value, value, value, value};
}

inline void divide_by_one_plus_ibeta(Vec4f ar, Vec4f ai, Vec4f beta, Vec4f& out_r, Vec4f& out_i) {
    const Vec4f one = splat(1.0f);
    const Vec4f denom = one + beta * beta;
    out_r = (ar + ai * beta) / denom;
    out_i = (ai - ar * beta) / denom;
}

inline void divide_by_general(Vec4f ar, Vec4f ai, Vec4f br, Vec4f bi, Vec4f& out_r, Vec4f& out_i) {
    const Vec4f denom = br * br + bi * bi;
    out_r = (ar * br + ai * bi) / denom;
    out_i = (ai * br - ar * bi) / denom;
}

inline void mul_i_alpha(Vec4f alpha, Vec4f br, Vec4f bi, Vec4f& out_r, Vec4f& out_i) {
    out_r = -alpha * bi;
    out_i = alpha * br;
}

inline void mul_general(Vec4f ar, Vec4f ai, Vec4f br, Vec4f bi, Vec4f& out_r, Vec4f& out_i) {
    out_r = ar * br - ai * bi;
    out_i = ar * bi + ai * br;
}
#else
constexpr bool kHasVec4 = false;
#endif

struct CNWorkspace {
    std::vector<float> beta_x;
    std::vector<float> beta_y;
    std::vector<std::vector<float>> c_real;
    std::vector<std::vector<float>> c_imag;
    std::vector<std::vector<float>> d_real;
    std::vector<std::vector<float>> d_imag;
    std::vector<std::vector<float>> c_real4;
    std::vector<std::vector<float>> c_imag4;
    std::vector<std::vector<float>> d_real4;
    std::vector<std::vector<float>> d_imag4;
    std::size_t worker_count = 1;
};

inline void divide_by_one_plus_ibeta(float ar, float ai, float beta, float& out_r, float& out_i) {
    const float denom = 1.0f + beta * beta;
    out_r = (ar + ai * beta) / denom;
    out_i = (ai - ar * beta) / denom;
}

inline void divide_by_general(float ar, float ai, float br, float bi, float& out_r, float& out_i) {
    const float denom = br * br + bi * bi;
    out_r = (ar * br + ai * bi) / denom;
    out_i = (ai * br - ar * bi) / denom;
}

inline void mul_i_alpha(float alpha, float br, float bi, float& out_r, float& out_i) {
    out_r = -alpha * bi;
    out_i = alpha * br;
}

inline void mul_general(float ar, float ai, float br, float bi, float& out_r, float& out_i) {
    out_r = ar * br - ai * bi;
    out_i = ar * bi + ai * br;
}

inline void set_zero_boundaries(SoAField& field) {
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

void apply_x_cayley(
    const SimulationConfig& config,
    const SoAField& input,
    SoAField& output,
    const std::vector<float>& beta_x,
    float off_i,
    CNWorkspace& ws,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t worker_id) {

    const std::size_t n = config.nx - 2;
    if (n == 0) {
        return;
    }

    auto* const c_real = ws.c_real[worker_id].data();
    auto* const c_imag = ws.c_imag[worker_id].data();
    auto* const d_real = ws.d_real[worker_id].data();
    auto* const d_imag = ws.d_imag[worker_id].data();

    for (std::size_t y = row_begin; y < row_end; ++y) {
        const std::size_t row = y * input.pitch;

        output.real[row] = 0.0f;
        output.imag[row] = 0.0f;
        output.real[row + config.nx - 1] = 0.0f;
        output.imag[row + config.nx - 1] = 0.0f;

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t x = i + 1;
            const std::size_t idx = row + x;
            const float beta = beta_x[idx];

            float rhs_r = input.real[idx] + beta * input.imag[idx];
            float rhs_i = input.imag[idx] - beta * input.real[idx];

            if (i > 0) {
                rhs_r += off_i * input.imag[idx - 1];
                rhs_i -= off_i * input.real[idx - 1];
            }
            if (i + 1 < n) {
                rhs_r += off_i * input.imag[idx + 1];
                rhs_i -= off_i * input.real[idx + 1];
            }

            if (i == 0) {
                if (n > 1) {
                    divide_by_one_plus_ibeta(0.0f, off_i, beta, c_real[0], c_imag[0]);
                } else {
                    c_real[0] = 0.0f;
                    c_imag[0] = 0.0f;
                }
                divide_by_one_plus_ibeta(rhs_r, rhs_i, beta, d_real[0], d_imag[0]);
            } else {
                float prod_r;
                float prod_i;
                mul_i_alpha(off_i, c_real[i - 1], c_imag[i - 1], prod_r, prod_i);
                const float denom_r = 1.0f - prod_r;
                const float denom_i = beta - prod_i;

                if (i + 1 < n) {
                    divide_by_general(0.0f, off_i, denom_r, denom_i, c_real[i], c_imag[i]);
                } else {
                    c_real[i] = 0.0f;
                    c_imag[i] = 0.0f;
                }

                mul_i_alpha(off_i, d_real[i - 1], d_imag[i - 1], prod_r, prod_i);
                divide_by_general(rhs_r - prod_r, rhs_i - prod_i, denom_r, denom_i, d_real[i], d_imag[i]);
            }
        }

        output.real[row + n] = d_real[n - 1];
        output.imag[row + n] = d_imag[n - 1];
        for (std::size_t i = n - 1; i >= 1; --i) {
            float prod_r;
            float prod_i;
            mul_general(
                c_real[i - 1], c_imag[i - 1],
                output.real[row + i + 1], output.imag[row + i + 1],
                prod_r, prod_i);
            output.real[row + i] = d_real[i - 1] - prod_r;
            output.imag[row + i] = d_imag[i - 1] - prod_i;
        }
    }
}

void apply_y_cayley(
    const SimulationConfig& config,
    const SoAField& input,
    SoAField& output,
    const std::vector<float>& beta_y,
    float off_i,
    CNWorkspace& ws,
    std::size_t col_begin,
    std::size_t col_end,
    std::size_t worker_id) {

    const std::size_t n = config.ny - 2;
    if (n == 0) {
        return;
    }

#if defined(__clang__) || defined(__GNUC__)
    if constexpr (kHasVec4) {
        auto* const c_real4 = ws.c_real4[worker_id].data();
        auto* const c_imag4 = ws.c_imag4[worker_id].data();
        auto* const d_real4 = ws.d_real4[worker_id].data();
        auto* const d_imag4 = ws.d_imag4[worker_id].data();
        const Vec4f off_v = splat(off_i);
        const Vec4f one_v = splat(1.0f);

        std::size_t x = col_begin;
        for (; x + 3 < col_end; x += 4) {
            for (std::size_t i = 0; i < n; ++i) {
                const std::size_t y = i + 1;
                const std::size_t idx = y * input.pitch + x;
                const Vec4f beta = load_vec4(beta_y.data() + idx);
                const Vec4f in_r = load_vec4(input.real.data() + idx);
                const Vec4f in_i = load_vec4(input.imag.data() + idx);

                Vec4f rhs_r = in_r + beta * in_i;
                Vec4f rhs_i = in_i - beta * in_r;

                if (i > 0) {
                    const Vec4f prev_r = load_vec4(input.real.data() + idx - input.pitch);
                    const Vec4f prev_i = load_vec4(input.imag.data() + idx - input.pitch);
                    rhs_r += off_v * prev_i;
                    rhs_i -= off_v * prev_r;
                }
                if (i + 1 < n) {
                    const Vec4f next_r = load_vec4(input.real.data() + idx + input.pitch);
                    const Vec4f next_i = load_vec4(input.imag.data() + idx + input.pitch);
                    rhs_r += off_v * next_i;
                    rhs_i -= off_v * next_r;
                }

                if (i == 0) {
                    if (n > 1) {
                        Vec4f out_r;
                        Vec4f out_i;
                        divide_by_one_plus_ibeta(splat(0.0f), off_v, beta, out_r, out_i);
                        store_vec4(c_real4 + 4 * i, out_r);
                        store_vec4(c_imag4 + 4 * i, out_i);
                    } else {
                        store_vec4(c_real4 + 4 * i, splat(0.0f));
                        store_vec4(c_imag4 + 4 * i, splat(0.0f));
                    }

                    Vec4f out_r;
                    Vec4f out_i;
                    divide_by_one_plus_ibeta(rhs_r, rhs_i, beta, out_r, out_i);
                    store_vec4(d_real4 + 4 * i, out_r);
                    store_vec4(d_imag4 + 4 * i, out_i);
                } else {
                    const Vec4f prev_c_r = load_vec4(c_real4 + 4 * (i - 1));
                    const Vec4f prev_c_i = load_vec4(c_imag4 + 4 * (i - 1));
                    Vec4f prod_r;
                    Vec4f prod_i;
                    mul_i_alpha(off_v, prev_c_r, prev_c_i, prod_r, prod_i);
                    const Vec4f denom_r = one_v - prod_r;
                    const Vec4f denom_i = beta - prod_i;

                    if (i + 1 < n) {
                        Vec4f out_r;
                        Vec4f out_i;
                        divide_by_general(splat(0.0f), off_v, denom_r, denom_i, out_r, out_i);
                        store_vec4(c_real4 + 4 * i, out_r);
                        store_vec4(c_imag4 + 4 * i, out_i);
                    } else {
                        store_vec4(c_real4 + 4 * i, splat(0.0f));
                        store_vec4(c_imag4 + 4 * i, splat(0.0f));
                    }

                    const Vec4f prev_d_r = load_vec4(d_real4 + 4 * (i - 1));
                    const Vec4f prev_d_i = load_vec4(d_imag4 + 4 * (i - 1));
                    mul_i_alpha(off_v, prev_d_r, prev_d_i, prod_r, prod_i);

                    Vec4f out_r;
                    Vec4f out_i;
                    divide_by_general(rhs_r - prod_r, rhs_i - prod_i, denom_r, denom_i, out_r, out_i);
                    store_vec4(d_real4 + 4 * i, out_r);
                    store_vec4(d_imag4 + 4 * i, out_i);
                }
            }

            store_vec4(output.real.data() + (config.ny - 2) * output.pitch + x, load_vec4(d_real4 + 4 * (n - 1)));
            store_vec4(output.imag.data() + (config.ny - 2) * output.pitch + x, load_vec4(d_imag4 + 4 * (n - 1)));
            for (std::size_t i = n - 1; i >= 1; --i) {
                Vec4f prod_r;
                Vec4f prod_i;
                mul_general(
                    load_vec4(c_real4 + 4 * (i - 1)),
                    load_vec4(c_imag4 + 4 * (i - 1)),
                    load_vec4(output.real.data() + (i + 1) * output.pitch + x),
                    load_vec4(output.imag.data() + (i + 1) * output.pitch + x),
                    prod_r, prod_i);
                store_vec4(output.real.data() + i * output.pitch + x, load_vec4(d_real4 + 4 * (i - 1)) - prod_r);
                store_vec4(output.imag.data() + i * output.pitch + x, load_vec4(d_imag4 + 4 * (i - 1)) - prod_i);
            }
        }

        col_begin = x;
    }
#endif

    auto* const c_real = ws.c_real[worker_id].data();
    auto* const c_imag = ws.c_imag[worker_id].data();
    auto* const d_real = ws.d_real[worker_id].data();
    auto* const d_imag = ws.d_imag[worker_id].data();

    for (std::size_t x = col_begin; x < col_end; ++x) {

        for (std::size_t i = 0; i < n; ++i) {
            const std::size_t y = i + 1;
            const std::size_t idx = y * input.pitch + x;
            const float beta = beta_y[idx];

            float rhs_r = input.real[idx] + beta * input.imag[idx];
            float rhs_i = input.imag[idx] - beta * input.real[idx];

            if (i > 0) {
                rhs_r += off_i * input.imag[idx - input.pitch];
                rhs_i -= off_i * input.real[idx - input.pitch];
            }
            if (i + 1 < n) {
                rhs_r += off_i * input.imag[idx + input.pitch];
                rhs_i -= off_i * input.real[idx + input.pitch];
            }

            if (i == 0) {
                if (n > 1) {
                    divide_by_one_plus_ibeta(0.0f, off_i, beta, c_real[0], c_imag[0]);
                } else {
                    c_real[0] = 0.0f;
                    c_imag[0] = 0.0f;
                }
                divide_by_one_plus_ibeta(rhs_r, rhs_i, beta, d_real[0], d_imag[0]);
            } else {
                float prod_r;
                float prod_i;
                mul_i_alpha(off_i, c_real[i - 1], c_imag[i - 1], prod_r, prod_i);
                const float denom_r = 1.0f - prod_r;
                const float denom_i = beta - prod_i;

                if (i + 1 < n) {
                    divide_by_general(0.0f, off_i, denom_r, denom_i, c_real[i], c_imag[i]);
                } else {
                    c_real[i] = 0.0f;
                    c_imag[i] = 0.0f;
                }

                mul_i_alpha(off_i, d_real[i - 1], d_imag[i - 1], prod_r, prod_i);
                divide_by_general(rhs_r - prod_r, rhs_i - prod_i, denom_r, denom_i, d_real[i], d_imag[i]);
            }
        }

        output.real[(config.ny - 2) * output.pitch + x] = d_real[n - 1];
        output.imag[(config.ny - 2) * output.pitch + x] = d_imag[n - 1];
        for (std::size_t i = n - 1; i >= 1; --i) {
            float prod_r;
            float prod_i;
            mul_general(
                c_real[i - 1], c_imag[i - 1],
                output.real[(i + 1) * output.pitch + x], output.imag[(i + 1) * output.pitch + x],
                prod_r, prod_i);
            output.real[i * output.pitch + x] = d_real[i - 1] - prod_r;
            output.imag[i * output.pitch + x] = d_imag[i - 1] - prod_i;
        }
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
    workers.reserve(worker_count > 0 ? worker_count - 1 : 0);

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

    const std::size_t main_end = std::min(end, begin + chunk);
    func(begin, main_end, 0);

    for (auto& worker : workers) {
        worker.join();
    }
}

CNWorkspace make_workspace(const SimulationConfig& config, const SoAField& field) {
    CNWorkspace ws;
    const std::size_t plane = field.pitch * field.ny;
    const std::size_t line = std::max(config.nx, config.ny);
    std::size_t worker_count = 1;
    if (config.threads > 0) {
        worker_count = static_cast<std::size_t>(config.threads);
    } else {
        const unsigned int hw = std::thread::hardware_concurrency();
        worker_count = hw == 0 ? 1 : static_cast<std::size_t>(hw);
    }
    worker_count = std::max<std::size_t>(1, worker_count);
    ws.beta_x.resize(plane, 0.0f);
    ws.beta_y.resize(plane, 0.0f);
    ws.c_real.assign(worker_count, std::vector<float>(line, 0.0f));
    ws.c_imag.assign(worker_count, std::vector<float>(line, 0.0f));
    ws.d_real.assign(worker_count, std::vector<float>(line, 0.0f));
    ws.d_imag.assign(worker_count, std::vector<float>(line, 0.0f));
    ws.c_real4.assign(worker_count, std::vector<float>(line * 4, 0.0f));
    ws.c_imag4.assign(worker_count, std::vector<float>(line * 4, 0.0f));
    ws.d_real4.assign(worker_count, std::vector<float>(line * 4, 0.0f));
    ws.d_imag4.assign(worker_count, std::vector<float>(line * 4, 0.0f));
    ws.worker_count = worker_count;

    const float kinetic_x = 1.0f / (config.mass * config.dx * config.dx);
    const float kinetic_y = 1.0f / (config.mass * config.dy * config.dy);
    const float half_dt = 0.5f * config.dt;
    const float quarter_dt = 0.25f * config.dt;

    for (std::size_t y = 0; y < field.ny; ++y) {
        const std::size_t row = y * field.pitch;
        for (std::size_t x = 0; x < field.nx; ++x) {
            const std::size_t idx = row + x;
            const float potential_half = 0.5f * field.potential[idx];
            ws.beta_x[idx] = quarter_dt * (kinetic_x + potential_half);
            ws.beta_y[idx] = half_dt * (kinetic_y + potential_half);
        }
    }

    return ws;
}

void advance_steps(
    const SimulationConfig& config,
    SoAField& current,
    SoAField& next,
    SoAField& scratch,
    CNWorkspace& ws,
    std::size_t steps) {
    const float off_x = -0.25f * config.dt / (config.mass * config.dx * config.dx);
    const float off_y = -0.25f * config.dt / (config.mass * config.dy * config.dy);

    for (std::size_t step = 0; step < steps; ++step) {
        set_zero_boundaries(scratch);
        parallel_chunks(1, config.ny - 1, ws.worker_count, [&](std::size_t begin, std::size_t end, std::size_t worker_id) {
            apply_x_cayley(config, current, scratch, ws.beta_x, off_x, ws, begin, end, worker_id);
        });

        set_zero_boundaries(next);
        parallel_chunks(1, config.nx - 1, ws.worker_count, [&](std::size_t begin, std::size_t end, std::size_t worker_id) {
            apply_y_cayley(config, scratch, next, ws.beta_y, off_y, ws, begin, end, worker_id);
        });

        set_zero_boundaries(scratch);
        parallel_chunks(1, config.ny - 1, ws.worker_count, [&](std::size_t begin, std::size_t end, std::size_t worker_id) {
            apply_x_cayley(config, next, scratch, ws.beta_x, off_x, ws, begin, end, worker_id);
        });

        current.real.swap(scratch.real);
        current.imag.swap(scratch.imag);
    }
}

void write_dump_header(
    const SimulationConfig& config,
    std::size_t n_frames,
    std::ofstream& file) {
    const uint64_t nx_u = config.nx;
    const uint64_t ny_u = config.ny;
    const uint64_t n_frames_u = n_frames;
    const uint64_t dump_every_u = config.dump_every;
    const float pad = 0.0f;

    const char magic[8] = {'W','A','V','E','2','D','\0','\0'};
    file.write(magic, 8);
    file.write(reinterpret_cast<const char*>(&nx_u), 8);
    file.write(reinterpret_cast<const char*>(&ny_u), 8);
    file.write(reinterpret_cast<const char*>(&n_frames_u), 8);
    file.write(reinterpret_cast<const char*>(&dump_every_u), 8);
    file.write(reinterpret_cast<const char*>(&config.dx), 4);
    file.write(reinterpret_cast<const char*>(&config.dy), 4);
    file.write(reinterpret_cast<const char*>(&config.dt), 4);
    file.write(reinterpret_cast<const char*>(&pad), 4);
}

void write_frame(std::ofstream& file, const SoAField& field) {
    for (std::size_t y = 0; y < field.ny; ++y) {
        const std::size_t row = y * field.pitch;
        for (std::size_t x = 0; x < field.nx; ++x) {
            const float real = field.real[row + x];
            const float imag = field.imag[row + x];
            const float prob = real * real + imag * imag;
            file.write(reinterpret_cast<const char*>(&prob), sizeof(float));
        }
    }
}

}  // namespace

BenchmarkResult run_cn_adi(const SimulationConfig& config, const InitialState& initial) {
    SoAField warm_current = make_soa_field(initial.nx, initial.ny);
    SoAField warm_next = make_soa_field(initial.nx, initial.ny);
    SoAField warm_scratch = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, warm_current);
    warm_next.potential = warm_current.potential;
    warm_scratch.potential = warm_current.potential;
    CNWorkspace warm_ws = make_workspace(config, warm_current);
    advance_steps(config, warm_current, warm_next, warm_scratch, warm_ws, config.warmup_steps);

    SoAField current = make_soa_field(initial.nx, initial.ny);
    SoAField next = make_soa_field(initial.nx, initial.ny);
    SoAField scratch = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, current);
    next.potential = current.potential;
    scratch.potential = current.potential;
    CNWorkspace ws = make_workspace(config, current);

    const auto t0 = std::chrono::steady_clock::now();
    advance_steps(config, current, next, scratch, ws, config.steps);
    const auto t1 = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double updated_cells =
        static_cast<double>((config.nx - 2) * (config.ny - 2)) * static_cast<double>(config.steps);

    return {
        .name = ws.worker_count > 1 ? "cn_adi_soa_threads" : "cn_adi_soa",
        .nx = config.nx,
        .ny = config.ny,
        .steps = config.steps,
        .seconds = seconds,
        .mlups = updated_cells / seconds / 1.0e6,
        .l2_norm = l2_norm(current),
        .max_amplitude = max_amplitude(current),
    };
}

void run_cn_adi_dump(const SimulationConfig& config, const InitialState& initial,
                     const std::string& output_path) {
    if (config.dump_every == 0) {
        throw std::invalid_argument("dump_every must be > 0");
    }

    std::ofstream file(output_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open output file: " + output_path);
    }

    const std::size_t n_frames = config.steps / config.dump_every + 1;
    write_dump_header(config, n_frames, file);

    SoAField current = make_soa_field(initial.nx, initial.ny);
    SoAField next = make_soa_field(initial.nx, initial.ny);
    SoAField scratch = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, current);
    next.potential = current.potential;
    scratch.potential = current.potential;
    CNWorkspace ws = make_workspace(config, current);

    write_frame(file, current);

    const std::size_t n_batches = config.steps / config.dump_every;
    for (std::size_t batch = 0; batch < n_batches; ++batch) {
        advance_steps(config, current, next, scratch, ws, config.dump_every);
        write_frame(file, current);
    }
}

}  // namespace physics
