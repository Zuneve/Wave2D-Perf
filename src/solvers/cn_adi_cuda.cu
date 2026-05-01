#include "physics/solvers.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "physics/field.hpp"

namespace physics {
namespace {

struct DeviceBuffers {
    float* real_current = nullptr;
    float* imag_current = nullptr;
    float* real_next = nullptr;
    float* imag_next = nullptr;
    float* real_scratch = nullptr;
    float* imag_scratch = nullptr;
    float* potential = nullptr;
    float* beta_x = nullptr;
    float* beta_y = nullptr;

    float* row_c_real = nullptr;
    float* row_c_imag = nullptr;
    float* row_d_real = nullptr;
    float* row_d_imag = nullptr;
    float* col_c_real = nullptr;
    float* col_c_imag = nullptr;
    float* col_d_real = nullptr;
    float* col_d_imag = nullptr;
};

inline void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

inline void cuda_malloc(float** ptr, std::size_t n_elems, const char* what) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(ptr), n_elems * sizeof(float)), what);
}

inline void cuda_free(float*& ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

__global__ void x_cayley_pass_kernel(
    std::size_t nx,
    std::size_t ny,
    std::size_t pitch,
    const float* in_real,
    const float* in_imag,
    float* out_real,
    float* out_imag,
    const float* beta_x,
    float off_i,
    float* c_real_rows,
    float* c_imag_rows,
    float* d_real_rows,
    float* d_imag_rows) {

    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t y = tid + 1;
    if (y + 1 >= ny) {
        return;
    }

    const std::size_t n = nx - 2;
    const std::size_t row_offset = y * n;
    float* c_real = c_real_rows + row_offset;
    float* c_imag = c_imag_rows + row_offset;
    float* d_real = d_real_rows + row_offset;
    float* d_imag = d_imag_rows + row_offset;

    const std::size_t row = y * pitch;
    out_real[row] = 0.0f;
    out_imag[row] = 0.0f;
    out_real[row + nx - 1] = 0.0f;
    out_imag[row + nx - 1] = 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t x = i + 1;
        const std::size_t idx = row + x;
        const float beta = beta_x[idx];

        float rhs_r = in_real[idx] + beta * in_imag[idx];
        float rhs_i = in_imag[idx] - beta * in_real[idx];

        if (i > 0) {
            rhs_r += off_i * in_imag[idx - 1];
            rhs_i -= off_i * in_real[idx - 1];
        }
        if (i + 1 < n) {
            rhs_r += off_i * in_imag[idx + 1];
            rhs_i -= off_i * in_real[idx + 1];
        }

        if (i == 0) {
            const float denom = 1.0f + beta * beta;
            if (n > 1) {
                c_real[0] = (off_i * beta) / denom;
                c_imag[0] = off_i / denom;
            } else {
                c_real[0] = 0.0f;
                c_imag[0] = 0.0f;
            }
            d_real[0] = (rhs_r + rhs_i * beta) / denom;
            d_imag[0] = (rhs_i - rhs_r * beta) / denom;
        } else {
            const float prod_r = -off_i * c_imag[i - 1];
            const float prod_i = off_i * c_real[i - 1];
            const float denom_r = 1.0f - prod_r;
            const float denom_i = beta - prod_i;
            const float denom = denom_r * denom_r + denom_i * denom_i;

            if (i + 1 < n) {
                c_real[i] = (off_i * denom_i) / denom;
                c_imag[i] = (off_i * denom_r) / denom;
            } else {
                c_real[i] = 0.0f;
                c_imag[i] = 0.0f;
            }

            const float pd_r = -off_i * d_imag[i - 1];
            const float pd_i = off_i * d_real[i - 1];
            const float num_r = rhs_r - pd_r;
            const float num_i = rhs_i - pd_i;

            d_real[i] = (num_r * denom_r + num_i * denom_i) / denom;
            d_imag[i] = (num_i * denom_r - num_r * denom_i) / denom;
        }
    }

    out_real[row + n] = d_real[n - 1];
    out_imag[row + n] = d_imag[n - 1];
    for (std::size_t i = n - 1; i >= 1; --i) {
        const float cr = c_real[i - 1];
        const float ci = c_imag[i - 1];
        const float xr = out_real[row + i + 1];
        const float xi = out_imag[row + i + 1];
        const float prod_r = cr * xr - ci * xi;
        const float prod_i = cr * xi + ci * xr;
        out_real[row + i] = d_real[i - 1] - prod_r;
        out_imag[row + i] = d_imag[i - 1] - prod_i;
    }
}

__global__ void y_cayley_pass_kernel(
    std::size_t nx,
    std::size_t ny,
    std::size_t pitch,
    const float* in_real,
    const float* in_imag,
    float* out_real,
    float* out_imag,
    const float* beta_y,
    float off_i,
    float* c_real_cols,
    float* c_imag_cols,
    float* d_real_cols,
    float* d_imag_cols) {

    const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t x = tid + 1;
    if (x + 1 >= nx) {
        return;
    }

    const std::size_t n = ny - 2;
    const std::size_t col_offset = x * n;
    float* c_real = c_real_cols + col_offset;
    float* c_imag = c_imag_cols + col_offset;
    float* d_real = d_real_cols + col_offset;
    float* d_imag = d_imag_cols + col_offset;

    out_real[x] = 0.0f;
    out_imag[x] = 0.0f;
    out_real[(ny - 1) * pitch + x] = 0.0f;
    out_imag[(ny - 1) * pitch + x] = 0.0f;

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t y = i + 1;
        const std::size_t idx = y * pitch + x;
        const float beta = beta_y[idx];

        float rhs_r = in_real[idx] + beta * in_imag[idx];
        float rhs_i = in_imag[idx] - beta * in_real[idx];

        if (i > 0) {
            rhs_r += off_i * in_imag[idx - pitch];
            rhs_i -= off_i * in_real[idx - pitch];
        }
        if (i + 1 < n) {
            rhs_r += off_i * in_imag[idx + pitch];
            rhs_i -= off_i * in_real[idx + pitch];
        }

        if (i == 0) {
            const float denom = 1.0f + beta * beta;
            if (n > 1) {
                c_real[0] = (off_i * beta) / denom;
                c_imag[0] = off_i / denom;
            } else {
                c_real[0] = 0.0f;
                c_imag[0] = 0.0f;
            }
            d_real[0] = (rhs_r + rhs_i * beta) / denom;
            d_imag[0] = (rhs_i - rhs_r * beta) / denom;
        } else {
            const float prod_r = -off_i * c_imag[i - 1];
            const float prod_i = off_i * c_real[i - 1];
            const float denom_r = 1.0f - prod_r;
            const float denom_i = beta - prod_i;
            const float denom = denom_r * denom_r + denom_i * denom_i;

            if (i + 1 < n) {
                c_real[i] = (off_i * denom_i) / denom;
                c_imag[i] = (off_i * denom_r) / denom;
            } else {
                c_real[i] = 0.0f;
                c_imag[i] = 0.0f;
            }

            const float pd_r = -off_i * d_imag[i - 1];
            const float pd_i = off_i * d_real[i - 1];
            const float num_r = rhs_r - pd_r;
            const float num_i = rhs_i - pd_i;

            d_real[i] = (num_r * denom_r + num_i * denom_i) / denom;
            d_imag[i] = (num_i * denom_r - num_r * denom_i) / denom;
        }
    }

    out_real[(ny - 2) * pitch + x] = d_real[n - 1];
    out_imag[(ny - 2) * pitch + x] = d_imag[n - 1];
    for (std::size_t i = n - 1; i >= 1; --i) {
        const float cr = c_real[i - 1];
        const float ci = c_imag[i - 1];
        const float xr = out_real[(i + 1) * pitch + x];
        const float xi = out_imag[(i + 1) * pitch + x];
        const float prod_r = cr * xr - ci * xi;
        const float prod_i = cr * xi + ci * xr;
        out_real[i * pitch + x] = d_real[i - 1] - prod_r;
        out_imag[i * pitch + x] = d_imag[i - 1] - prod_i;
    }
}

void allocate_device_buffers(const SoAField& field, DeviceBuffers& b) {
    const std::size_t plane = field.pitch * field.ny;
    const std::size_t n_row = field.nx - 2;
    const std::size_t n_col = field.ny - 2;

    cuda_malloc(&b.real_current, plane, "cudaMalloc real_current");
    cuda_malloc(&b.imag_current, plane, "cudaMalloc imag_current");
    cuda_malloc(&b.real_next, plane, "cudaMalloc real_next");
    cuda_malloc(&b.imag_next, plane, "cudaMalloc imag_next");
    cuda_malloc(&b.real_scratch, plane, "cudaMalloc real_scratch");
    cuda_malloc(&b.imag_scratch, plane, "cudaMalloc imag_scratch");
    cuda_malloc(&b.potential, plane, "cudaMalloc potential");
    cuda_malloc(&b.beta_x, plane, "cudaMalloc beta_x");
    cuda_malloc(&b.beta_y, plane, "cudaMalloc beta_y");

    cuda_malloc(&b.row_c_real, field.ny * n_row, "cudaMalloc row_c_real");
    cuda_malloc(&b.row_c_imag, field.ny * n_row, "cudaMalloc row_c_imag");
    cuda_malloc(&b.row_d_real, field.ny * n_row, "cudaMalloc row_d_real");
    cuda_malloc(&b.row_d_imag, field.ny * n_row, "cudaMalloc row_d_imag");
    cuda_malloc(&b.col_c_real, field.nx * n_col, "cudaMalloc col_c_real");
    cuda_malloc(&b.col_c_imag, field.nx * n_col, "cudaMalloc col_c_imag");
    cuda_malloc(&b.col_d_real, field.nx * n_col, "cudaMalloc col_d_real");
    cuda_malloc(&b.col_d_imag, field.nx * n_col, "cudaMalloc col_d_imag");
}

void free_device_buffers(DeviceBuffers& b) {
    cuda_free(b.real_current);
    cuda_free(b.imag_current);
    cuda_free(b.real_next);
    cuda_free(b.imag_next);
    cuda_free(b.real_scratch);
    cuda_free(b.imag_scratch);
    cuda_free(b.potential);
    cuda_free(b.beta_x);
    cuda_free(b.beta_y);
    cuda_free(b.row_c_real);
    cuda_free(b.row_c_imag);
    cuda_free(b.row_d_real);
    cuda_free(b.row_d_imag);
    cuda_free(b.col_c_real);
    cuda_free(b.col_c_imag);
    cuda_free(b.col_d_real);
    cuda_free(b.col_d_imag);
}

void compute_beta_arrays(const SimulationConfig& config, const SoAField& field,
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

void upload_initial(const SoAField& field, const std::vector<float>& beta_x, const std::vector<float>& beta_y,
                    DeviceBuffers& b) {
    const std::size_t plane = field.pitch * field.ny;
    const std::size_t bytes = plane * sizeof(float);
    check_cuda(cudaMemcpy(b.real_current, field.real.data(), bytes, cudaMemcpyHostToDevice), "Memcpy real_current");
    check_cuda(cudaMemcpy(b.imag_current, field.imag.data(), bytes, cudaMemcpyHostToDevice), "Memcpy imag_current");
    check_cuda(cudaMemcpy(b.real_next, field.real.data(), bytes, cudaMemcpyHostToDevice), "Memcpy real_next");
    check_cuda(cudaMemcpy(b.imag_next, field.imag.data(), bytes, cudaMemcpyHostToDevice), "Memcpy imag_next");
    check_cuda(cudaMemcpy(b.real_scratch, field.real.data(), bytes, cudaMemcpyHostToDevice), "Memcpy real_scratch");
    check_cuda(cudaMemcpy(b.imag_scratch, field.imag.data(), bytes, cudaMemcpyHostToDevice), "Memcpy imag_scratch");
    check_cuda(cudaMemcpy(b.potential, field.potential.data(), bytes, cudaMemcpyHostToDevice), "Memcpy potential");
    check_cuda(cudaMemcpy(b.beta_x, beta_x.data(), bytes, cudaMemcpyHostToDevice), "Memcpy beta_x");
    check_cuda(cudaMemcpy(b.beta_y, beta_y.data(), bytes, cudaMemcpyHostToDevice), "Memcpy beta_y");
}

void advance_steps_cuda(const SimulationConfig& config, const SoAField& field, DeviceBuffers& b, std::size_t steps) {
    const float off_x = -0.25f * config.dt / (config.mass * config.dx * config.dx);
    const float off_y = -0.25f * config.dt / (config.mass * config.dy * config.dy);

    float* real_current = b.real_current;
    float* imag_current = b.imag_current;
    float* real_next = b.real_next;
    float* imag_next = b.imag_next;
    float* real_scratch = b.real_scratch;
    float* imag_scratch = b.imag_scratch;

    constexpr std::size_t block = 128;
    const std::size_t grid_y = ((field.ny > 2 ? field.ny - 2 : 0) + block - 1) / block;
    const std::size_t grid_x = ((field.nx > 2 ? field.nx - 2 : 0) + block - 1) / block;

    for (std::size_t step = 0; step < steps; ++step) {
        x_cayley_pass_kernel<<<grid_y, block>>>(
            field.nx, field.ny, field.pitch,
            real_current, imag_current,
            real_scratch, imag_scratch,
            b.beta_x, off_x,
            b.row_c_real, b.row_c_imag, b.row_d_real, b.row_d_imag);

        y_cayley_pass_kernel<<<grid_x, block>>>(
            field.nx, field.ny, field.pitch,
            real_scratch, imag_scratch,
            real_next, imag_next,
            b.beta_y, off_y,
            b.col_c_real, b.col_c_imag, b.col_d_real, b.col_d_imag);

        x_cayley_pass_kernel<<<grid_y, block>>>(
            field.nx, field.ny, field.pitch,
            real_next, imag_next,
            real_scratch, imag_scratch,
            b.beta_x, off_x,
            b.row_c_real, b.row_c_imag, b.row_d_real, b.row_d_imag);

        std::swap(real_current, real_scratch);
        std::swap(imag_current, imag_scratch);
    }

    check_cuda(cudaGetLastError(), "CUDA kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    b.real_current = real_current;
    b.imag_current = imag_current;
    b.real_scratch = real_scratch;
    b.imag_scratch = imag_scratch;
}

double l2_norm_host(const std::vector<float>& real, const std::vector<float>& imag,
                    std::size_t nx, std::size_t ny, std::size_t pitch) {
    double sum = 0.0;
    for (std::size_t y = 0; y < ny; ++y) {
        const std::size_t row = y * pitch;
        for (std::size_t x = 0; x < nx; ++x) {
            const double r = real[row + x];
            const double i = imag[row + x];
            sum += r * r + i * i;
        }
    }
    return sum;
}

float max_amp_host(const std::vector<float>& real, const std::vector<float>& imag,
                   std::size_t nx, std::size_t ny, std::size_t pitch) {
    float max_value = 0.0f;
    for (std::size_t y = 0; y < ny; ++y) {
        const std::size_t row = y * pitch;
        for (std::size_t x = 0; x < nx; ++x) {
            const float r = real[row + x];
            const float i = imag[row + x];
            const float amp = std::sqrt(r * r + i * i);
            max_value = std::max(max_value, amp);
        }
    }
    return max_value;
}

}  // namespace

BenchmarkResult run_cn_adi_cuda_impl(const SimulationConfig& config, const InitialState& initial) {
    SoAField field = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, field);

    std::vector<float> beta_x;
    std::vector<float> beta_y;
    compute_beta_arrays(config, field, beta_x, beta_y);

    DeviceBuffers buffers;
    allocate_device_buffers(field, buffers);
    try {
        upload_initial(field, beta_x, beta_y, buffers);
        advance_steps_cuda(config, field, buffers, config.warmup_steps);

        upload_initial(field, beta_x, beta_y, buffers);
        const auto t0 = std::chrono::steady_clock::now();
        advance_steps_cuda(config, field, buffers, config.steps);
        const auto t1 = std::chrono::steady_clock::now();

        const std::size_t plane = field.pitch * field.ny;
        std::vector<float> out_real(plane, 0.0f);
        std::vector<float> out_imag(plane, 0.0f);
        check_cuda(cudaMemcpy(out_real.data(), buffers.real_current, plane * sizeof(float), cudaMemcpyDeviceToHost),
                   "Memcpy result real");
        check_cuda(cudaMemcpy(out_imag.data(), buffers.imag_current, plane * sizeof(float), cudaMemcpyDeviceToHost),
                   "Memcpy result imag");

        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double updated_cells =
            static_cast<double>((config.nx - 2) * (config.ny - 2)) * static_cast<double>(config.steps);

        const BenchmarkResult result{
            .name = "cn_adi_cuda",
            .nx = config.nx,
            .ny = config.ny,
            .steps = config.steps,
            .seconds = seconds,
            .mlups = updated_cells / seconds / 1.0e6,
            .l2_norm = l2_norm_host(out_real, out_imag, field.nx, field.ny, field.pitch),
            .max_amplitude = max_amp_host(out_real, out_imag, field.nx, field.ny, field.pitch),
        };
        free_device_buffers(buffers);
        return result;
    } catch (...) {
        free_device_buffers(buffers);
        throw;
    }
}

}  // namespace physics
