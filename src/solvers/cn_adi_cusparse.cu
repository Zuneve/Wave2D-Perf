#include "physics/solvers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include "physics/field.hpp"

// ============================================================================
//  cuSPARSE batched-tridiagonal CN-ADI solver.
//
//  Same algorithm and same tridiagonal systems as the hand-written CUDA kernel,
//  but each ADI sweep's batch of independent tridiagonal solves is handed to
//  cusparseCgtsvInterleavedBatch. cuSPARSE wants the systems in *interleaved*
//  layout: element j of system i sits at index  j * batchCount + i  (this gives
//  coalesced access across the batch).
//
//  WARNING: this file has been prepared "blind" on a machine without CUDA and
//  has NOT been compiled or run. Treat the first build on a real GPU box as the
//  validation step (compare the L2 norm against the CPU cn-adi solver).
// ============================================================================

namespace physics {
namespace {

inline void check_cuda(cudaError_t s, const char* what) {
    if (s != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
    }
}

inline void check_cusparse(cusparseStatus_t s, const char* what) {
    if (s != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cusparse status " + std::to_string(static_cast<int>(s)));
    }
}

inline void cuda_malloc(void** p, std::size_t bytes, const char* what) {
    check_cuda(cudaMalloc(p, bytes), what);
}

// (1 - i*beta) * psi  +  (0,-off) * (na + nb)   — the explicit ADI half-operator.
__device__ inline cuComplex build_rhs(float beta, float off,
                                      cuComplex psi, cuComplex na, cuComplex nb) {
    const float sum_r = na.x + nb.x;
    const float sum_i = na.y + nb.y;
    cuComplex r;
    r.x = (psi.x + beta * psi.y) + off * sum_i;   // real
    r.y = (psi.y - beta * psi.x) - off * sum_r;   // imag
    return r;
}

// Build the interleaved tridiagonal batch for the x-sweep (one system per row).
// batchCount = ny-2, m = nx-2.
__global__ void build_x_kernel(std::size_t nx, std::size_t ny, std::size_t pitch,
                               const float* in_r, const float* in_i, const float* beta_x, float off,
                               cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* x) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // element index 0..m-1 (grid x = j+1)
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // system index 0..batch-1 (grid y = i+1)
    const int m = static_cast<int>(nx) - 2;
    const int batch = static_cast<int>(ny) - 2;
    if (j >= m || i >= batch) {
        return;
    }
    const std::size_t gy = static_cast<std::size_t>(i) + 1;
    const std::size_t gx = static_cast<std::size_t>(j) + 1;
    const std::size_t idx = gy * pitch + gx;
    const float beta = beta_x[idx];

    const cuComplex psi = make_cuComplex(in_r[idx], in_i[idx]);
    const cuComplex left  = (j > 0)     ? make_cuComplex(in_r[idx - 1], in_i[idx - 1]) : make_cuComplex(0, 0);
    const cuComplex right = (j + 1 < m) ? make_cuComplex(in_r[idx + 1], in_i[idx + 1]) : make_cuComplex(0, 0);

    const std::size_t k = static_cast<std::size_t>(j) * batch + i;  // interleaved position
    d[k]  = make_cuComplex(1.0f, beta);
    dl[k] = (j > 0)     ? make_cuComplex(0.0f, off) : make_cuComplex(0.0f, 0.0f);
    du[k] = (j + 1 < m) ? make_cuComplex(0.0f, off) : make_cuComplex(0.0f, 0.0f);
    x[k]  = build_rhs(beta, off, psi, left, right);
}

// Scatter the x-sweep solution back into the SoA field and zero the boundaries.
__global__ void scatter_x_kernel(std::size_t nx, std::size_t ny, std::size_t pitch,
                                 const cuComplex* x, float* out_r, float* out_i) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int m = static_cast<int>(nx) - 2;
    const int batch = static_cast<int>(ny) - 2;
    if (j >= m || i >= batch) {
        return;
    }
    const std::size_t gy = static_cast<std::size_t>(i) + 1;
    const std::size_t gx = static_cast<std::size_t>(j) + 1;
    const std::size_t idx = gy * pitch + gx;
    const std::size_t k = static_cast<std::size_t>(j) * batch + i;
    out_r[idx] = x[k].x;
    out_i[idx] = x[k].y;
    if (j == 0) {            // left/right column boundaries of this row
        out_r[gy * pitch] = 0.0f;
        out_i[gy * pitch] = 0.0f;
        out_r[gy * pitch + nx - 1] = 0.0f;
        out_i[gy * pitch + nx - 1] = 0.0f;
    }
}

// Build the interleaved tridiagonal batch for the y-sweep (one system per column).
// batchCount = nx-2, m = ny-2.
__global__ void build_y_kernel(std::size_t nx, std::size_t ny, std::size_t pitch,
                               const float* in_r, const float* in_i, const float* beta_y, float off,
                               cuComplex* dl, cuComplex* d, cuComplex* du, cuComplex* x) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // element index 0..m-1 (grid y = j+1)
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // system index 0..batch-1 (grid x = i+1)
    const int m = static_cast<int>(ny) - 2;
    const int batch = static_cast<int>(nx) - 2;
    if (j >= m || i >= batch) {
        return;
    }
    const std::size_t gx = static_cast<std::size_t>(i) + 1;
    const std::size_t gy = static_cast<std::size_t>(j) + 1;
    const std::size_t idx = gy * pitch + gx;
    const float beta = beta_y[idx];

    const cuComplex psi = make_cuComplex(in_r[idx], in_i[idx]);
    const cuComplex up   = (j > 0)     ? make_cuComplex(in_r[idx - pitch], in_i[idx - pitch]) : make_cuComplex(0, 0);
    const cuComplex down = (j + 1 < m) ? make_cuComplex(in_r[idx + pitch], in_i[idx + pitch]) : make_cuComplex(0, 0);

    const std::size_t k = static_cast<std::size_t>(j) * batch + i;
    d[k]  = make_cuComplex(1.0f, beta);
    dl[k] = (j > 0)     ? make_cuComplex(0.0f, off) : make_cuComplex(0.0f, 0.0f);
    du[k] = (j + 1 < m) ? make_cuComplex(0.0f, off) : make_cuComplex(0.0f, 0.0f);
    x[k]  = build_rhs(beta, off, psi, up, down);
}

__global__ void scatter_y_kernel(std::size_t nx, std::size_t ny, std::size_t pitch,
                                 const cuComplex* x, float* out_r, float* out_i) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int m = static_cast<int>(ny) - 2;
    const int batch = static_cast<int>(nx) - 2;
    if (j >= m || i >= batch) {
        return;
    }
    const std::size_t gx = static_cast<std::size_t>(i) + 1;
    const std::size_t gy = static_cast<std::size_t>(j) + 1;
    const std::size_t idx = gy * pitch + gx;
    const std::size_t k = static_cast<std::size_t>(j) * batch + i;
    out_r[idx] = x[k].x;
    out_i[idx] = x[k].y;
    if (j == 0) {            // top/bottom row boundaries of this column
        out_r[gx] = 0.0f;
        out_i[gx] = 0.0f;
        out_r[(ny - 1) * pitch + gx] = 0.0f;
        out_i[(ny - 1) * pitch + gx] = 0.0f;
    }
}

dim3 block2d() { return dim3(32, 8); }
dim3 grid2d(int m, int batch) {
    const dim3 b = block2d();
    return dim3((m + b.x - 1) / b.x, (batch + b.y - 1) / b.y);
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

}  // namespace

BenchmarkResult run_cn_adi_cusparse_impl(const SimulationConfig& config, const InitialState& initial) {
    SoAField field = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, field);

    std::vector<float> beta_x;
    std::vector<float> beta_y;
    compute_beta(config, field, beta_x, beta_y);

    const std::size_t pitch = field.pitch;
    const std::size_t plane = pitch * field.ny;
    const std::size_t bytes = plane * sizeof(float);
    const int m_x = static_cast<int>(config.nx) - 2;
    const int m_y = static_cast<int>(config.ny) - 2;
    const std::size_t sys_elems = static_cast<std::size_t>(m_x) * static_cast<std::size_t>(m_y);
    const float off_x = -0.25f * config.dt / (config.mass * config.dx * config.dx);
    const float off_y = -0.25f * config.dt / (config.mass * config.dy * config.dy);

    float* d_cur_r = nullptr; float* d_cur_i = nullptr;
    float* d_scr_r = nullptr; float* d_scr_i = nullptr;
    float* d_nxt_r = nullptr; float* d_nxt_i = nullptr;
    float* d_beta_x = nullptr; float* d_beta_y = nullptr;
    cuComplex* dl = nullptr; cuComplex* dd = nullptr; cuComplex* du = nullptr; cuComplex* xx = nullptr;
    void* buffer = nullptr;
    cusparseHandle_t handle = nullptr;

    auto cleanup = [&]() {
        for (float* p : {d_cur_r, d_cur_i, d_scr_r, d_scr_i, d_nxt_r, d_nxt_i, d_beta_x, d_beta_y}) {
            if (p) cudaFree(p);
        }
        for (cuComplex* p : {dl, dd, du, xx}) {
            if (p) cudaFree(p);
        }
        if (buffer) cudaFree(buffer);
        if (handle) cusparseDestroy(handle);
    };

    try {
        cuda_malloc(reinterpret_cast<void**>(&d_cur_r), bytes, "malloc cur_r");
        cuda_malloc(reinterpret_cast<void**>(&d_cur_i), bytes, "malloc cur_i");
        cuda_malloc(reinterpret_cast<void**>(&d_scr_r), bytes, "malloc scr_r");
        cuda_malloc(reinterpret_cast<void**>(&d_scr_i), bytes, "malloc scr_i");
        cuda_malloc(reinterpret_cast<void**>(&d_nxt_r), bytes, "malloc nxt_r");
        cuda_malloc(reinterpret_cast<void**>(&d_nxt_i), bytes, "malloc nxt_i");
        cuda_malloc(reinterpret_cast<void**>(&d_beta_x), bytes, "malloc beta_x");
        cuda_malloc(reinterpret_cast<void**>(&d_beta_y), bytes, "malloc beta_y");
        cuda_malloc(reinterpret_cast<void**>(&dl), sys_elems * sizeof(cuComplex), "malloc dl");
        cuda_malloc(reinterpret_cast<void**>(&dd), sys_elems * sizeof(cuComplex), "malloc d");
        cuda_malloc(reinterpret_cast<void**>(&du), sys_elems * sizeof(cuComplex), "malloc du");
        cuda_malloc(reinterpret_cast<void**>(&xx), sys_elems * sizeof(cuComplex), "malloc x");

        check_cuda(cudaMemcpy(d_beta_x, beta_x.data(), bytes, cudaMemcpyHostToDevice), "H2D beta_x");
        check_cuda(cudaMemcpy(d_beta_y, beta_y.data(), bytes, cudaMemcpyHostToDevice), "H2D beta_y");

        check_cusparse(cusparseCreate(&handle), "cusparseCreate");

        // The interleaved-batch buffer size is the same for both sweeps because
        // m*batch is identical; query with the larger m to be safe.
        const int algo = 0;  // 0 = cuThomas (no pivoting), matches our hand solver
        const int batch_x = m_y;  // x-sweep: one system per interior row
        const int batch_y = m_x;  // y-sweep: one system per interior column
        std::size_t buf_x = 0;
        std::size_t buf_y = 0;
        check_cusparse(cusparseCgtsvInterleavedBatch_bufferSizeExt(
            handle, algo, m_x, dl, dd, du, xx, batch_x, &buf_x), "gtsv bufferSize x");
        check_cusparse(cusparseCgtsvInterleavedBatch_bufferSizeExt(
            handle, algo, m_y, dl, dd, du, xx, batch_y, &buf_y), "gtsv bufferSize y");
        cuda_malloc(&buffer, std::max(buf_x, buf_y), "malloc gtsv buffer");

        const dim3 blk = block2d();
        const dim3 grid_x = grid2d(m_x, batch_x);
        const dim3 grid_y = grid2d(m_y, batch_y);

        auto sweep_x = [&](const float* in_r, const float* in_i, float* out_r, float* out_i) {
            build_x_kernel<<<grid_x, blk>>>(config.nx, config.ny, pitch, in_r, in_i, d_beta_x, off_x, dl, dd, du, xx);
            check_cusparse(cusparseCgtsvInterleavedBatch(handle, algo, m_x, dl, dd, du, xx, batch_x, buffer),
                           "gtsv x");
            scatter_x_kernel<<<grid_x, blk>>>(config.nx, config.ny, pitch, xx, out_r, out_i);
        };
        auto sweep_y = [&](const float* in_r, const float* in_i, float* out_r, float* out_i) {
            build_y_kernel<<<grid_y, blk>>>(config.nx, config.ny, pitch, in_r, in_i, d_beta_y, off_y, dl, dd, du, xx);
            check_cusparse(cusparseCgtsvInterleavedBatch(handle, algo, m_y, dl, dd, du, xx, batch_y, buffer),
                           "gtsv y");
            scatter_y_kernel<<<grid_y, blk>>>(config.nx, config.ny, pitch, xx, out_r, out_i);
        };

        auto run_steps = [&](std::size_t steps) {
            for (std::size_t step = 0; step < steps; ++step) {
                sweep_x(d_cur_r, d_cur_i, d_scr_r, d_scr_i);
                sweep_y(d_scr_r, d_scr_i, d_nxt_r, d_nxt_i);
                sweep_x(d_nxt_r, d_nxt_i, d_scr_r, d_scr_i);
                std::swap(d_cur_r, d_scr_r);
                std::swap(d_cur_i, d_scr_i);
            }
            check_cuda(cudaGetLastError(), "kernel launch");
            check_cuda(cudaDeviceSynchronize(), "synchronize");
        };

        auto upload_initial = [&]() {
            check_cuda(cudaMemcpy(d_cur_r, field.real.data(), bytes, cudaMemcpyHostToDevice), "H2D cur_r");
            check_cuda(cudaMemcpy(d_cur_i, field.imag.data(), bytes, cudaMemcpyHostToDevice), "H2D cur_i");
        };

        upload_initial();
        run_steps(config.warmup_steps);

        upload_initial();
        const auto t0 = std::chrono::steady_clock::now();
        run_steps(config.steps);
        const auto t1 = std::chrono::steady_clock::now();

        std::vector<float> out_r(plane, 0.0f);
        std::vector<float> out_i(plane, 0.0f);
        check_cuda(cudaMemcpy(out_r.data(), d_cur_r, bytes, cudaMemcpyDeviceToHost), "D2H r");
        check_cuda(cudaMemcpy(out_i.data(), d_cur_i, bytes, cudaMemcpyDeviceToHost), "D2H i");

        SoAField result = field;
        result.real = out_r;
        result.imag = out_i;

        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double updated = static_cast<double>((config.nx - 2) * (config.ny - 2)) *
                               static_cast<double>(config.steps);

        BenchmarkResult br{
            .name = "cn_adi_cusparse",
            .nx = config.nx,
            .ny = config.ny,
            .steps = config.steps,
            .seconds = seconds,
            .mlups = updated / seconds / 1.0e6,
            .l2_norm = l2_norm(result),
            .max_amplitude = max_amplitude(result),
        };
        cleanup();
        return br;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace physics
