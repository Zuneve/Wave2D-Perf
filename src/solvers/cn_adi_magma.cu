#include "physics/solvers.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <magma_v2.h>

#include "physics/field.hpp"

// ============================================================================
//  MAGMA batched-dense-LU CN-ADI solver.
//
//  MAGMA has no tridiagonal / banded batched routine, so every tridiagonal
//  system of an ADI sweep is materialised as a DENSE n-by-n matrix (mostly
//  zeros) and solved with magma_cgesv_batched (batched LU with partial
//  pivoting). This is deliberately the "naive use of a heavyweight library":
//    * memory  ~ batchCount * n^2   (square grids: O(N^3) total)
//    * compute ~ batchCount * n^3   per sweep
//  so it only fits small grids and is expected to lose by orders of magnitude
//  at large N. Showing that gap is the point of the duel.
//
//  WARNING: prepared "blind" without CUDA/MAGMA available — NOT compiled or run
//  here. Validate on a real GPU box by comparing the L2 norm to CPU cn-adi.
//  A soft memory guard throws (→ CPU fallback) when the dense matrices would
//  exceed MAGMA_MAX_DENSE_BYTES.
// ============================================================================

namespace physics {
namespace {

constexpr std::size_t kMaxDenseBytes = 4ull * 1024 * 1024 * 1024;  // 4 GiB cap

inline void check_cuda(cudaError_t s, const char* what) {
    if (s != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(s));
    }
}

inline std::size_t roundup(std::size_t v, std::size_t m) { return ((v + m - 1) / m) * m; }

// (1 - i*beta) * psi + (0,-off) * (na + nb) — explicit ADI half-operator.
__device__ inline magmaFloatComplex build_rhs(float beta, float off,
                                              magmaFloatComplex psi,
                                              magmaFloatComplex na, magmaFloatComplex nb) {
    const float sum_r = MAGMA_C_REAL(na) + MAGMA_C_REAL(nb);
    const float sum_i = MAGMA_C_IMAG(na) + MAGMA_C_IMAG(nb);
    const float pr = MAGMA_C_REAL(psi);
    const float pi = MAGMA_C_IMAG(psi);
    return MAGMA_C_MAKE((pr + beta * pi) + off * sum_i, (pi - beta * pr) - off * sum_r);
}

// Fill the batched dense matrices + RHS for the x-sweep (matrices pre-zeroed).
// n = nx-2 (system size), batch = ny-2 (one system per interior row).
__global__ void fill_x_kernel(std::size_t nx, std::size_t ny, std::size_t pitch,
                              const float* in_r, const float* in_i, const float* beta_x, float off,
                              magmaFloatComplex* dA, std::size_t ldda, std::size_t mat_stride,
                              magmaFloatComplex* dB, std::size_t lddb) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // element / matrix row 0..n-1
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // system index 0..batch-1
    const int n = static_cast<int>(nx) - 2;
    const int batch = static_cast<int>(ny) - 2;
    if (j >= n || i >= batch) {
        return;
    }
    const std::size_t gy = static_cast<std::size_t>(i) + 1;
    const std::size_t gx = static_cast<std::size_t>(j) + 1;
    const std::size_t idx = gy * pitch + gx;
    const float beta = beta_x[idx];

    const magmaFloatComplex psi = MAGMA_C_MAKE(in_r[idx], in_i[idx]);
    const magmaFloatComplex left  = (j > 0)     ? MAGMA_C_MAKE(in_r[idx - 1], in_i[idx - 1]) : MAGMA_C_ZERO;
    const magmaFloatComplex right = (j + 1 < n) ? MAGMA_C_MAKE(in_r[idx + 1], in_i[idx + 1]) : MAGMA_C_ZERO;

    magmaFloatComplex* A = dA + static_cast<std::size_t>(i) * mat_stride;  // column-major: A(r,c) = A[c*ldda + r]
    const magmaFloatComplex off_c = MAGMA_C_MAKE(0.0f, off);
    A[static_cast<std::size_t>(j) * ldda + j] = MAGMA_C_MAKE(1.0f, beta);          // diagonal
    if (j > 0)     A[static_cast<std::size_t>(j - 1) * ldda + j] = off_c;          // sub-diagonal A(j,j-1)
    if (j + 1 < n) A[static_cast<std::size_t>(j + 1) * ldda + j] = off_c;          // super-diagonal A(j,j+1)

    dB[static_cast<std::size_t>(i) * lddb + j] = build_rhs(beta, off, psi, left, right);
}

__global__ void scatter_x_kernel(std::size_t nx, std::size_t ny, std::size_t pitch,
                                 const magmaFloatComplex* dB, std::size_t lddb,
                                 float* out_r, float* out_i) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = static_cast<int>(nx) - 2;
    const int batch = static_cast<int>(ny) - 2;
    if (j >= n || i >= batch) {
        return;
    }
    const std::size_t gy = static_cast<std::size_t>(i) + 1;
    const std::size_t gx = static_cast<std::size_t>(j) + 1;
    const std::size_t idx = gy * pitch + gx;
    const magmaFloatComplex v = dB[static_cast<std::size_t>(i) * lddb + j];
    out_r[idx] = MAGMA_C_REAL(v);
    out_i[idx] = MAGMA_C_IMAG(v);
    if (j == 0) {
        out_r[gy * pitch] = 0.0f; out_i[gy * pitch] = 0.0f;
        out_r[gy * pitch + nx - 1] = 0.0f; out_i[gy * pitch + nx - 1] = 0.0f;
    }
}

// y-sweep: n = ny-2, batch = nx-2 (one system per interior column).
__global__ void fill_y_kernel(std::size_t nx, std::size_t ny, std::size_t pitch,
                              const float* in_r, const float* in_i, const float* beta_y, float off,
                              magmaFloatComplex* dA, std::size_t ldda, std::size_t mat_stride,
                              magmaFloatComplex* dB, std::size_t lddb) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // element / matrix row 0..n-1
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // system 0..batch-1
    const int n = static_cast<int>(ny) - 2;
    const int batch = static_cast<int>(nx) - 2;
    if (j >= n || i >= batch) {
        return;
    }
    const std::size_t gx = static_cast<std::size_t>(i) + 1;
    const std::size_t gy = static_cast<std::size_t>(j) + 1;
    const std::size_t idx = gy * pitch + gx;
    const float beta = beta_y[idx];

    const magmaFloatComplex psi = MAGMA_C_MAKE(in_r[idx], in_i[idx]);
    const magmaFloatComplex up   = (j > 0)     ? MAGMA_C_MAKE(in_r[idx - pitch], in_i[idx - pitch]) : MAGMA_C_ZERO;
    const magmaFloatComplex down = (j + 1 < n) ? MAGMA_C_MAKE(in_r[idx + pitch], in_i[idx + pitch]) : MAGMA_C_ZERO;

    magmaFloatComplex* A = dA + static_cast<std::size_t>(i) * mat_stride;
    const magmaFloatComplex off_c = MAGMA_C_MAKE(0.0f, off);
    A[static_cast<std::size_t>(j) * ldda + j] = MAGMA_C_MAKE(1.0f, beta);
    if (j > 0)     A[static_cast<std::size_t>(j - 1) * ldda + j] = off_c;
    if (j + 1 < n) A[static_cast<std::size_t>(j + 1) * ldda + j] = off_c;

    dB[static_cast<std::size_t>(i) * lddb + j] = build_rhs(beta, off, psi, up, down);
}

__global__ void scatter_y_kernel(std::size_t nx, std::size_t ny, std::size_t pitch,
                                 const magmaFloatComplex* dB, std::size_t lddb,
                                 float* out_r, float* out_i) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = static_cast<int>(ny) - 2;
    const int batch = static_cast<int>(nx) - 2;
    if (j >= n || i >= batch) {
        return;
    }
    const std::size_t gx = static_cast<std::size_t>(i) + 1;
    const std::size_t gy = static_cast<std::size_t>(j) + 1;
    const std::size_t idx = gy * pitch + gx;
    const magmaFloatComplex v = dB[static_cast<std::size_t>(i) * lddb + j];
    out_r[idx] = MAGMA_C_REAL(v);
    out_i[idx] = MAGMA_C_IMAG(v);
    if (j == 0) {
        out_r[gx] = 0.0f; out_i[gx] = 0.0f;
        out_r[(ny - 1) * pitch + gx] = 0.0f; out_i[(ny - 1) * pitch + gx] = 0.0f;
    }
}

dim3 block2d() { return dim3(32, 8); }
dim3 grid2d(int n, int batch) {
    const dim3 b = block2d();
    return dim3((n + b.x - 1) / b.x, (batch + b.y - 1) / b.y);
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

// Build a device array of pointers spaced by `stride` elements over `base`.
template <typename T>
T** make_ptr_array(T* base, std::size_t stride, int count) {
    std::vector<T*> host(count);
    for (int i = 0; i < count; ++i) {
        host[i] = base + static_cast<std::size_t>(i) * stride;
    }
    T** dev = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&dev), count * sizeof(T*)), "malloc ptr array");
    check_cuda(cudaMemcpy(dev, host.data(), count * sizeof(T*), cudaMemcpyHostToDevice), "H2D ptr array");
    return dev;
}

}  // namespace

BenchmarkResult run_cn_adi_magma_impl(const SimulationConfig& config, const InitialState& initial) {
    SoAField field = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, field);

    std::vector<float> beta_x;
    std::vector<float> beta_y;
    compute_beta(config, field, beta_x, beta_y);

    const std::size_t pitch = field.pitch;
    const std::size_t plane = pitch * field.ny;
    const std::size_t bytes = plane * sizeof(float);

    const int n_x = static_cast<int>(config.nx) - 2;
    const int n_y = static_cast<int>(config.ny) - 2;
    const int n_max = std::max(n_x, n_y);                 // largest system size
    const int batch_max = std::max(n_x, n_y);             // largest batch count
    const std::size_t ldda = roundup(static_cast<std::size_t>(n_max), 32);
    const std::size_t lddb = ldda;
    const std::size_t mat_stride = ldda * static_cast<std::size_t>(n_max);
    const std::size_t dense_bytes = mat_stride * static_cast<std::size_t>(batch_max) * sizeof(magmaFloatComplex);

    if (dense_bytes > kMaxDenseBytes) {
        throw std::runtime_error("dense MAGMA matrices need " +
                                 std::to_string(dense_bytes / (1024 * 1024)) +
                                 " MiB (> cap); grid too large for batched dense LU");
    }

    const float off_x = -0.25f * config.dt / (config.mass * config.dx * config.dx);
    const float off_y = -0.25f * config.dt / (config.mass * config.dy * config.dy);

    float* d_cur_r = nullptr; float* d_cur_i = nullptr;
    float* d_scr_r = nullptr; float* d_scr_i = nullptr;
    float* d_nxt_r = nullptr; float* d_nxt_i = nullptr;
    float* d_beta_x = nullptr; float* d_beta_y = nullptr;
    magmaFloatComplex* dA = nullptr;
    magmaFloatComplex* dB = nullptr;
    magma_int_t* d_ipiv = nullptr;
    magma_int_t* d_info = nullptr;
    magmaFloatComplex** dA_array = nullptr;
    magmaFloatComplex** dB_array = nullptr;
    magma_int_t** dipiv_array = nullptr;
    magma_queue_t queue = nullptr;
    bool magma_started = false;

    auto cleanup = [&]() {
        for (float* p : {d_cur_r, d_cur_i, d_scr_r, d_scr_i, d_nxt_r, d_nxt_i, d_beta_x, d_beta_y}) {
            if (p) cudaFree(p);
        }
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (d_ipiv) cudaFree(d_ipiv);
        if (d_info) cudaFree(d_info);
        if (dA_array) cudaFree(dA_array);
        if (dB_array) cudaFree(dB_array);
        if (dipiv_array) cudaFree(dipiv_array);
        if (queue) magma_queue_destroy(queue);
        if (magma_started) magma_finalize();
    };

    try {
        if (magma_init() != MAGMA_SUCCESS) {
            throw std::runtime_error("magma_init failed");
        }
        magma_started = true;
        magma_device_t device;
        magma_getdevice(&device);
        magma_queue_create(device, &queue);

        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cur_r), bytes), "malloc cur_r");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cur_i), bytes), "malloc cur_i");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_scr_r), bytes), "malloc scr_r");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_scr_i), bytes), "malloc scr_i");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nxt_r), bytes), "malloc nxt_r");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nxt_i), bytes), "malloc nxt_i");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_beta_x), bytes), "malloc beta_x");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_beta_y), bytes), "malloc beta_y");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dA), dense_bytes), "malloc dA");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&dB),
                              lddb * static_cast<std::size_t>(batch_max) * sizeof(magmaFloatComplex)), "malloc dB");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_ipiv),
                              static_cast<std::size_t>(n_max) * batch_max * sizeof(magma_int_t)), "malloc ipiv");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_info), batch_max * sizeof(magma_int_t)), "malloc info");

        dA_array = make_ptr_array(dA, mat_stride, batch_max);
        dB_array = make_ptr_array(dB, lddb, batch_max);
        dipiv_array = make_ptr_array(d_ipiv, static_cast<std::size_t>(n_max), batch_max);

        check_cuda(cudaMemcpy(d_beta_x, beta_x.data(), bytes, cudaMemcpyHostToDevice), "H2D beta_x");
        check_cuda(cudaMemcpy(d_beta_y, beta_y.data(), bytes, cudaMemcpyHostToDevice), "H2D beta_y");

        const dim3 blk = block2d();
        const dim3 grid_x = grid2d(n_x, n_y);
        const dim3 grid_y = grid2d(n_y, n_x);

        auto sweep_x = [&](const float* in_r, const float* in_i, float* out_r, float* out_i) {
            check_cuda(cudaMemset(dA, 0, dense_bytes), "memset dA x");
            fill_x_kernel<<<grid_x, blk>>>(config.nx, config.ny, pitch, in_r, in_i, d_beta_x, off_x,
                                           dA, ldda, mat_stride, dB, lddb);
            magma_cgesv_batched(n_x, 1, dA_array, static_cast<magma_int_t>(ldda), dipiv_array,
                                dB_array, static_cast<magma_int_t>(lddb), d_info, n_y, queue);
            scatter_x_kernel<<<grid_x, blk>>>(config.nx, config.ny, pitch, dB, lddb, out_r, out_i);
        };
        auto sweep_y = [&](const float* in_r, const float* in_i, float* out_r, float* out_i) {
            check_cuda(cudaMemset(dA, 0, dense_bytes), "memset dA y");
            fill_y_kernel<<<grid_y, blk>>>(config.nx, config.ny, pitch, in_r, in_i, d_beta_y, off_y,
                                           dA, ldda, mat_stride, dB, lddb);
            magma_cgesv_batched(n_y, 1, dA_array, static_cast<magma_int_t>(ldda), dipiv_array,
                                dB_array, static_cast<magma_int_t>(lddb), d_info, n_x, queue);
            scatter_y_kernel<<<grid_y, blk>>>(config.nx, config.ny, pitch, dB, lddb, out_r, out_i);
        };

        auto run_steps = [&](std::size_t steps) {
            for (std::size_t step = 0; step < steps; ++step) {
                sweep_x(d_cur_r, d_cur_i, d_scr_r, d_scr_i);
                sweep_y(d_scr_r, d_scr_i, d_nxt_r, d_nxt_i);
                sweep_x(d_nxt_r, d_nxt_i, d_scr_r, d_scr_i);
                std::swap(d_cur_r, d_scr_r);
                std::swap(d_cur_i, d_scr_i);
            }
            magma_queue_sync(queue);
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
            .name = "cn_adi_magma_dense",
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
