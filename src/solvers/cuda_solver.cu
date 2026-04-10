#include "physics/solvers.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace physics {
namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__global__ void advance_wavefunction(
    const float* curr_real,
    const float* curr_imag,
    float* next_real,
    float* next_imag,
    const float* potential,
    std::size_t nx,
    std::size_t ny,
    std::size_t pitch,
    float dt,
    float alpha) {
    const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nx || y >= ny) {
        return;
    }

    const std::size_t idx = y * pitch + x;

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny) {
        next_real[idx] = 0.0f;
        next_imag[idx] = 0.0f;
        return;
    }

    const float center_real = curr_real[idx];
    const float center_imag = curr_imag[idx];
    const float lap_real =
        curr_real[idx - 1] + curr_real[idx + 1] + curr_real[idx - pitch] + curr_real[idx + pitch] - 4.0f * center_real;
    const float lap_imag =
        curr_imag[idx - 1] + curr_imag[idx + 1] + curr_imag[idx - pitch] + curr_imag[idx + pitch] - 4.0f * center_imag;
    const float local_potential = potential[idx];

    next_real[idx] = center_real + dt * (local_potential * center_imag - alpha * lap_imag);
    next_imag[idx] = center_imag + dt * (alpha * lap_real - local_potential * center_real);
}

BenchmarkResult run_cuda_impl(const SimulationConfig& config, const InitialState& initial) {
    if (!cuda_backend_available()) {
        throw std::runtime_error("CUDA backend is unavailable on this machine.");
    }

    SoAField host_initial = make_soa_field(initial.nx, initial.ny);
    fill_soa_from_initial(initial, host_initial);

    const std::size_t element_count = host_initial.pitch * host_initial.ny;
    const std::size_t bytes = element_count * sizeof(float);
    const float alpha = 0.5f / config.mass;

    float* curr_real = nullptr;
    float* curr_imag = nullptr;
    float* next_real = nullptr;
    float* next_imag = nullptr;
    float* potential = nullptr;

    const auto cleanup = [&]() {
        cudaFree(curr_real);
        cudaFree(curr_imag);
        cudaFree(next_real);
        cudaFree(next_imag);
        cudaFree(potential);
    };

    check_cuda(cudaMalloc(&curr_real, bytes), "cudaMalloc(curr_real)");
    check_cuda(cudaMalloc(&curr_imag, bytes), "cudaMalloc(curr_imag)");
    check_cuda(cudaMalloc(&next_real, bytes), "cudaMalloc(next_real)");
    check_cuda(cudaMalloc(&next_imag, bytes), "cudaMalloc(next_imag)");
    check_cuda(cudaMalloc(&potential, bytes), "cudaMalloc(potential)");

    try {
        check_cuda(cudaMemcpy(potential, host_initial.potential.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(potential)");

        if (config.warmup_steps > 0) {
            check_cuda(cudaMemcpy(curr_real, host_initial.real.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(curr_real warmup)");
            check_cuda(cudaMemcpy(curr_imag, host_initial.imag.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(curr_imag warmup)");

            const dim3 block(16, 16);
            const dim3 grid(
                static_cast<unsigned int>((host_initial.nx + block.x - 1) / block.x),
                static_cast<unsigned int>((host_initial.ny + block.y - 1) / block.y));

            for (std::size_t step = 0; step < config.warmup_steps; ++step) {
                advance_wavefunction<<<grid, block>>>(
                    curr_real,
                    curr_imag,
                    next_real,
                    next_imag,
                    potential,
                    host_initial.nx,
                    host_initial.ny,
                    host_initial.pitch,
                    config.dt,
                    alpha);
                check_cuda(cudaGetLastError(), "CUDA warmup kernel launch");
                std::swap(curr_real, next_real);
                std::swap(curr_imag, next_imag);
            }

            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");
        }

        check_cuda(cudaMemcpy(curr_real, host_initial.real.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(curr_real)");
        check_cuda(cudaMemcpy(curr_imag, host_initial.imag.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(curr_imag)");

        const dim3 block(16, 16);
        const dim3 grid(
            static_cast<unsigned int>((host_initial.nx + block.x - 1) / block.x),
            static_cast<unsigned int>((host_initial.ny + block.y - 1) / block.y));

        const auto start = std::chrono::steady_clock::now();

        for (std::size_t step = 0; step < config.steps; ++step) {
            advance_wavefunction<<<grid, block>>>(
                curr_real,
                curr_imag,
                next_real,
                next_imag,
                potential,
                host_initial.nx,
                host_initial.ny,
                host_initial.pitch,
                config.dt,
                alpha);
            check_cuda(cudaGetLastError(), "CUDA kernel launch");
            std::swap(curr_real, next_real);
            std::swap(curr_imag, next_imag);
        }

        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        const auto end = std::chrono::steady_clock::now();

        SoAField result_state = make_soa_field(initial.nx, initial.ny);
        result_state.potential = host_initial.potential;
        check_cuda(cudaMemcpy(result_state.real.data(), curr_real, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(result_real)");
        check_cuda(cudaMemcpy(result_state.imag.data(), curr_imag, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(result_imag)");

        const double seconds = std::chrono::duration<double>(end - start).count();
        const double updated_cells = static_cast<double>((config.nx - 2) * (config.ny - 2) * config.steps);

        BenchmarkResult result{
            .name = "cuda",
            .seconds = seconds,
            .mlups = updated_cells / seconds / 1.0e6,
            .l2_norm = l2_norm(result_state),
            .max_amplitude = max_amplitude(result_state),
        };
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}

}  // namespace

bool cuda_backend_available() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    return status == cudaSuccess && device_count > 0;
}

BenchmarkResult run_cuda_backend(const SimulationConfig& config, const InitialState& initial) {
    return run_cuda_impl(config, initial);
}

}  // namespace physics
