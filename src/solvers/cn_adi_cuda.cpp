#include "physics/solvers.hpp"

#include <iostream>
#include <stdexcept>

namespace physics {

#if defined(WAVE2D_ENABLE_CUDA)
BenchmarkResult run_cn_adi_cuda_impl(const SimulationConfig& config, const InitialState& initial);
#endif

BenchmarkResult run_cn_adi_cuda(const SimulationConfig& config, const InitialState& initial) {
#if defined(WAVE2D_ENABLE_CUDA)
    try {
        return run_cn_adi_cuda_impl(config, initial);
    } catch (const std::exception& error) {
        std::cerr << "Warning: CUDA path is unavailable (" << error.what()
                  << "). Falling back to CPU cn-adi.\n";
        return run_cn_adi(config, initial);
    }
#else
    std::cerr << "Warning: CUDA support is not enabled in this build. "
                 "Falling back to CPU cn-adi.\n";
    return run_cn_adi(config, initial);
#endif
}

}  // namespace physics
