#include "physics/solvers.hpp"

#include <iostream>
#include <stdexcept>

// Fair GPU opponent: the CN-ADI algorithm is unchanged, but every batch of
// tridiagonal systems in an ADI sweep is solved by cuSPARSE's batched
// tridiagonal routine (cusparseCgtsvInterleavedBatch). This is the library
// that directly matches our hand-written CUDA kernel.
//
// The real implementation lives in cn_adi_cusparse.cu and is compiled only
// when the build is configured with -DWAVE2D_WITH_CUSPARSE=ON. Otherwise we
// fall back to the CPU cn-adi solver, mirroring the cuda-cn-adi fallback.

namespace physics {

#if defined(WAVE2D_ENABLE_CUSPARSE)
BenchmarkResult run_cn_adi_cusparse_impl(const SimulationConfig& config, const InitialState& initial);
#endif

BenchmarkResult run_cn_adi_cusparse(const SimulationConfig& config, const InitialState& initial) {
#if defined(WAVE2D_ENABLE_CUSPARSE)
    try {
        return run_cn_adi_cusparse_impl(config, initial);
    } catch (const std::exception& error) {
        std::cerr << "Warning: cuSPARSE path is unavailable (" << error.what()
                  << "). Falling back to CPU cn-adi.\n";
        return run_cn_adi(config, initial);
    }
#else
    std::cerr << "Warning: cuSPARSE support is not enabled in this build. "
                 "Falling back to CPU cn-adi.\n";
    return run_cn_adi(config, initial);
#endif
}

}  // namespace physics
