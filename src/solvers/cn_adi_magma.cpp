#include "physics/solvers.hpp"

#include <iostream>
#include <stdexcept>

// "Standard library" GPU opponent requested explicitly (MAGMA). MAGMA has no
// tridiagonal / banded batched solver, so each tridiagonal system of an ADI
// sweep is materialised as a small DENSE matrix and solved by MAGMA's batched
// dense LU driver magma_cgesv_batched. This is intentionally the "naive use of
// a standard GPU library": it is O(n^3) per system and O(batch*n^2) in memory,
// so it is only feasible for small grids and is expected to lose badly at large
// n. That contrast is exactly what the duel is meant to show.
//
// The real implementation lives in cn_adi_magma.cu and is compiled only when
// the build is configured with -DWAVE2D_WITH_MAGMA=ON. Otherwise we fall back
// to the CPU cn-adi solver.

namespace physics {

#if defined(WAVE2D_ENABLE_MAGMA)
BenchmarkResult run_cn_adi_magma_impl(const SimulationConfig& config, const InitialState& initial);
#endif

BenchmarkResult run_cn_adi_magma(const SimulationConfig& config, const InitialState& initial) {
#if defined(WAVE2D_ENABLE_MAGMA)
    try {
        return run_cn_adi_magma_impl(config, initial);
    } catch (const std::exception& error) {
        std::cerr << "Warning: MAGMA path is unavailable (" << error.what()
                  << "). Falling back to CPU cn-adi.\n";
        return run_cn_adi(config, initial);
    }
#else
    std::cerr << "Warning: MAGMA support is not enabled in this build. "
                 "Falling back to CPU cn-adi.\n";
    return run_cn_adi(config, initial);
#endif
}

}  // namespace physics
