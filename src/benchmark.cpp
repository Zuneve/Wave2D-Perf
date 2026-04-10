#include "physics/benchmark.hpp"

#include <stdexcept>

#include "physics/field.hpp"
#include "physics/solvers.hpp"

namespace physics {

std::vector<BenchmarkResult> run_requested_benchmarks(const SimulationConfig& config) {
    const InitialState initial = make_initial_state(config);
    std::vector<BenchmarkResult> results;

    for (const SolverKind solver : expand_requested_solvers(config.solver)) {
        switch (solver) {
            case SolverKind::naive_aos:
                results.push_back(run_naive_aos(config, initial));
                break;
            case SolverKind::soa_simd:
                results.push_back(run_soa_simd(config, initial, false));
                break;
            case SolverKind::threads:
                results.push_back(run_soa_threads(config, initial));
                break;
            case SolverKind::omp:
#if defined(PHYSICS_HAS_OPENMP) && PHYSICS_HAS_OPENMP
                results.push_back(run_soa_simd(config, initial, true));
#else
                if (config.solver == SolverKind::omp) {
                    throw std::runtime_error("OpenMP backend was requested but this build does not include OpenMP support.");
                }
#endif
                break;
            case SolverKind::cuda:
                if (!cuda_backend_available()) {
                    if (config.solver == SolverKind::cuda) {
                        throw std::runtime_error("CUDA backend was requested, but CUDA is not available in this build or on this machine.");
                    }
                    break;
                }
                results.push_back(run_cuda_backend(config, initial));
                break;
            case SolverKind::all:
                break;
        }
    }

    return results;
}

}  // namespace physics
