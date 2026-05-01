#include "physics/benchmark.hpp"

#include "physics/field.hpp"
#include "physics/solvers.hpp"

namespace physics {

BenchmarkResult run_benchmark(const SimulationConfig& config) {
    const InitialState initial = make_initial_state(config);
    switch (config.integrator) {
        case IntegratorKind::cn_adi:
            return run_cn_adi(config, initial);
        case IntegratorKind::explicit_euler:
            return run_euler(config, initial);
        case IntegratorKind::cuda_cn_adi:
            return run_cn_adi_cuda(config, initial);
    }

    throw std::runtime_error("Unknown integrator.");
}

}  // namespace physics
