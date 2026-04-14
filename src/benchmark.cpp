#include "physics/benchmark.hpp"

#include "physics/field.hpp"
#include "physics/solvers.hpp"

namespace physics {

BenchmarkResult run_benchmark(const SimulationConfig& config) {
    const InitialState initial = make_initial_state(config);
    return run_naive(config, initial);
}

}  // namespace physics
