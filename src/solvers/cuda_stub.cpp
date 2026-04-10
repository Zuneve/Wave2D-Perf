#include "physics/solvers.hpp"

#include <stdexcept>

namespace physics {

bool cuda_backend_available() {
    return false;
}

BenchmarkResult run_cuda_backend(const SimulationConfig&, const InitialState&) {
    throw std::runtime_error("CUDA backend is unavailable in this build.");
}

}  // namespace physics
