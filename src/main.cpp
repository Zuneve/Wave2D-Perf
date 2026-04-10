#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "physics/benchmark.hpp"

int main(int argc, char** argv) {
    try {
        const physics::SimulationConfig config = physics::parse_args(argc, argv);
        const auto results = physics::run_requested_benchmarks(config);
        const char* requested_solver = "all";

        switch (config.solver) {
            case physics::SolverKind::all:
                requested_solver = "all";
                break;
            case physics::SolverKind::naive_aos:
                requested_solver = "naive";
                break;
            case physics::SolverKind::soa_simd:
                requested_solver = "soa";
                break;
            case physics::SolverKind::threads:
                requested_solver = "threads";
                break;
            case physics::SolverKind::omp:
                requested_solver = "omp";
                break;
            case physics::SolverKind::cuda:
                requested_solver = "cuda";
                break;
        }

        std::cout << "2D wave-function benchmark\n";
        std::cout << "grid=" << config.nx << "x" << config.ny
                  << ", steps=" << config.steps
                  << ", dt=" << config.dt
                  << ", solver=" << requested_solver << "\n\n";

        std::cout << std::left
                  << std::setw(18) << "backend"
                  << std::setw(14) << "time (s)"
                  << std::setw(14) << "MLUPS"
                  << std::setw(18) << "L2 norm"
                  << std::setw(18) << "max |psi|"
                  << "\n";

        for (const auto& result : results) {
            std::cout << std::left
                      << std::setw(18) << result.name
                      << std::setw(14) << std::fixed << std::setprecision(6) << result.seconds
                      << std::setw(14) << std::fixed << std::setprecision(2) << result.mlups
                      << std::setw(18) << std::scientific << std::setprecision(6) << result.l2_norm
                      << std::setw(18) << std::scientific << std::setprecision(6) << result.max_amplitude
                      << "\n";
        }

#if !defined(PHYSICS_HAS_OPENMP) || !PHYSICS_HAS_OPENMP
        if (config.solver == physics::SolverKind::all) {
            std::cout << "\nOpenMP backend skipped: this build has no OpenMP support.\n";
        }
#endif

        if (config.solver == physics::SolverKind::all && !physics::cuda_backend_available()) {
            std::cout << "CUDA backend skipped: toolkit or runtime device is unavailable.\n";
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}

//
// cmake --build build -j
// ./build/physicsProject --nx 1024 --ny 1024 --steps 200 --warmup 3
// ./build/physicsProject --solver threads --threads 8 --nx 1024 --ny 1024 --steps 200 --warmup 3
//
