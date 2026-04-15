#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "physics/benchmark.hpp"
#include "physics/config.hpp"
#include "physics/field.hpp"
#include "physics/solvers.hpp"

namespace {

void print_header() {
    std::cout << std::left
              << std::setw(10) << "nx"
              << std::setw(10) << "ny"
              << std::setw(10) << "steps"
              << std::setw(14) << "time (s)"
              << std::setw(14) << "MLUPS"
              << std::setw(18) << "L2 norm"
              << std::setw(18) << "max |psi|"
              << "\n"
              << std::string(94, '-') << "\n";
}

void print_result(const physics::BenchmarkResult& r) {
    std::cout << std::left
              << std::setw(10) << r.nx
              << std::setw(10) << r.ny
              << std::setw(10) << r.steps
              << std::setw(14) << std::fixed      << std::setprecision(6) << r.seconds
              << std::setw(14) << std::fixed      << std::setprecision(2) << r.mlups
              << std::setw(18) << std::scientific << std::setprecision(6) << r.l2_norm
              << std::setw(18) << std::scientific << std::setprecision(6) << r.max_amplitude
              << "\n";
}

void write_csv(const std::string& path, const std::vector<physics::BenchmarkResult>& results) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open CSV output: " + path);

    f << "integrator,nx,ny,cells,steps,seconds,mlups,l2_norm,max_amplitude\n";
    for (const auto& r : results) {
        f << r.name << "," << r.nx << "," << r.ny << "," << (r.nx * r.ny) << "," << r.steps << ","
          << std::fixed      << std::setprecision(9) << r.seconds      << ","
          << std::fixed      << std::setprecision(6) << r.mlups        << ","
          << std::scientific << std::setprecision(9) << r.l2_norm      << ","
          << std::scientific << std::setprecision(6) << r.max_amplitude << "\n";
    }
}

void run_single(const physics::SimulationConfig& config) {
    std::cout << "Wave2D — 2D time-dependent Schrodinger benchmark\n"
              << "grid: " << config.nx << "x" << config.ny
              << "  steps: " << config.steps
              << "  dt: " << config.dt
              << "  integrator: " << physics::to_string(config.integrator)
              << "  dx: " << config.dx << "\n\n";

    if (config.integrator == physics::IntegratorKind::explicit_euler) {
        std::cout << "Warning: explicit Euler is kept only as a naive baseline. "
                     "It is not norm-preserving and is unreliable for long-time Schrödinger dynamics.\n\n";
    }

    print_header();
    print_result(physics::run_benchmark(config));
}

void run_sweep(const physics::SimulationConfig& base) {
    const std::vector<std::size_t> sizes = {128, 256, 512, 1024, 2048};

    std::cout << "Sweep benchmark — integrator: " << physics::to_string(base.integrator) << "\n"
              << "steps: " << base.steps
              << "  dt: " << base.dt
              << "  dx: " << base.dx << "\n\n";
    print_header();

    std::vector<physics::BenchmarkResult> results;
    for (const auto size : sizes) {
        physics::SimulationConfig config = base;
        config.nx = size;
        config.ny = size;
        const auto result = physics::run_benchmark(config);
        print_result(result);
        results.push_back(result);
    }

    const std::string csv_path = base.output.empty() ? "benchmark.csv" : base.output;
    write_csv(csv_path, results);
    std::cout << "\nResults saved to " << csv_path << "\n"
              << "Visualize: python3 tools/plot_benchmark.py " << csv_path << "\n";
}

void run_dump(const physics::SimulationConfig& config) {
    const std::string out_path = config.output.empty() ? "simulation.bin" : config.output;
    const std::size_t n_frames = config.steps / config.dump_every + 1;
    const std::size_t mb = n_frames * config.nx * config.ny * sizeof(float) / (1024 * 1024);

    std::cout << "Simulation dump — " << config.nx << "x" << config.ny
              << "  steps: " << config.steps
              << "  integrator: " << physics::to_string(config.integrator)
              << "  dump every: " << config.dump_every << " steps\n"
              << n_frames << " frames  ~" << mb << " MB  -> " << out_path << "\n";

    const physics::InitialState initial = physics::make_initial_state(config);
    switch (config.integrator) {
        case physics::IntegratorKind::cn_adi:
            physics::run_cn_adi_dump(config, initial, out_path);
            break;
        case physics::IntegratorKind::explicit_euler:
            physics::run_euler_dump(config, initial, out_path);
            break;
    }

    std::cout << "Done.\n"
              << "Animate:  python3 tools/animate_simulation.py " << out_path << "\n"
              << "Save MP4: python3 tools/animate_simulation.py " << out_path << " -o wave.mp4\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const physics::SimulationConfig config = physics::parse_args(argc, argv);

        if (config.sweep) {
            run_sweep(config);
        } else if (config.dump_every > 0) {
            run_dump(config);
        } else {
            run_single(config);
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
