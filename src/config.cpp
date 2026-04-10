#include "physics/config.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace physics {
namespace {

[[noreturn]] void print_help_and_exit() {
    std::cout
        << "Usage: physicsProject [options]\n"
        << "  --solver all|naive|soa|threads|omp|cuda\n"
        << "  --nx N --ny N --steps N --warmup N\n"
        << "  --dt DT --dx DX --dy DY --mass M\n"
        << "  --tile-x N --tile-y N --threads N\n"
        << "  --potential-strength V --packet-sigma S --packet-kx KX --packet-ky KY\n";
    std::exit(EXIT_SUCCESS);
}

SolverKind parse_solver(std::string_view value) {
    if (value == "all") {
        return SolverKind::all;
    }
    if (value == "naive") {
        return SolverKind::naive_aos;
    }
    if (value == "soa") {
        return SolverKind::soa_simd;
    }
    if (value == "threads") {
        return SolverKind::threads;
    }
    if (value == "omp") {
        return SolverKind::omp;
    }
    if (value == "cuda") {
        return SolverKind::cuda;
    }
    throw std::runtime_error("Unknown solver: " + std::string(value));
}

std::size_t parse_size(std::string_view label, const char* value) {
    try {
        return static_cast<std::size_t>(std::stoull(value));
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for " + std::string(label) + ": " + value);
    }
}

int parse_int(std::string_view label, const char* value) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for " + std::string(label) + ": " + value);
    }
}

float parse_float(std::string_view label, const char* value) {
    try {
        return std::stof(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid float for " + std::string(label) + ": " + value);
    }
}

const char* require_value(int argc, char** argv, int index) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for option ") + argv[index]);
    }
    return argv[index + 1];
}

}  // namespace

SimulationConfig parse_args(int argc, char** argv) {
    SimulationConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_help_and_exit();
        }

        if (arg == "--solver") {
            config.solver = parse_solver(require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--nx") {
            config.nx = parse_size(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--ny") {
            config.ny = parse_size(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--steps") {
            config.steps = parse_size(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--warmup") {
            config.warmup_steps = parse_size(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--dt") {
            config.dt = parse_float(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--dx") {
            config.dx = parse_float(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--dy") {
            config.dy = parse_float(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--mass") {
            config.mass = parse_float(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--potential-strength") {
            config.potential_strength = parse_float(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--packet-sigma") {
            config.packet_sigma = parse_float(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--packet-kx") {
            config.packet_kx = parse_float(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--packet-ky") {
            config.packet_ky = parse_float(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--tile-x") {
            config.tile_x = parse_size(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--tile-y") {
            config.tile_y = parse_size(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }
        if (arg == "--threads") {
            config.threads = parse_int(arg, require_value(argc, argv, i));
            ++i;
            continue;
        }

        throw std::runtime_error("Unknown option: " + std::string(arg));
    }

    if (config.nx < 4 || config.ny < 4) {
        throw std::runtime_error("Grid must be at least 4x4.");
    }
    if (config.steps == 0) {
        throw std::runtime_error("Step count must be positive.");
    }
    if (config.mass <= 0.0f || config.dx <= 0.0f || config.dy <= 0.0f || config.dt <= 0.0f) {
        throw std::runtime_error("dt, dx, dy and mass must be positive.");
    }
    if (config.tile_x == 0 || config.tile_y == 0) {
        throw std::runtime_error("tile sizes must be positive.");
    }

    return config;
}

std::string to_string(SolverKind kind) {
    switch (kind) {
        case SolverKind::all:
            return "all";
        case SolverKind::naive_aos:
            return "naive";
        case SolverKind::soa_simd:
            return "soa";
        case SolverKind::threads:
            return "threads";
        case SolverKind::omp:
            return "omp";
        case SolverKind::cuda:
            return "cuda";
    }

    return "unknown";
}

std::vector<SolverKind> expand_requested_solvers(SolverKind requested) {
    if (requested == SolverKind::all) {
        return {
            SolverKind::naive_aos,
            SolverKind::soa_simd,
            SolverKind::threads,
            SolverKind::omp,
            SolverKind::cuda,
        };
    }

    return {requested};
}

}  // namespace physics
