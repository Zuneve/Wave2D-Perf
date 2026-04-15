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
        << "Usage: wave2d [options]\n\n"
        << "Simulation parameters:\n"
        << "  --nx N            Grid width  (default: 1024)\n"
        << "  --ny N            Grid height (default: 1024)\n"
        << "  --steps N         Time steps  (default: 400)\n"
        << "  --warmup N        Warmup steps before timing (default: 5)\n"
        << "  --dt DT           Time step size (default: 1e-4)\n"
        << "  --dx DX           Spatial step x (default: 0.1)\n"
        << "  --dy DY           Spatial step y (default: 0.1)\n"
        << "  --mass M          Particle mass (default: 1.0)\n"
        << "  --threads N       Worker threads for parallel CN-ADI (default: auto)\n"
        << "  --integrator cn-adi|euler  Time integrator (default: cn-adi)\n"
        << "  --potential-strength V\n"
        << "  --packet-sigma S  --packet-kx KX  --packet-ky KY\n\n"
        << "Modes:\n"
        << "  --sweep           Run benchmark over grid sizes 128..2048, write CSV\n"
        << "  --dump-every N    Dump |psi|^2 every N steps to a binary file\n"
        << "  --output PATH     Output file for --sweep (CSV) or --dump-every (bin)\n";
    std::exit(EXIT_SUCCESS);
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

IntegratorKind parse_integrator(const char* value) {
    const std::string_view v = value;
    if (v == "cn-adi") {
        return IntegratorKind::cn_adi;
    }
    if (v == "euler") {
        return IntegratorKind::explicit_euler;
    }
    throw std::runtime_error("Unknown integrator: " + std::string(value));
}

}  // namespace

SimulationConfig parse_args(int argc, char** argv) {
    SimulationConfig config;

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            print_help_and_exit();
        }
        if (arg == "--nx") {
            config.nx = parse_size(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--ny") {
            config.ny = parse_size(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--steps") {
            config.steps = parse_size(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--warmup") {
            config.warmup_steps = parse_size(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--dt") {
            config.dt = parse_float(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--dx") {
            config.dx = parse_float(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--dy") {
            config.dy = parse_float(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--mass") {
            config.mass = parse_float(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--threads") {
            config.threads = parse_int(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--integrator") {
            config.integrator = parse_integrator(require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--potential-strength") {
            config.potential_strength = parse_float(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--packet-sigma") {
            config.packet_sigma = parse_float(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--packet-kx") {
            config.packet_kx = parse_float(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--packet-ky") {
            config.packet_ky = parse_float(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--dump-every") {
            config.dump_every = parse_size(arg, require_value(argc, argv, i++));
            continue;
        }
        if (arg == "--output") {
            config.output = require_value(argc, argv, i++);
            continue;
        }
        if (arg == "--sweep") {
            config.sweep = true;
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
    if (config.threads < 0) {
        throw std::runtime_error("threads must be non-negative.");
    }

    return config;
}

std::string to_string(IntegratorKind integrator) {
    switch (integrator) {
        case IntegratorKind::cn_adi:
            return "cn-adi";
        case IntegratorKind::explicit_euler:
            return "euler";
    }

    return "unknown";
}

}  // namespace physics
