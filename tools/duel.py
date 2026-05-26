#!/usr/bin/env python3
"""Run the "battle": our hand-written CN-ADI against standard-library solvers.

Runs each requested integrator across a set of grid sizes, parses the MLUPS and
L2 norm from wave2d's output, and prints a comparison table with the speedup of
our hand-written cn-adi over every opponent. Results are also written to CSV.

An opponent that is not available in the current build (e.g. a GPU library on a
CPU-only machine) prints a "Falling back to CPU cn-adi" warning on stderr; such
rows are flagged as [fallback] and excluded from the speedup verdict.

Examples
--------
    # CPU duel on this machine
    python3 tools/duel.py --integrators cn-adi lapack-cn-adi --threads 8

    # GPU duel on a CUDA box (build with -DWAVE2D_WITH_CUSPARSE=ON etc.)
    python3 tools/duel.py --bin ./build_cuda/wave2d \
        --integrators cuda-cn-adi cusparse-cn-adi magma-cn-adi
"""

import argparse
import csv
import re
import subprocess
import sys

# Grid size -> default step count (smaller grids run more steps for stable timing).
DEFAULT_GRID_STEPS = {128: 400, 256: 300, 512: 150, 1024: 60, 2048: 20}

ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
)


def run_one(binary, integrator, n, steps, threads):
    """Return (mlups, l2, seconds, fell_back) for a single run."""
    cmd = [binary, "--integrator", integrator, "--nx", str(n), "--ny", str(n),
           "--steps", str(steps), "--threads", str(threads)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{proc.stderr}")
    fell_back = "Falling back to CPU" in proc.stderr
    mlups = l2 = seconds = None
    for line in proc.stdout.splitlines():
        m = ROW_RE.match(line)
        if m:
            seconds = float(m.group(4))
            mlups = float(m.group(5))
            l2 = float(m.group(6))
    if mlups is None:
        raise RuntimeError(f"could not parse output of {' '.join(cmd)}:\n{proc.stdout}")
    return mlups, l2, seconds, fell_back


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="./build/wave2d", help="path to wave2d binary")
    ap.add_argument("--integrators", nargs="+",
                    default=["cn-adi", "lapack-cn-adi"],
                    help="first one is treated as 'ours' (the speedup reference)")
    ap.add_argument("--grids", nargs="+", type=int, default=[128, 256, 512, 1024])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=0, help="override steps for every grid")
    ap.add_argument("--csv", default="duel.csv")
    args = ap.parse_args()

    ours = args.integrators[0]
    print(f"Duel — reference (ours): {ours}   threads: {args.threads}   binary: {args.bin}\n")

    rows = []
    for n in args.grids:
        steps = args.steps or DEFAULT_GRID_STEPS.get(n, 100)
        print(f"grid {n}x{n}, {steps} steps")
        ref_mlups = None
        for integ in args.integrators:
            try:
                mlups, l2, seconds, fb = run_one(args.bin, integ, n, steps, args.threads)
            except RuntimeError as e:
                print(f"  {integ:<16} ERROR: {e}")
                continue
            if integ == ours:
                ref_mlups = mlups
            speed = f"{ref_mlups / mlups:6.2f}x" if (ref_mlups and mlups) else "   —  "
            tag = " [fallback]" if fb else ""
            print(f"  {integ:<16} {mlups:9.2f} MLUPS   {seconds:9.4f} s   "
                  f"L2={l2:.6e}   ours/this={speed}{tag}")
            rows.append({"grid": n, "steps": steps, "integrator": integ,
                         "mlups": mlups, "seconds": seconds, "l2_norm": l2,
                         "fallback": fb, "speedup_ours_over_this":
                         (ref_mlups / mlups) if (ref_mlups and mlups) else ""})
        print()

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Results written to {args.csv}")


if __name__ == "__main__":
    sys.exit(main())
