#!/usr/bin/env python3
"""
Plot Wave2D benchmark sweep results.

Usage:
    python3 tools/plot_benchmark.py benchmark.csv
    python3 tools/plot_benchmark.py benchmark.csv -o benchmark.png
"""

from __future__ import annotations

import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def read_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "integrator": row.get("integrator", "unknown"),
                    "nx": int(row["nx"]),
                    "ny": int(row["ny"]),
                    "cells": int(row["cells"]),
                    "steps": int(row["steps"]),
                    "seconds": float(row["seconds"]),
                    "mlups": float(row["mlups"]),
                    "l2_norm": float(row["l2_norm"]),
                    "max_amplitude": float(row["max_amplitude"]),
                }
            )
    return rows


def plot(csv_path: str, output_path: str | None = None) -> None:
    rows = read_csv(csv_path)
    integrators = sorted({r["integrator"] for r in rows})
    integrator_label = ", ".join(integrators)

    labels  = [f"{r['nx']}×{r['ny']}" for r in rows]
    mlups   = [r["mlups"]   for r in rows]
    seconds = [r["seconds"] for r in rows]
    cells   = [r["cells"]   for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Wave2D — {integrator_label} — Benchmark Sweep",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(labels, mlups, color="steelblue", edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, mlups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Grid size")
    ax.set_ylabel("MLUPS")
    ax.set_title("Throughput (higher = better)")
    ax.set_ylim(0, max(mlups) * 1.25)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    x = np.arange(len(labels))
    ax.plot(x, seconds, "o-", color="tomato", linewidth=2, markersize=8)
    for xi, val in zip(x, seconds):
        ax.annotate(f"{val:.3f}s", (xi, val),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Time (s)")
    ax.set_title("Wall-clock time (lower = better)")
    ax.set_ylim(0, max(seconds) * 1.25)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(cells, mlups, "s-", color="mediumseagreen", linewidth=2, markersize=8)
    for c, val, lbl in zip(cells, mlups, labels):
        ax.annotate(lbl, (c, val),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Total cells (log₂ scale)")
    ax.set_ylabel("MLUPS")
    ax.set_title("Throughput vs working-set size")
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Wave2D benchmark sweep")
    parser.add_argument("csv", help="Path to benchmark CSV file")
    parser.add_argument("--output", "-o", help="Output image file (default: interactive)")
    args = parser.parse_args()
    plot(args.csv, args.output)
