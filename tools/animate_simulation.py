#!/usr/bin/env python3
"""Animate a Wave2D simulation dump (|psi|^2 heatmap).

Usage:
    python3 tools/animate_simulation.py simulation.bin          # interactive
    python3 tools/animate_simulation.py simulation.bin -o wave.mp4   # save video
    python3 tools/animate_simulation.py simulation.bin --frame 42    # single frame PNG
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def read_dump(path: str) -> dict:
    """Parse the binary dump written by Wave2D dump mode."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic[:6] != b"WAVE2D":
            raise ValueError(f"Unrecognised file magic: {magic!r}")

        nx, ny, n_frames, dump_every = struct.unpack("<QQQQ", f.read(32))
        dx, dy, dt, _pad             = struct.unpack("<ffff", f.read(16))

        raw = np.frombuffer(f.read(), dtype=np.float32)

    if raw.size != n_frames * ny * nx:
        raise ValueError(
            f"Expected {n_frames * ny * nx} floats, got {raw.size}. File may be truncated."
        )

    return {
        "nx": nx, "ny": ny,
        "n_frames": n_frames, "dump_every": dump_every,
        "dx": float(dx), "dy": float(dy), "dt": float(dt),
        "frames": raw.reshape(n_frames, ny, nx),
    }


_CMAP = "inferno"
_BG   = "#0d0d0d"


def _make_figure(dump: dict):
    nx, ny = dump["nx"], dump["ny"]
    extent = (0.0, nx * dump["dx"], 0.0, ny * dump["dy"])
    frames = dump["frames"]
    vmax = float(np.percentile(frames, 99.5)) or 1.0

    fig, ax = plt.subplots(figsize=(7, 6.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    im = ax.imshow(
        frames[0], origin="lower", extent=extent,
        cmap=_CMAP, vmin=0.0, vmax=vmax,
        interpolation="bilinear", aspect="auto",
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label("|ψ|²", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.tick_params(colors="white")
    ax.set_xlabel("x", color="white")
    ax.set_ylabel("y", color="white")
    title = ax.set_title("", color="white", pad=10)

    fig.subplots_adjust(top=0.88, bottom=0.09, left=0.10, right=0.92)
    fig.set_layout_engine("none")

    return fig, ax, im, title, vmax


def _frame_title(dump: dict, frame_idx: int) -> str:
    t = frame_idx * dump["dump_every"] * dump["dt"]
    return (
        f"|ψ|²   t = {t:.5f}   "
        f"(frame {frame_idx}/{dump['n_frames'] - 1})"
    )


def _save_video(dump: dict, fig, im, title, output_path: str, fps: int) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    frames_data = dump["frames"]
    n = len(frames_data)

    fig.set_dpi(150)
    canvas = FigureCanvasAgg(fig)

    im.set_data(frames_data[0])
    title.set_text(_frame_title(dump, 0))
    canvas.draw()
    w, h = canvas.get_width_height()
    w_out, h_out = (w // 2) * 2, (h // 2) * 2

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgba",
        "-framerate", str(fps), "-i", "pipe:",
        "-vf", f"scale={w_out}:{h_out}",
        "-vcodec", "h264", "-pix_fmt", "yuv420p", "-b:v", "3000k",
        output_path,
    ]

    print(f"Rendering {n} frames to {output_path} …")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        proc.stdin.write(bytes(canvas.buffer_rgba()))
        for i in range(1, n):
            im.set_data(frames_data[i])
            title.set_text(_frame_title(dump, i))
            canvas.draw()
            proc.stdin.write(bytes(canvas.buffer_rgba()))
    finally:
        proc.stdin.close()
        proc.wait()
    print(f"Saved: {output_path}")


def animate(dump: dict, output_path: str | None, fps: int, interval_ms: int) -> None:
    fig, _, im, title, _ = _make_figure(dump)
    frames = dump["frames"]

    if output_path:
        _save_video(dump, fig, im, title, output_path, fps)
        plt.close(fig)
        return

    def update(i):
        im.set_data(frames[i])
        title.set_text(_frame_title(dump, i))
        return []

    animation.FuncAnimation(
        fig, update, frames=dump["n_frames"],
        interval=interval_ms, blit=False,
    )
    plt.show()
    plt.close(fig)


def save_frame(dump: dict, frame_idx: int, output_path: str) -> None:
    if frame_idx < 0 or frame_idx >= dump["n_frames"]:
        raise ValueError(f"Frame index {frame_idx} out of range [0, {dump['n_frames'] - 1}]")

    fig, _, im, title, _ = _make_figure(dump)
    im.set_data(dump["frames"][frame_idx])
    title.set_text(_frame_title(dump, frame_idx))
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"Saved frame {frame_idx} to {output_path}")


def print_info(dump: dict) -> None:
    total_time = (dump["n_frames"] - 1) * dump["dump_every"] * dump["dt"]
    size_mb = dump["frames"].nbytes / (1024 ** 2)
    print(
        f"Grid:        {dump['nx']} x {dump['ny']}\n"
        f"Frames:      {dump['n_frames']}\n"
        f"Dump every:  {dump['dump_every']} steps\n"
        f"dt:          {dump['dt']}\n"
        f"Time span:   0 .. {total_time:.4f}\n"
        f"Data:        {size_mb:.1f} MB in memory\n"
        f"|psi|^2 max: {dump['frames'].max():.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Animate or inspect a Wave2D binary simulation dump"
    )
    parser.add_argument("dump", help="Path to .bin dump file")
    parser.add_argument("-o", "--output",
                        help="Output file: .mp4/.gif for video, .png for single frame")
    parser.add_argument("--frame", type=int, default=None,
                        help="Export a single frame as PNG (requires --output)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for video output (default: 30)")
    parser.add_argument("--interval", type=int, default=40,
                        help="Frame interval in ms for interactive mode (default: 40)")
    parser.add_argument("--info", action="store_true",
                        help="Print file metadata and exit")
    args = parser.parse_args()

    dump = read_dump(args.dump)

    if args.info:
        print_info(dump)
        sys.exit(0)

    if args.frame is not None:
        out = args.output or f"frame_{args.frame:04d}.png"
        save_frame(dump, args.frame, out)
    else:
        animate(dump, args.output, args.fps, args.interval)
