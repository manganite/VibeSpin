"""
Diagnostic tool for visualizing two-start equilibration traces.

This script runs simultaneous random-start and ordered-start simulations of the
2D Ising model at multiple representative temperatures (deep ordered, critical,
and paramagnetic). It identifies seeds that lead to metastable stuck states at
low temperatures and generates a multi-panel visualization of the magnetization
trajectories.

Results are saved to ``results/ising/diagnostic_eq_traces_L<size>.png``.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from models.ising_model import IsingSimulation as IsingModel
from utils.cli_helpers import parse_args_compat
from utils.system_helpers import setup_logging

#: Exact Onsager critical temperature for the 2D nearest-neighbour Ising model.
TC_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))

#: Colors used for different seed trajectories.
SEED_COLORS: list[str] = ["#2166ac", "#d6604d", "#4dac26", "#984ea3", "#ff7f00", "#a65628"]


def collect_trace(
    *,
    size: int,
    temp: float,
    seed: int,
    init_state: str,
    n_chunks: int,
    chunk_size: int,
) -> np.ndarray:
    """
    Run a simulation and collect the magnetization trace.

    Parameters
    ----------
    size : int
        Linear lattice size L.
    temp : float
        Temperature T.
    seed : int
        RNG seed.
    init_state : str
        Initial state ('random' or 'ordered').
    n_chunks : int
        Number of measurement chunks.
    chunk_size : int
        Sweeps per chunk.

    Returns
    -------
    np.ndarray
        Array of magnetization values at each sweep.
    """
    sim = IsingModel(
        size=size,
        temp=temp,
        init_state=init_state,
        update='checkerboard',
        seed=seed,
    )
    mags: list[float] = []
    for _ in range(n_chunks):
        m, _ = sim.run(n_steps=chunk_size)
        mags.extend(m)
    return np.array(mags)


def main() -> None:
    """
    Execute the equilibration diagnostic and generate plots.
    """
    parser = argparse.ArgumentParser(
        description='Generate two-start equilibration traces for the 2D Ising model.',
    )
    parser.add_argument('--size', type=int, default=64, help='Linear lattice size L')
    parser.add_argument('--max-steps', type=int, default=5000, help='Total MC sweeps to record')
    parser.add_argument('--chunk-size', type=int, default=50, help='Sweeps per measurement chunk')
    parser.add_argument(
        '--scan-seeds', type=int, default=25,
        help='Number of seeds to pre-screen for stuck states at low T',
    )
    parser.add_argument(
        '--scan-steps', type=int, default=500,
        help='Pre-screen length for identifying stuck states',
    )
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)
    logger = setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    size = args.size
    max_steps = args.max_steps
    chunk_size = args.chunk_size
    n_chunks = max_steps // chunk_size

    temperatures = [0.5, 2.0, TC_ISING, 2.5]
    t_labels = [
        r"$T = 0.5$ (deep ordered)",
        r"$T = 2.00$ (below $T_c$)",
        rf"$T = T_c \approx {TC_ISING:.3f}$",
        r"$T = 2.50$ (above $T_c$)",
    ]

    # Pre-screen seeds at low T to find interesting trajectories
    st_temp = temperatures[0]
    logger.info(
        f"Scanning {args.scan_seeds} seeds at T={st_temp} L={size} for stuck vs. ordering..."
    )

    stuck_seeds: list[int] = []
    ordering_seeds: list[int] = []
    scan_chunks = args.scan_steps // chunk_size

    for s in range(args.scan_seeds):
        tr = collect_trace(
            size=size,
            temp=st_temp,
            seed=s,
            init_state='random',
            n_chunks=scan_chunks,
            chunk_size=chunk_size,
        )
        final_m = float(np.abs(tr[-50:]).mean())
        if final_m < 0.5:
            stuck_seeds.append(s)
        else:
            ordering_seeds.append(s)

    logger.info(f"  Stuck seeds (|M|<0.5 after {args.scan_steps} sweeps): {stuck_seeds[:6]}")
    logger.info(f"  Ordering seeds: {ordering_seeds[:6]}")

    # Select seeds for display: up to 3 stuck and 3 ordering
    seeds_stuck = stuck_seeds[:3]
    seeds_ordering = ordering_seeds[:3]
    seeds_low_t = sorted(set(seeds_stuck + seeds_ordering))
    seeds_other = [1, 2, 3]

    logger.info(f"  Using seeds for T=0.5 panel: stuck={seeds_stuck}, ordering={seeds_ordering}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flat
    sweeps = np.arange(1, max_steps + 1)

    for ax, temp, label in zip(axes_flat, temperatures, t_labels, strict=True):
        seeds_to_use = seeds_low_t if temp == st_temp else seeds_other
        for idx, seed in enumerate(seeds_to_use):
            color = SEED_COLORS[idx % len(SEED_COLORS)]
            tr = collect_trace(
                size=size,
                temp=temp,
                seed=seed,
                init_state='random',
                n_chunks=n_chunks,
                chunk_size=chunk_size,
            )
            to = collect_trace(
                size=size,
                temp=temp,
                seed=seed,
                init_state='ordered',
                n_chunks=n_chunks,
                chunk_size=chunk_size,
            )

            stuck_tag = " ★stuck" if (temp == st_temp and seed in seeds_stuck) else ""
            ax.plot(
                sweeps, np.abs(tr), color=color, lw=0.7, alpha=0.65,
                label=f"seed {seed}{stuck_tag} (rand)" if temp == st_temp else None,
            )
            ax.plot(
                sweeps, to, color=color, lw=0.7, alpha=0.65, linestyle='--',
                label=f"seed {seed}{stuck_tag} (ord)" if temp == st_temp else None,
            )

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("MC sweep", fontsize=9)
        ax.set_ylabel(r"$|M|$", fontsize=10)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, max_steps)

        def x_fmt(x: float, _: Any) -> str:
            return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_fmt))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(True, linewidth=0.4, alpha=0.5)

    # Global legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc='lower center', ncol=6, fontsize=8,
        title="solid = random start  |  dashed = ordered start",
        title_fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        rf"Two-start equilibration traces  ·  Ising $L={size}$  ·  Representative Seeds",
        fontsize=13, y=1.01,
    )

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, f"diagnostic_eq_traces_L{size}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization → {outpath}")


if __name__ == '__main__':
    main()
