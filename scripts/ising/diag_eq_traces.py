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
from utils.equilibration import estimate_relaxation_time_two_start
from utils.system import parse_args_compat, setup_logging

#: Exact Onsager critical temperature for the 2D nearest-neighbour Ising model.
TC_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))

#: Color palette for traces.
PALETTE: dict[str, str] = {
    'random': '#4878CF',
    'ordered': '#D65F5F',
    'convergence': '#888888',
}


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

    args = parse_args_compat(parser=parser)
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

    # Select seeds for display: 6 seeds across all temperatures
    seeds_stuck = stuck_seeds[:3]
    seeds_ordering = ordering_seeds[:3]
    seeds_all = sorted(set(seeds_stuck + seeds_ordering))[:6]

    logger.info(f"  Using 6 seeds across all temperatures: {seeds_all}")

    fig, axes = plt.subplots(4, 6, figsize=(20, 13), sharey='row')
    sweeps = np.arange(1, max_steps + 1)
    smooth_window = 120

    # Cache full-length traces so the T=2.00 detail figure below can reuse
    # them instead of re-simulating each (seed, start) pair.
    trace_cache: dict[tuple[float, int, str], np.ndarray] = {}

    for t_idx, (temp, label) in enumerate(zip(temperatures, t_labels, strict=True)):
        for s_idx, seed in enumerate(seeds_all):
            ax = axes[t_idx, s_idx]

            tr = np.asarray(collect_trace(
                size=size,
                temp=temp,
                seed=seed,
                init_state='random',
                n_chunks=n_chunks,
                chunk_size=chunk_size,
            ), dtype=float)
            to = np.asarray(collect_trace(
                size=size,
                temp=temp,
                seed=seed,
                init_state='ordered',
                n_chunks=n_chunks,
                chunk_size=chunk_size,
            ), dtype=float)
            trace_cache[(temp, seed, 'random')] = tr
            trace_cache[(temp, seed, 'ordered')] = to

            # Calculate smoothed traces
            tr_smooth = np.convolve(tr, np.ones(smooth_window) / smooth_window, mode='valid')
            to_smooth = np.convolve(to, np.ones(smooth_window) / smooth_window, mode='valid')

            # Estimate convergence time
            tau_est = estimate_relaxation_time_two_start(
                trace_random=tr, trace_ordered=to,
                smooth_window=smooth_window, dwell_window=smooth_window,
            )

            # Plot raw data with low alpha as background
            ax.plot(
                sweeps, np.abs(tr), color=PALETTE['random'], alpha=0.15, lw=0.5,
            )
            ax.plot(
                sweeps, np.abs(to), color=PALETTE['ordered'], alpha=0.15, lw=0.5,
            )

            # Plot smoothed data
            smooth_start = smooth_window - 1
            ax.plot(
                sweeps[smooth_start:], np.abs(tr_smooth), color=PALETTE['random'],
                lw=1.8, alpha=0.85, label="random start" if (t_idx == 0 and s_idx == 0) else None,
            )
            ax.plot(
                sweeps[smooth_start:], np.abs(to_smooth), color=PALETTE['ordered'],
                lw=1.8, alpha=0.85, label="ordered start" if (t_idx == 0 and s_idx == 0) else None,
            )

            # Mark convergence point if detected
            if tau_est < max_steps:
                ax.axvline(
                    sweeps[tau_est], color=PALETTE['convergence'],
                    linestyle='--', lw=1.2, alpha=0.6,
                )
                ax.text(
                    sweeps[tau_est], 0.97, f"{int(sweeps[tau_est])}",
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=6, color=PALETTE['convergence'],
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7),
                )

            # Row labels (temperatures) on the left
            if s_idx == 0:
                ax.set_ylabel(label, fontsize=10, fontweight='bold')

            # Column labels (seeds) on top
            if t_idx == 0:
                stuck_marker = " ★" if seed in seeds_stuck else ""
                ax.set_title(f"seed {seed}{stuck_marker}", fontsize=9, fontweight='semibold')

            ax.set_xlim(0, max_steps)
            ax.set_ylim(-0.02, 1.05)

            def x_fmt(x: float, _: Any) -> str:
                return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"

            ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_fmt))
            ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            ax.grid(True, linewidth=0.4, alpha=0.25)
            ax.tick_params(labelsize=7)

            # Add legend only to the first subplot
            if t_idx == 0 and s_idx == 0:
                ax.legend(fontsize=7, loc='upper right', framealpha=0.9)

    fig.suptitle(
        rf"Two-start Equilibration Traces  ·  Ising $L={size}$  ·  6 Seeds × 4 Temperatures",
        fontsize=13, y=0.995,
    )

    # Add common x-label for bottom row
    fig.text(0.5, 0.02, 'MC sweep', ha='center', fontsize=10)

    plt.tight_layout(rect=(0, 0.035, 1, 0.985))
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, f"diagnostic_eq_traces_L{size}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization → {outpath}")

    # Second panel: T = 2.00 detailed view in 2×3 grid.
    # Styled after the Two-Start Convergence Example in the notebook:
    # faint raw traces, bold smoothed traces, mutual cross-bands, and tau marker.
    t200_temp = temperatures[1]  # T = 2.00
    t200_label = t_labels[1]

    TOLERANCE_K: float = 1.0
    SIGMA_FLOOR: float = 0.02

    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes2_flat = axes2.flatten()

    for s_idx, seed in enumerate(seeds_all):
        ax = axes2_flat[s_idx]

        # Reuse the traces computed for the overview grid above.
        tr = np.abs(trace_cache[(t200_temp, seed, 'random')])
        to = np.abs(trace_cache[(t200_temp, seed, 'ordered')])

        # Smoothed traces aligned to the right edge of the rolling window
        tr_smooth = np.convolve(tr, np.ones(smooth_window) / smooth_window, mode='valid')
        to_smooth = np.convolve(to, np.ones(smooth_window) / smooth_window, mode='valid')
        n_sm = min(len(tr_smooth), len(to_smooth))
        tr_smooth = tr_smooth[:n_sm]
        to_smooth = to_smooth[:n_sm]
        steps_sm = sweeps[smooth_window - 1: smooth_window - 1 + n_sm]

        # Mutual cross-band criterion: tail statistics from the second half
        half = n_sm // 2
        mu_r = float(tr_smooth[half:].mean())
        mu_o = float(to_smooth[half:].mean())
        sig_r = max(float(tr_smooth[half:].std()), SIGMA_FLOOR)
        sig_o = max(float(to_smooth[half:].std()), SIGMA_FLOOR)
        band_r = TOLERANCE_K * sig_r   # ordered must enter random's band
        band_o = TOLERANCE_K * sig_o   # random must enter ordered's band

        # Estimate convergence time (raw-trace index, already mapped from the
        # smoothed array by the estimator)
        tau_est = estimate_relaxation_time_two_start(
            trace_random=tr, trace_ordered=to,
            smooth_window=smooth_window, dwell_window=smooth_window,
        )

        # Raw traces (faint background)
        ax.plot(
            sweeps, tr, color=PALETTE['random'], alpha=0.2, lw=0.6, label='Random start (raw)',
        )
        ax.plot(
            sweeps, to, color=PALETTE['ordered'], alpha=0.2, lw=0.6, label='Ordered start (raw)',
        )

        # Smoothed traces (bold)
        ax.plot(
            steps_sm, tr_smooth, color=PALETTE['random'], lw=2.0, label='Random start (smoothed)',
        )
        ax.plot(
            steps_sm, to_smooth, color=PALETTE['ordered'], lw=2.0, label='Ordered start (smoothed)',
        )

        # Ordered-start band: random must enter (centred on ordered tail mean)
        ax.axhline(mu_o, color=PALETTE['ordered'], linestyle='--', alpha=0.7,
                   label=f'Ordered tail mean ({mu_o:.3f})')
        ax.axhspan(mu_o - band_o, mu_o + band_o, color=PALETTE['ordered'], alpha=0.12,
                   label=r'Ordered band ($\pm k\sigma_o$)')

        # Random-start band: ordered must enter (centred on random tail mean)
        ax.axhline(mu_r, color=PALETTE['random'], linestyle='--', alpha=0.7,
                   label=f'Random tail mean ({mu_r:.3f})')
        ax.axhspan(mu_r - band_r, mu_r + band_r, color=PALETTE['random'], alpha=0.12,
                   label=r'Random band ($\pm k\sigma_r$)')

        # Convergence marker mapped onto the original sweep axis
        if tau_est < max_steps:
            x_tau = int(sweeps[tau_est])
            ax.axvline(
                x_tau, color='green', linewidth=2, linestyle=':',
                label=rf'$\hat{{\tau}}_\mathrm{{relax}} \approx {x_tau}$',
            )
            ax.text(
                x_tau, 0.97, f"{x_tau}",
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8, color='green',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7),
            )

        stuck_marker = " ★" if seed in seeds_stuck else ""
        ax.set_title(f"seed {seed}{stuck_marker}", fontsize=10, fontweight='semibold')
        ax.set_xlim(0, max_steps)
        ax.set_ylim(0, 1.05)

        def x_fmt(x: float, _: Any) -> str:
            return f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(x_fmt))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)

        if s_idx >= 3:
            ax.set_xlabel('MC Steps', fontsize=9)

        if s_idx == 0:
            ax.set_ylabel(r'Absolute Magnetization $|M|$', fontsize=10)
            ax.legend(loc='lower right', fontsize=7, framealpha=0.9)

    fig2.suptitle(
        rf"Two-start Equilibration Traces  ·  Ising $L={size}$  ·  {t200_label}",
        fontsize=13, y=0.995,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.985))
    outpath2 = os.path.join(args.output_dir, f"diagnostic_eq_traces_T200_L{size}.png")
    fig2.savefig(outpath2, dpi=150, bbox_inches="tight")
    logger.info(f"Saved detailed T=2.00 visualization → {outpath2}")


if __name__ == '__main__':
    main()
