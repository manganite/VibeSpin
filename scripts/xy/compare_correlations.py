"""
Comparison of spin-spin correlation functions G(r) for the XY model.
Contrasts power-law decay (low T) with exponential decay (high T).
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.xy_model import XYSimulation
from scripts._cli import parse_args_compat
from utils.physics_helpers import get_averaged_correlation
from utils.system_helpers import ensure_results_dir, parallel_sweep, save_plot, setup_logging


def simulate_correlation(params: tuple[float, int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Worker function: simulate and return the averaged correlation function at temperature T.

    Parameters
    ----------
        params: Tuple of (T, L, steps, eq_steps, sample_interval).

    Returns
    -------
        A tuple of (r, G_r) - radial distances and averaged correlations.
    """
    T, L, steps, eq_steps, sample_interval = params
    logger = logging.getLogger('vibespin')
    logger.debug(f'Collecting data for T={T}...')
    sim = XYSimulation(size=L, temp=T)
    sim.equilibrate(n_steps=eq_steps)
    return get_averaged_correlation(sim=sim, total_steps=steps, sample_interval=sample_interval)


def main() -> None:
    """Run the correlation comparison analysis for the XY model."""
    parser = argparse.ArgumentParser(description='2D XY Model Correlation Comparison')
    parser.add_argument('--size', type=int, default=50, help='Linear lattice size L')
    parser.add_argument('--steps', type=int, default=10000, help='Measurement steps')
    parser.add_argument('--eq-steps', type=int, default=2000, help='Equilibration steps')
    parser.add_argument('--interval', type=int, default=20, help='Sample interval')
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Temperatures
    T_LOW: float = 0.4  # Well below BKT (Power law expected)
    T_HIGH: float = 1.5  # Well above BKT (Exponential expected)

    logger.info(f'Starting XY correlation comparison (L={args.size})...')
    temperatures = [T_LOW, T_HIGH]
    sweep_params = [(T, args.size, args.steps, args.eq_steps, args.interval) for T in temperatures]

    results = parallel_sweep(worker_func=simulate_correlation, params=sweep_params)
    (r_low, G_low), (r_high, G_high) = results

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Log-Log Plot (Best for Power Law / Low T)
    # We skip r=0 to avoid log(0)
    ax1.loglog(r_low[1:], G_low[1:], 'o-', label=f'T={T_LOW} (Low Temp)')
    ax1.loglog(r_high[1:], G_high[1:], 'x-', label=f'T={T_HIGH} (High Temp)')
    ax1.set_title('Log-Log Plot\n(Straight line = Power Law Decay)')
    ax1.set_xlabel('Distance r')
    ax1.set_ylabel('Correlation G(r)')
    ax1.legend()
    ax1.grid(True, which='both', ls='-', alpha=0.5)

    # 2. Semi-Log Plot (Best for Exponential / High T)
    ax2.plot(r_low, G_low, 'o-', label=f'T={T_LOW} (Low Temp)')
    ax2.plot(r_high, G_high, 'x-', label=f'T={T_HIGH} (High Temp)')
    ax2.set_yscale('log')
    ax2.set_title('Semi-Log Plot\n(Straight line = Exponential Decay)')
    ax2.set_xlabel('Distance r')
    ax2.set_ylabel('Correlation G(r)')
    ax2.legend()
    ax2.grid(True, which='both', ls='-', alpha=0.5)

    output_dir: str = ensure_results_dir(directory=args.output_dir)
    save_plot(filename='correlation_comparison.png', directory=output_dir)


if __name__ == '__main__':
    main()
