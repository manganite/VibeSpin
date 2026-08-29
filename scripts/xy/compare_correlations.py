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
from utils.equilibration import convergence_equilibrate_with_status
from utils.observables import get_averaged_correlation
from utils.plotting import ensure_results_dir, save_plot
from utils.system import parse_args_compat, setup_logging


def simulate_correlation(
    *,
    T: float,
    L: int,
    steps: int,
    eq_probe: int,
    eq_max: int,
    sample_interval: int,
    seed: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrate and measure the averaged correlation function at temperature T.

    Uses two-start convergence equilibration to avoid initialization bias.

    Parameters
    ----------
    T : float
        Temperature for the measurement.
    L : int
        Linear lattice size.
    steps : int
        Measurement steps after equilibration.
    eq_probe : int
        Chunk size for convergence equilibration probes.
    eq_max : int
        Maximum equilibration steps.
    sample_interval : int
        Spacing between correlation samples during measurement.
    seed : int
        Random seed for reproducibility.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Radial distances r and averaged correlations G(r).
    """
    logger.debug(f'Equilibrating at T={T:.3f} (L={L}, seed={seed})...')
    sim_r = XYSimulation(
        size=L, temp=T, update='checkerboard', init_state='random', seed=seed,
    )
    sim_o = XYSimulation(
        size=L, temp=T, update='checkerboard', init_state='ordered', seed=seed,
    )
    _, converged = convergence_equilibrate_with_status(
        sim_random=sim_r, sim_ordered=sim_o, chunk_size=eq_probe, max_steps=eq_max,
    )
    sim_meas = sim_r if converged else sim_o
    if not converged:
        logger.info(f'T={T:.3f}: convergence not reached, falling back to ordered start')
    logger.debug(f'Measuring correlations at T={T:.3f}...')
    return get_averaged_correlation(
        sim=sim_meas, total_steps=steps, sample_interval=sample_interval,
    )


def main() -> None:
    """Run the correlation comparison analysis for the XY model."""
    parser = argparse.ArgumentParser(description='2D XY Model Correlation Comparison')
    parser.add_argument('--size', type=int, default=128, help='Linear lattice size L')
    parser.add_argument('--steps', type=int, default=10000, help='Measurement steps')
    parser.add_argument('--eq-probe', type=int, default=200, help='Convergence probe chunk size')
    parser.add_argument('--eq-max', type=int, default=50000, help='Max equilibration steps')
    parser.add_argument('--interval', type=int, default=20, help='Sample interval')
    parser.add_argument('--seed', type=int, default=510, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Temperatures
    T_LOW: float = 0.4  # Well below BKT (Power law expected)
    T_HIGH: float = 1.5  # Well above BKT (Exponential expected)

    logger.info(f'Starting XY correlation comparison (L={args.size})...')

    results = {}
    for label, T in [('low', T_LOW), ('high', T_HIGH)]:
        r, G = simulate_correlation(
            T=T, L=args.size, steps=args.steps, eq_probe=args.eq_probe,
            eq_max=args.eq_max, sample_interval=args.interval, seed=args.seed,
            logger=logger,
        )
        results[label] = (r, G)

    r_low, G_low = results['low']
    r_high, G_high = results['high']

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

    # Save data for notebook consumption
    npz_path = f'{output_dir}/correlation_comparison.npz'
    np.savez_compressed(
        npz_path,
        r_low=r_low,
        G_low=G_low,
        r_high=r_high,
        G_high=G_high,
        T_low=T_LOW,
        T_high=T_HIGH,
        L=args.size,
        steps=args.steps,
        eq_probe=args.eq_probe,
        eq_max=args.eq_max,
        sample_interval=args.interval,
        seed=args.seed,
    )
    logger.info(f'Data saved to {npz_path}')


if __name__ == '__main__':
    main()
