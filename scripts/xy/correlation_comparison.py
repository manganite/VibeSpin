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
from utils.observables import (
    CorrelationPoint,
    fit_correlation_exponent,
    fit_correlation_length,
    measure_correlation_point,
)
from utils.plotting import ensure_results_dir, save_plot
from utils.system import parallel_sweep, parse_args_compat, setup_logging


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

    points = [
        CorrelationPoint(
            label=label, temperature=T, model_cls=XYSimulation, model_kwargs={},
            size=args.size, seed=args.seed, eq_probe=args.eq_probe,
            eq_max=args.eq_max, meas_steps=args.steps, interval=args.interval,
        )
        for label, T in (('low', T_LOW), ('high', T_HIGH))
    ]
    results = {
        label: (r, G)
        for label, r, G in parallel_sweep(
            worker_func=measure_correlation_point, params=points,
        )
    }

    r_low, G_low = results['low']
    r_high, G_high = results['high']

    # Below the transition the XY model is quasi-ordered, so the expected form
    # is a power law whose exponent spin-wave theory fixes at eta = T/(2 pi J);
    # above it correlations decay exponentially with a finite length.
    eta_low = fit_correlation_exponent(r=r_low, G=G_low)
    xi_high = fit_correlation_length(r=r_high, G=G_high)
    eta_spin_wave = T_LOW / (2.0 * np.pi)
    logger.info(
        f'T={T_LOW}: fitted eta = {eta_low:.4f} '
        f'(spin-wave prediction {eta_spin_wave:.4f})'
    )
    logger.info(f'T={T_HIGH}: fitted correlation length xi = {xi_high:.4f}')

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
        eta_low=eta_low,
        eta_low_spin_wave=eta_spin_wave,
        xi_high=xi_high,
    )
    logger.info(f'Data saved to {npz_path}')


if __name__ == '__main__':
    main()
