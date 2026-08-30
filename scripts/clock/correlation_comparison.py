"""
Comparison of spin-spin correlation functions G(r) for the q-state Clock model.
Analyzes correlation behavior in ordered, quasi-ordered, and disordered phases.

The q=6 clock model has two Kosterlitz-Thouless transitions at T1 ≈ 0.68
and T2 ≈ 0.92 (José et al. 1977), yielding three distinct correlation regimes:
long-range order below T1, algebraic (quasi-long-range) order between T1 and T2,
and exponential decay above T2.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from models.clock_model import ClockSimulation
from utils.observables import simulate_equilibrium_correlation
from utils.plotting import ensure_results_dir, save_plot
from utils.system import parse_args_compat, setup_logging

# Approximate KT transition temperatures for q=6 (José et al. 1977).
T1_CLOCK6: float = 0.68
T2_CLOCK6: float = 0.92


def main() -> None:
    """Run the clock model correlation comparison analysis."""
    parser = argparse.ArgumentParser(
        description='q-state Clock Model Correlation Comparison',
    )
    parser.add_argument('--size', type=int, default=128, help='Linear lattice size L')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    parser.add_argument('--steps', type=int, default=4000, help='Measurement steps')
    parser.add_argument('--eq-probe', type=int, default=200, help='Convergence probe chunk size')
    parser.add_argument('--eq-max', type=int, default=50000, help='Max equilibration steps')
    parser.add_argument('--interval', type=int, default=20, help='Sample interval')
    parser.add_argument(
        '--seed', type=int, default=520,
        help='Random seed shared by all three temperature points',
    )
    parser.add_argument('--output-dir', type=str, default='results/clock', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Three representative temperatures spanning the three phases.
    T_ORDERED: float = 0.5   # T < T1  (long-range order)
    T_QUASI: float = 0.8     # T1 < T < T2  (algebraic quasi-order)
    T_DISORDERED: float = 1.2  # T > T2  (exponential decay)

    logger.info(
        f'Starting clock correlation comparison '
        f'(q={args.q}, L={args.size}, steps={args.steps})'
    )
    logger.info(
        f'Temperatures: ordered={T_ORDERED}, quasi={T_QUASI}, disordered={T_DISORDERED}'
    )

    common: dict[str, Any] = dict(
        model_cls=ClockSimulation, model_kwargs={'q': args.q}, size=args.size,
        seed=args.seed, eq_probe=args.eq_probe, eq_max=args.eq_max,
        meas_steps=args.steps, interval=args.interval, logger=logger,
    )

    r_ordered, G_ordered = simulate_equilibrium_correlation(temp=T_ORDERED, **common)
    r_quasi, G_quasi = simulate_equilibrium_correlation(temp=T_QUASI, **common)
    r_disordered, G_disordered = simulate_equilibrium_correlation(temp=T_DISORDERED, **common)

    # Verify r arrays are identical (all from same lattice size).
    if not (np.array_equal(r_ordered, r_quasi) and np.array_equal(r_quasi, r_disordered)):
        raise RuntimeError(
            'Radial distance arrays differ between temperature points; '
            'all three measurements must share the same lattice size.'
        )
    r = r_ordered

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.loglog(r[1:], G_ordered[1:], 'o-', label=f'T={T_ORDERED} (ordered, T < T₁)', alpha=0.7)
    ax1.loglog(r[1:], G_quasi[1:], 's-', label=f'T={T_QUASI} (quasi-ordered)', alpha=0.7)
    ax1.loglog(
        r[1:], G_disordered[1:], 'x-',
        label=f'T={T_DISORDERED} (disordered, T > T\u2082)', alpha=0.7,
    )
    ax1.set_title('Log-Log Plot')
    ax1.set_xlabel('Distance r')
    ax1.set_ylabel('Correlation G(r)')
    ax1.legend()
    ax1.grid(True, which='both', ls='-', alpha=0.5)

    ax2.plot(r, G_ordered, 'o-', label=f'T={T_ORDERED} (ordered)', alpha=0.7)
    ax2.plot(r, G_quasi, 's-', label=f'T={T_QUASI} (quasi-ordered)', alpha=0.7)
    ax2.plot(r, G_disordered, 'x-', label=f'T={T_DISORDERED} (disordered)', alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_title('Semi-Log Plot')
    ax2.set_xlabel('Distance r')
    ax2.set_ylabel('Correlation G(r)')
    ax2.legend()
    ax2.grid(True, which='both', ls='-', alpha=0.5)

    fig.suptitle(f'{args.q}-state Clock Model: Correlation Comparison (L={args.size})')

    output_dir: str = ensure_results_dir(directory=args.output_dir)
    save_plot(filename='correlation_comparison.png', directory=output_dir)

    # Save data for notebook consumption.
    npz_path = f'{output_dir}/correlation_comparison.npz'
    np.savez_compressed(
        npz_path,
        r=r,
        G_ordered=G_ordered,
        G_quasi=G_quasi,
        G_disordered=G_disordered,
        T_ordered=T_ORDERED,
        T_quasi=T_QUASI,
        T_disordered=T_DISORDERED,
        T1=T1_CLOCK6,
        T2=T2_CLOCK6,
        L=args.size,
        q=args.q,
        steps=args.steps,
        eq_probe=args.eq_probe,
        eq_max=args.eq_max,
        sample_interval=args.interval,
        seed=args.seed,
    )
    logger.info(f'Data saved to {npz_path}')


if __name__ == '__main__':
    main()
