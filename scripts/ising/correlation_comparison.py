"""
Comparison of spin-spin correlation functions for the 2D Ising model.
Analyzes correlation behavior in ferromagnetic, critical, and paramagnetic phases.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from models.ising_model import IsingSimulation
from utils.observables import (
    CorrelationPoint,
    fit_correlation_exponent,
    fit_correlation_length,
    measure_correlation_point,
)
from utils.plotting import ensure_results_dir, save_plot
from utils.system import parallel_sweep, parse_args_compat, setup_logging


def main() -> None:
    """Run the correlation comparison analysis."""
    parser = argparse.ArgumentParser(description='2D Ising Model Correlation Comparison')
    parser.add_argument('--size', type=int, default=64, help='Linear lattice size L')
    parser.add_argument('--steps', type=int, default=10000, help='Measurement steps')
    parser.add_argument('--eq-probe', type=int, default=200, help='Convergence probe chunk size')
    parser.add_argument('--eq-max', type=int, default=50000, help='Max equilibration steps')
    parser.add_argument('--interval', type=int, default=20, help='Sample interval')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Ising 2D Critical Temperature approx 2.269
    T_FERRO: float = 1.8  # Below Tc (Long range order)
    T_CRIT: float = 2.269  # At Tc (Power law decay)
    T_PARA: float = 3.0  # Above Tc (Exponential decay)

    logger.info(f'Starting Ising correlation comparison (L={args.size})...')
    points = [
        CorrelationPoint(
            label=label, temperature=T, model_cls=IsingSimulation, model_kwargs={},
            size=args.size, seed=args.seed, eq_probe=args.eq_probe,
            eq_max=args.eq_max, meas_steps=args.steps, interval=args.interval,
        )
        for label, T in (('ferro', T_FERRO), ('crit', T_CRIT), ('para', T_PARA))
    ]
    results = {
        label: (r, G)
        for label, r, G in parallel_sweep(
            worker_func=measure_correlation_point, params=points,
        )
    }
    r, G_ferro = results['ferro']
    _, G_crit = results['crit']
    _, G_para = results['para']

    # At T_c the Ising correlation function decays as a power law with the
    # exactly known exponent eta = 1/4; above T_c it decays exponentially with
    # a finite correlation length. Each temperature is fitted with the form it
    # is expected to take, over distances up to L/4 so that periodic images do
    # not flatten the tail.
    eta_crit = fit_correlation_exponent(r=r, G=G_crit)
    xi_para = fit_correlation_length(r=r, G=G_para)
    logger.info(f'T={T_CRIT}: fitted eta = {eta_crit:.4f} (exact value 0.25)')
    logger.info(f'T={T_PARA}: fitted correlation length xi = {xi_para:.4f}')

    # Anchor the drawn guide to the first fitted point, so that the line sits
    # on the data rather than on an extrapolated intercept.
    _FIT_ANCHOR = 2
    fit_line: np.ndarray | None = None
    if np.isfinite(xi_para):
        fit_line = G_para[_FIT_ANCHOR] * np.exp(-(r - r[_FIT_ANCHOR]) / xi_para)
    else:
        logger.warning(f'Exponential fit failed for T={T_PARA}; no correlation length reported.')

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Log-Log Plot (Best for Power Law / Critical)
    ax1.loglog(r[1:], G_ferro[1:], 's-', label=f'T={T_FERRO} (T < Tc)', alpha=0.7)
    ax1.loglog(r[1:], G_crit[1:], 'o-', label=f'T={T_CRIT} (T ~ Tc)', alpha=0.7)
    ax1.loglog(r[1:], G_para[1:], 'x-', label=f'T={T_PARA} (T > Tc)', alpha=0.7)
    ax1.set_title('Log-Log Plot')
    ax1.set_xlabel('Distance r')
    ax1.set_ylabel('Correlation G(r)')
    ax1.legend()
    ax1.grid(True, which='both', ls='-', alpha=0.5)

    # 2. Semi-Log Plot (Best for Exponential / High T)
    ax2.plot(r, G_ferro, 's-', label=f'T={T_FERRO} (T < Tc)', alpha=0.7)
    ax2.plot(r, G_crit, 'o-', label=f'T={T_CRIT} (T ~ Tc)', alpha=0.7)
    ax2.plot(r, G_para, 'x-', label=f'T={T_PARA} (T > Tc)', alpha=0.7)
    if fit_line is not None:
        ax2.plot(r, fit_line, 'r--', linewidth=2, label=f'Fit ($\\xi={xi_para:.2f}$)')
    ax2.set_yscale('log')
    ax2.set_title('Semi-Log Plot')
    ax2.set_xlabel('Distance r')
    ax2.set_ylabel('Correlation G(r)')
    ax2.legend()
    ax2.grid(True, which='both', ls='-', alpha=0.5)

    output_dir: str = ensure_results_dir(directory=args.output_dir)
    save_plot(filename='correlation_comparison.png', directory=output_dir)

    # Save data for notebook consumption
    npz_path = f'{output_dir}/correlation_comparison.npz'
    save_kwargs: dict[str, Any] = dict(
        r=r,
        G_ferro=G_ferro,
        G_crit=G_crit,
        G_para=G_para,
        T_ferro=T_FERRO,
        T_crit=T_CRIT,
        T_para=T_PARA,
        L=args.size,
        steps=args.steps,
        eq_probe=args.eq_probe,
        eq_max=args.eq_max,
        sample_interval=args.interval,
        seed=args.seed,
    )
    save_kwargs['xi_para'] = xi_para
    save_kwargs['eta_crit'] = eta_crit
    np.savez_compressed(npz_path, **save_kwargs)
    logger.info(f'Data saved to {npz_path}')


if __name__ == '__main__':
    main()
