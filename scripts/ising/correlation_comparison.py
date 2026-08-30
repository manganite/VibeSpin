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
from utils.observables import simulate_equilibrium_correlation
from utils.plotting import ensure_results_dir, save_plot
from utils.system import parallel_sweep, parse_args_compat, setup_logging


def simulate_correlation(
    params: tuple[float, int, int, int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Worker function: simulate and return the averaged correlation function at temperature T.

    Uses two-start convergence equilibration to avoid initialization bias.

    Parameters
    ----------
    params : tuple[float, int, int, int, int, int, int]
        Tuple of (T, L, steps, eq_probe, eq_max, sample_interval, seed).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple of (r, G_r) - radial distances and averaged correlations.
    """
    T, L, steps, eq_probe, eq_max, sample_interval, seed = params
    logger = logging.getLogger('vibespin')
    logger.debug(f'Collecting data for T={T}...')
    return simulate_equilibrium_correlation(
        model_cls=IsingSimulation, model_kwargs={}, size=L, temp=T, seed=seed,
        eq_probe=eq_probe, eq_max=eq_max, meas_steps=steps, interval=sample_interval,
        logger=logger,
    )


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

    # Fitting Parameters
    FIT_START_R: int = 2
    FIT_END_R: int = 15

    logger.info(f'Starting Ising correlation comparison (L={args.size})...')
    temperatures = [T_FERRO, T_CRIT, T_PARA]
    sweep_params = [
        (T, args.size, args.steps, args.eq_probe, args.eq_max, args.interval, args.seed)
        for T in temperatures
    ]

    results = parallel_sweep(worker_func=simulate_correlation, params=sweep_params)
    (r, G_ferro), (_, G_crit), (_, G_para) = results

    # --- Fit for correlation length xi in paramagnetic phase ---
    r_fit: np.ndarray = r[FIT_START_R:FIT_END_R]
    G_para_fit: np.ndarray = G_para[FIT_START_R:FIT_END_R]

    xi_para: float | None = None
    fit_line: np.ndarray | None = None

    # Ensure we only fit positive values to avoid log(0) errors
    valid_indices: np.ndarray = G_para_fit > 1e-10
    if np.count_nonzero(valid_indices) >= 2:
        log_G_para_fit: np.ndarray = np.log(G_para_fit[valid_indices])
        r_fit_valid: np.ndarray = r_fit[valid_indices]

        try:
            slope, intercept = np.polyfit(r_fit_valid, log_G_para_fit, 1)
            if slope == 0.0:
                logger.warning(
                    f'Exponential fit failed for T={T_PARA}: fitted slope is zero; '
                    'cannot compute correlation length.'
                )
            else:
                xi_para = -1.0 / slope
                logger.info(
                    f'Fitted correlation length for T={T_PARA} '
                    f'(paramagnetic): xi = {xi_para:.4f}'
                )
                fit_line = np.exp(intercept + slope * r)
        except np.linalg.LinAlgError as exc:
            logger.warning(f'Exponential fit failed for T={T_PARA}: {exc}')

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
    if xi_para is not None and fit_line is not None:
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
    if xi_para is not None:
        save_kwargs['xi_para'] = xi_para
    np.savez_compressed(npz_path, **save_kwargs)
    logger.info(f'Data saved to {npz_path}')


if __name__ == '__main__':
    main()
