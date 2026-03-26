"""
Comparison of spin-spin correlation functions for the 2D Ising model.
Analyzes correlation behavior in ferromagnetic, critical, and paramagnetic phases.
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.ising_model import IsingSimulation
from utils.analysis import get_averaged_correlation
from utils.plotting import ensure_results_dir, save_plot
from utils.system import parallel_sweep, parse_args_compat, setup_logging


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
    sim = IsingSimulation(size=L, temp=T)
    sim.equilibrate(n_steps=eq_steps)
    return get_averaged_correlation(sim=sim, total_steps=steps, sample_interval=sample_interval)


def main() -> None:
    """Run the correlation comparison analysis."""
    parser = argparse.ArgumentParser(description='2D Ising Model Correlation Comparison')
    parser.add_argument('--size', type=int, default=64, help='Linear lattice size L')
    parser.add_argument('--steps', type=int, default=10000, help='Measurement steps')
    parser.add_argument('--eq-steps', type=int, default=2000, help='Equilibration steps')
    parser.add_argument('--interval', type=int, default=20, help='Sample interval')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

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
    sweep_params = [(T, args.size, args.steps, args.eq_steps, args.interval) for T in temperatures]

    results = parallel_sweep(worker_func=simulate_correlation, params=sweep_params)
    (r, G_ferro), (_, G_crit), (_, G_para) = results

    # --- Fit for correlation length xi in paramagnetic phase ---
    r_fit: np.ndarray = r[FIT_START_R:FIT_END_R]
    G_para_fit: np.ndarray = G_para[FIT_START_R:FIT_END_R]

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
                xi: float = -1.0 / slope
                logger.info(
                    f'Fitted correlation length for T={T_PARA} '
                    f'(paramagnetic): xi = {xi:.4f}'
                )
                fit_line: np.ndarray = np.exp(intercept + slope * r)
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
    if 'xi' in locals():
        ax2.plot(r, fit_line, 'r--', linewidth=2, label=f'Fit ($\\xi={xi:.2f}$)')
    ax2.set_yscale('log')
    ax2.set_title('Semi-Log Plot')
    ax2.set_xlabel('Distance r')
    ax2.set_ylabel('Correlation G(r)')
    ax2.legend()
    ax2.grid(True, which='both', ls='-', alpha=0.5)

    output_dir: str = ensure_results_dir(directory=args.output_dir)
    save_plot(filename='correlation_comparison.png', directory=output_dir)


if __name__ == '__main__':
    main()
