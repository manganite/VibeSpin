"""
Analysis of correlation length divergence in the 2D Ising model.
Extracts the critical exponent nu by fitting correlation lengths near Tc.
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.ising_model import IsingSimulation
from utils.physics_helpers import get_averaged_correlation
from utils.system_helpers import ensure_results_dir, parallel_sweep, save_plot, setup_logging


def get_correlation_length(params: tuple[float, int, int, int, int]) -> tuple[float, float]:
    """Simulate and extract correlation length xi for a given temperature.

    Args:
        params: Tuple of (T, L, steps, eq_steps, sample_interval).

    Returns:
        A tuple of (T, xi).
    """
    T, L, steps, eq_steps, sample_interval = params
    logger = logging.getLogger('multiferroic')
    logger.debug(f'Calculating xi for T={T}...')
    sim = IsingSimulation(L, T)
    sim.equilibrate(eq_steps)

    r, G_r = get_averaged_correlation(sim, steps, sample_interval)

    # Filter for valid range
    # r > 1 to avoid short-range lattice effects
    # r < L/4 to avoid finite size effects / periodic boundary artifacts
    # G_r > noise floor
    mask: np.ndarray = (r > 1) & (r < L // 4) & (G_r > 1e-4)

    if np.sum(mask) < 3:
        return T, np.nan

    r_fit: np.ndarray = r[mask]
    log_G: np.ndarray = np.log(G_r[mask])

    try:
        slope, intercept = np.polyfit(r_fit, log_G, 1)
        xi: float = -1.0 / slope
    except np.linalg.LinAlgError, ValueError, ZeroDivisionError:
        xi = np.nan

    return T, xi


def run_divergence_analysis() -> None:
    """Run parallel simulation to extract the critical exponent nu from xi(T) divergence."""
    parser = argparse.ArgumentParser(description='2D Ising Model Correlation Divergence Analysis')
    parser.add_argument('--size', type=int, default=128, help='Linear lattice size L')
    parser.add_argument('--steps', type=int, default=50000, help='Measurement steps')
    parser.add_argument('--eq-steps', type=int, default=10000, help='Equilibration steps')
    parser.add_argument('--interval', type=int, default=20, help='Sample interval')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Physical Constants
    TC_THEORETICAL: float = 2.269

    # Sweep Temperatures (Paramagnetic phase T > Tc)
    TEMPERATURES: list[float] = [2.4, 2.45, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5]

    logger.info(f'Calculating correlation lengths for T > Tc (L={args.size})...')
    logger.info(f'Approaching Tc={TC_THEORETICAL} with {len(TEMPERATURES)} points.')

    sweep_params = [(T, args.size, args.steps, args.eq_steps, args.interval) for T in TEMPERATURES]
    results: list[tuple[float, float]] = parallel_sweep(get_correlation_length, sweep_params)

    temps_list, xis_list = zip(*results, strict=True)
    temps: np.ndarray = np.array(temps_list)
    xis: np.ndarray = np.array(xis_list)

    # Filter out failed fits
    valid: np.ndarray = ~np.isnan(xis)
    temps = temps[valid]
    xis = xis[valid]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Linear Plot: xi vs T
    ax1.plot(temps, xis, 'o-', markersize=6)
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel(r'Correlation Length $\xi$')
    ax1.set_title(r'Divergence of $\xi$ approaching $T_c$')
    ax1.grid(True)

    # 2. Log-Log Plot: xi vs (T - Tc)
    # Theory: xi ~ |T - Tc|^(-nu)
    reduced_T: np.ndarray = temps - TC_THEORETICAL

    # Fit power law (only possible when all reduced_T > 0 and xis > 0)
    nu: float | None = None
    try:
        log_t: np.ndarray = np.log(reduced_T)
        log_xi: np.ndarray = np.log(xis)
        slope, intercept = np.polyfit(log_t, log_xi, 1)
        nu = -slope
    except (np.linalg.LinAlgError, ValueError) as exc:
        logger.warning(f'Power-law fit failed: {exc}')

    ax2.loglog(reduced_T, xis, 'o', label='Simulation Data')

    if nu is not None:
        # Plot fit line
        fit_x: np.ndarray = np.linspace(min(reduced_T), max(reduced_T), 100)
        fit_y: np.ndarray = np.exp(intercept) * fit_x ** (-nu)
        ax2.loglog(fit_x, fit_y, 'r--', label=f'Fit ($\\nu \\approx {nu:.2f}$)')

        # Plot theoretical slope (nu=1) for comparison
        theory_y: np.ndarray = fit_y[len(fit_y) // 2] * (fit_x / fit_x[len(fit_x) // 2]) ** (-1)
        ax2.loglog(fit_x, theory_y, 'g:', label=r'Theory ($\nu=1$)')

    ax2.set_xlabel(r'$T - T_c$')
    ax2.set_ylabel(r'Correlation Length $\xi$')
    ax2.set_title(r'Critical Exponent $\nu$ Extraction')
    ax2.grid(True, which='both', ls='-', alpha=0.5)
    ax2.legend()

    output_dir: str = ensure_results_dir(args.output_dir)
    save_plot('correlation_divergence.png', directory=output_dir)


if __name__ == '__main__':
    run_divergence_analysis()
