"""
Analysis of correlation length divergence in the 2D Ising model.
Extracts the critical exponent nu by fitting correlation lengths near Tc.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from models.ising_model import IsingSimulation
from utils.plotting import ensure_results_dir, save_plot
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    summarize_seed_ensemble,
)
from utils.sweep_helpers import build_single_run_schema, derive_point_seed
from utils.system import parallel_sweep, parse_args_compat, setup_logging

#: Contiguous blocks of correlation samples used to estimate the error of xi.
_XI_BLOCKS = 16


def _fit_xi(*, r: np.ndarray, G_r: np.ndarray, L: int) -> float:
    """Fit log G(r) linearly over the clean range and return xi = -1/slope.

    Parameters
    ----------
    r : np.ndarray
        Radial distances.
    G_r : np.ndarray
        Averaged correlation function normalised to G(0) = 1.
    L : int
        Linear lattice size, which bounds the fit range.

    Returns
    -------
    float
        Correlation length, or NaN when fewer than three points are usable.
    """
    # r > 1 avoids short-range lattice effects, r < L/4 the periodic images,
    # and G > 1e-4 the noise floor.
    mask: np.ndarray = (r > 1) & (r < L // 4) & (G_r > 1e-4)
    if np.sum(mask) < 3:
        return float('nan')
    try:
        slope, _ = np.polyfit(r[mask], np.log(G_r[mask]), 1)
    except np.linalg.LinAlgError:
        return float('nan')
    return float('nan') if slope == 0.0 else float(-1.0 / slope)


def get_correlation_length(
    params: tuple[float, int, int, int, int, int],
) -> tuple[float, dict[str, float]]:
    """Simulate and extract the correlation length xi for a given temperature.

    The point estimate fits the correlation function averaged over the whole
    measurement. Its uncertainty comes from blocking: the samples are split
    into contiguous blocks, each block's averaged G(r) is fitted on its own,
    and the spread of the block estimates gives the standard error and a
    Student-t interval, centred on the full-run fit.

    Parameters
    ----------
    params : tuple[float, int, int, int, int, int]
        Tuple of (T, L, steps, eq_steps, sample_interval, seed).

    Returns
    -------
    tuple[float, dict[str, float]]
        The temperature and the summary ``value``, ``err``, ``ci_low``,
        ``ci_high``, ``tau_int`` (NaN, not defined for a fitted length),
        ``n_eff`` (NaN), and ``samples`` (number of correlation samples).
    """
    T, L, steps, eq_steps, sample_interval, seed = params
    logger = logging.getLogger('vibespin')
    logger.debug(f'Calculating xi for T={T}...')
    # All temperatures lie above T_c, where a fixed burn-in of many
    # autocorrelation times suffices and no ordered-start bias can arise.
    sim = IsingSimulation(size=L, temp=T, seed=seed)
    sim.equilibrate(n_steps=eq_steps)

    r = np.empty(0)
    samples: list[np.ndarray] = []
    for i in range(steps):
        sim.step()
        if i % sample_interval == 0:
            r, G = sim.calculate_correlation_function()
            samples.append(np.asarray(G, dtype=np.float64))
    G_all = np.asarray(samples)
    xi = _fit_xi(r=r, G_r=G_all.mean(axis=0), L=L)

    n_blocks = min(_XI_BLOCKS, G_all.shape[0] // 2)
    block_xis = np.array([
        _fit_xi(r=r, G_r=block.mean(axis=0), L=L)
        for block in np.array_split(G_all, n_blocks)
    ]) if n_blocks >= 2 else np.empty(0)
    finite = block_xis[np.isfinite(block_xis)]
    nan = float('nan')
    summary = {
        'value': xi, 'err': nan, 'ci_low': nan, 'ci_high': nan,
        'tau_int': nan, 'n_eff': nan, 'samples': float(G_all.shape[0]),
    }
    if finite.size >= 2 and np.isfinite(xi):
        spread = summarize_seed_ensemble(
            values=finite, within_seed_errors=np.full(finite.size, nan),
            confidence=DEFAULT_CONFIDENCE_LEVEL,
        )
        half_width = 0.5 * (spread['ci_high'] - spread['ci_low'])
        summary.update(
            err=float(spread['err']), ci_low=xi - half_width, ci_high=xi + half_width,
        )
    return T, summary


def main() -> None:
    """Run parallel simulation to extract the critical exponent nu from xi(T) divergence."""
    parser = argparse.ArgumentParser(description='2D Ising Model Correlation Divergence Analysis')
    parser.add_argument('--size', type=int, default=128, help='Linear lattice size L')
    parser.add_argument('--steps', type=int, default=50000, help='Measurement steps')
    parser.add_argument('--eq-steps', type=int, default=10000, help='Equilibration steps')
    parser.add_argument('--interval', type=int, default=20, help='Sample interval')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Replica index selecting a reproducible seed per temperature (default: 0)',
    )
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Physical Constants
    TC_THEORETICAL: float = 2.269

    # Sweep Temperatures (Paramagnetic phase T > Tc)
    TEMPERATURES: list[float] = [2.4, 2.45, 2.5, 2.6, 2.7, 2.8, 3.0, 3.2, 3.5]

    logger.info(f'Calculating correlation lengths for T > Tc (L={args.size})...')
    logger.info(f'Approaching Tc={TC_THEORETICAL} with {len(TEMPERATURES)} points.')

    sweep_params = [
        (
            float(T), args.size, args.steps, args.eq_steps, args.interval,
            derive_point_seed(temperature_index=i, seed_index=args.seed),
        )
        for i, T in enumerate(TEMPERATURES)
    ]
    results: list[tuple[float, dict[str, float]]] = parallel_sweep(
        worker_func=get_correlation_length, params=sweep_params
    )

    temps_list, summaries_list = zip(*results, strict=True)
    temps: np.ndarray = np.array(temps_list)
    xis: np.ndarray = np.array([summ['value'] for summ in summaries_list])

    # Filter out failed fits
    valid: np.ndarray = ~np.isnan(xis)
    temps = temps[valid]
    xis = xis[valid]
    summaries = [summ for summ, ok in zip(summaries_list, valid, strict=True) if ok]
    xi_err = np.array([summ['err'] for summ in summaries])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Linear Plot: xi vs T
    ax1.errorbar(temps, xis, yerr=xi_err, fmt='o-', markersize=6, capsize=2)
    ax1.set_xlabel('Temperature T')
    ax1.set_ylabel(r'Correlation Length $\xi$')
    ax1.set_title(r'Divergence of $\xi$ approaching $T_c$')
    ax1.grid(True)

    # 2. Log-Log Plot: xi vs (T - Tc)
    # Theory: xi ~ |T - Tc|^(-nu)
    reduced_T: np.ndarray = temps - TC_THEORETICAL

    # Fit power law (only possible when all reduced_T > 0 and xis > 0)
    nu: float | None = None
    fit_mask = (reduced_T > 0.0) & (xis > 0.0)
    if np.count_nonzero(fit_mask) >= 2:
        try:
            log_t = np.log(reduced_T[fit_mask])
            log_xi = np.log(xis[fit_mask])
            slope, intercept = np.polyfit(log_t, log_xi, 1)
            nu = -slope
        except np.linalg.LinAlgError as exc:
            logger.warning(f'Power-law fit failed: {exc}')
    else:
        logger.warning('Power-law fit skipped: need at least two positive xi(T - Tc) points.')

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

    output_dir: str = ensure_results_dir(directory=args.output_dir)
    save_plot(filename='correlation_divergence.png', directory=output_dir)

    # Save data for notebook consumption
    npz_path = f'{output_dir}/correlation_divergence.npz'
    save_kwargs: dict[str, Any] = dict(
        temperatures=temps,
        xi=xis,
        T_c=TC_THEORETICAL,
        seed_index=args.seed,
        **build_single_run_schema(
            prefix='xi', summaries=summaries,
            uncertainty_method=UNCERTAINTY_METHOD_BLOCKING,
            confidence=DEFAULT_CONFIDENCE_LEVEL,
        ),
        L=args.size,
        steps=args.steps,
        eq_steps=args.eq_steps,
        sample_interval=args.interval,
    )
    if nu is not None:
        save_kwargs['nu'] = nu
    np.savez_compressed(npz_path, **save_kwargs)
    logger.info(f'Data saved to {npz_path}')


if __name__ == '__main__':
    main()
