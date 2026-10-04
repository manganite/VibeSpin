"""
Analysis of the helicity modulus (superfluid stiffness) in the 2D XY model.
Used to identify the universal jump at the BKT transition.
"""
from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.xy_model import XYSimulation
from utils.equilibration import prepare_equilibrated_simulation
from utils.plotting import ensure_results_dir, save_plot
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    summarize_primary_observable,
)
from utils.sweep_helpers import SUMMARY_FIELDS, build_single_run_schema, derive_point_seed
from utils.system import parallel_sweep, parse_args_compat, setup_logging


def simulate_helicity(params: tuple[float, int, int, int, int, int]) -> dict[str, float]:
    """
    Run simulation for a single temperature and compute the helicity modulus.

    Parameters
    ----------
    params : tuple[float, int, int, int, int, int]
        Tuple of (T, L, eq_probe_steps, eq_max_steps, meas_steps, seed). The
        seed belongs to the random start; the ordered start derives its own.

    Returns
    -------
    dict[str, float]
        Blocking summary of the helicity modulus (``value``, ``err``,
        ``ci_low``, ``ci_high``, ``tau_int``, ``n_eff``, ``samples``).
        Upsilon is the mean of the per-sweep quantity
        ``(Sum cos - (Sum sin)^2 / T) / L^2``, so its error follows from
        blocking that series directly. All fields are NaN when the two
        starts did not converge within ``eq_max_steps``.

    Raises
    ------
    ValueError
        If ``T`` is less than or equal to 0.
    """
    T, L, eq_probe_steps, eq_max_steps, meas_steps, seed = params
    if T <= 0.0:
        raise ValueError(f'Temperature must be positive to compute helicity modulus, got {T}')

    # The XY model has no metastable domain states, but its random start can
    # relax for thousands of sweeps near T_BKT; the stuck detector would end
    # such runs early, so the pair runs until it converges. If it never does,
    # the point is not certified and is stored as NaN.
    sim, outcome = prepare_equilibrated_simulation(
        model_cls=XYSimulation, model_kwargs={}, size=L, temp=T, seed=seed,
        chunk_size=eq_probe_steps, max_steps=eq_max_steps, detect_stuck=False,
    )
    if not outcome.certified:
        return dict.fromkeys(SUMMARY_FIELDS, float('nan'))

    cos_sums: np.ndarray = np.empty(meas_steps)
    sin_sums: np.ndarray = np.empty(meas_steps)

    for k in range(meas_steps):
        sim.step()
        cos_sums[k], sin_sums[k] = sim.get_helicity_data()

    # Upsilon = (1/L^2) * (<Sum cos> - (1/T) * <(Sum sin)^2>) is linear in the
    # two averages, so it is the mean of the per-sweep combination below.
    per_sweep = (cos_sums - sin_sums**2 / T) / (L**2)
    return summarize_primary_observable(
        time_series=per_sweep, confidence=DEFAULT_CONFIDENCE_LEVEL,
    )


def main() -> None:
    """
    Run parallel sweep to calculate helicity modulus across the BKT region.
    """
    parser = argparse.ArgumentParser(description='2D XY Model Helicity Modulus Analysis')
    parser.add_argument('--size', type=int, default=64, help='Linear lattice size L')
    parser.add_argument(
        '--eq-probe-steps', type=int, default=1000,
        help='Chunk size for convergence check during equilibration (default: 1000)',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=200000,
        help='Hard cap on equilibration steps (default: 200000)',
    )
    parser.add_argument('--meas-steps', type=int, default=20000, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=1.5, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=30, help='Number of temperature points')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Replica index selecting a reproducible seed per temperature (default: 0)',
    )
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    # Generate temperature points
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)

    logger.info(f'Starting Helicity Modulus sweep for L={args.size}...')
    logger.info(f'Range: [{args.t_min}, {args.t_max}] with {args.t_points} points.')

    sweep_params = [
        (
            float(T), args.size, args.eq_probe_steps, args.eq_max_steps, args.meas_steps,
            derive_point_seed(temperature_index=i, seed_index=args.seed),
        )
        for i, T in enumerate(temperatures)
    ]
    summaries: list[dict[str, float]] = parallel_sweep(
        worker_func=simulate_helicity, params=sweep_params,
    )
    upsilons = np.array([s['value'] for s in summaries])
    upsilon_err = np.array([s['err'] for s in summaries])

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        temperatures, upsilons, yerr=upsilon_err, fmt='o-', capsize=2,
        label=r'Helicity Modulus $\Upsilon$',
    )

    # Plot the universal jump line: 2/pi * T
    t_line: np.ndarray = np.linspace(args.t_min, args.t_max, 100)
    plt.plot(t_line, 2 * t_line / np.pi, 'r--', label=r'Universal Jump $\frac{2}{\pi} k_B T$')

    plt.xlabel('Temperature (T)')
    plt.ylabel(r'Helicity Modulus $\Upsilon$')
    plt.title('BKT Transition: Superfluid Stiffness')
    plt.grid(True)
    plt.legend()
    # Above T_BKT the estimator scatters around zero and can go negative.
    if np.any(np.isfinite(upsilons)):
        plt.ylim(bottom=min(0.0, float(np.nanmin(upsilons)) - 0.03))

    output_dir: str = ensure_results_dir(directory=args.output_dir)
    save_plot(filename='helicity_modulus.png', directory=output_dir)

    # Save data for notebook consumption
    npz_path = f'{output_dir}/helicity_modulus.npz'
    np.savez_compressed(
        npz_path,
        temperatures=temperatures,
        helicity_modulus=upsilons,
        # The worker returns NaN only for points whose equilibration hit
        # eq_max_steps without convergence.
        equilibrated=np.isfinite(upsilons),
        **build_single_run_schema(
            prefix='helicity_modulus', summaries=summaries,
            uncertainty_method=UNCERTAINTY_METHOD_BLOCKING,
            confidence=DEFAULT_CONFIDENCE_LEVEL,
        ),
        L=args.size,
        eq_probe_steps=args.eq_probe_steps,
        eq_max_steps=args.eq_max_steps,
        meas_steps=args.meas_steps,
        seed_index=args.seed,
    )
    logger.info(f'Data saved to {npz_path}')


if __name__ == '__main__':
    main()
