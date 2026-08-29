"""
Measure the dynamical critical exponent z for Metropolis and Wolff algorithms.

This script runs simulations of the 2D Ising model at the critical temperature Tc
for various lattice sizes L. It calculates the integrated autocorrelation time
tau_int and fits tau_int ~ L^z to extract the dynamic exponent z.

Results are saved to ``results/ising/dynamic_exponent_z.npz``.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

import numpy as np

from models.ising_model import IsingSimulation
from utils.equilibration import convergence_equilibrate
from utils.exceptions import ZeroVarianceAutocorrelationError
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_REPLICATE,
    calculate_autocorr,
    summarize_replicate_samples,
)
from utils.system import parallel_sweep, setup_logging

#: Exact Onsager critical temperature for the 2D nearest-neighbour Ising model.
TC_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))


def _measure_tau_point(
    params: tuple[int, int, str, int, int, int, int, int],
) -> dict[str, Any]:
    """
    Worker: measure tau_int for a specific algorithm and lattice size at Tc.

    Parameters
    ----------
    params : tuple
        ``(size_idx, seed_idx, update, L, eq_probe_steps, eq_max_steps,
        meas_steps, seed)`` — grid indices for result aggregation, update
        scheme, lattice size, chunk size for convergence, hard cap on
        equilibration, measurement steps, and RNG seed.

    Returns
    -------
    dict
        Keys: ``size_idx``, ``seed_idx``, ``update``, ``L``, ``tau_int``,
        ``wall_time``.
    """
    size_idx, seed_idx, update, L, eq_probe_steps, eq_max_steps, meas_steps, seed = params
    sim_r = IsingSimulation(size=L, temp=TC_ISING, update=update, init_state='random', seed=seed)
    sim_o = IsingSimulation(size=L, temp=TC_ISING, update=update, init_state='ordered', seed=seed)

    # Thorough equilibration at Tc via two-start convergence
    convergence_equilibrate(
        sim_random=sim_r, sim_ordered=sim_o,
        chunk_size=eq_probe_steps, max_steps=eq_max_steps,
    )

    t0 = time.perf_counter()
    mags, _ = sim_r.run(n_steps=meas_steps)
    wall_time = time.perf_counter() - t0

    mags_arr = np.array(mags)
    try:
        _, tau_int = calculate_autocorr(time_series=mags_arr)
    except ZeroVarianceAutocorrelationError:
        tau_int = float('nan')

    return {
        'size_idx': int(size_idx),
        'seed_idx': int(seed_idx),
        'update': update,
        'L': float(L),
        'tau_int': float(tau_int),
        'wall_time': float(wall_time),
    }


def main() -> None:
    """
    Execute the dynamical critical exponent measurement sweep.
    """
    parser = argparse.ArgumentParser(
        description='Measure dynamical critical exponent z for the 2D Ising model.',
    )
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[16, 32, 48, 64, 96, 128],
        help='Lattice sizes L to sweep (default: 16 32 48 64 96 128)',
    )
    parser.add_argument(
        '--eq-probe-steps', type=int, default=1000,
        help='Chunk size for convergence check during equilibration (default: 1000)',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=500000,
        help='Hard cap on equilibration steps at Tc (default: 500000)',
    )
    parser.add_argument(
        '--meas-steps-metro', type=int, default=400000,
        help='Measurement steps for Metropolis (default: 400000)',
    )
    parser.add_argument(
        '--meas-steps-wolff', type=int, default=100000,
        help='Measurement steps for Wolff (default: 100000)',
    )
    parser.add_argument(
        '--n-seeds', type=int, default=10,
        help='Independent seed replicas per (algorithm, L) point (default: 10)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/ising',
        help='Output directory (default: results/ising)',
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)

    sizes = sorted(args.sizes)
    n_seeds = int(args.n_seeds)
    logger.info(
        'Measuring dynamical exponent z at Tc=%.6f for L in %s, %d seed replicas.',
        TC_ISING, sizes, n_seeds,
    )

    # Seeds are derived deterministically from (size_idx, seed_idx).
    # Metropolis uses base = size_idx * 100_000 + seed_idx * 1_000.
    # Wolff uses base + 50_000 to ensure non-overlapping sequences.
    sweep_params: list[tuple[int, int, str, int, int, int, int, int]] = []
    for size_idx, L in enumerate(sizes):
        for seed_idx in range(n_seeds):
            base = size_idx * 100_000 + seed_idx * 1_000
            sweep_params.append((
                size_idx, seed_idx, 'random', L,
                args.eq_probe_steps, args.eq_max_steps, args.meas_steps_metro,
                base,
            ))
            sweep_params.append((
                size_idx, seed_idx, 'wolff', L,
                args.eq_probe_steps, args.eq_max_steps, args.meas_steps_wolff,
                base + 50_000,
            ))

    raw: list[dict[str, Any]] = parallel_sweep(
        worker_func=_measure_tau_point, params=sweep_params,
    )

    n_sizes = len(sizes)
    tau_metro_samples = np.full((n_sizes, n_seeds), np.nan)
    tau_wolff_samples = np.full((n_sizes, n_seeds), np.nan)

    for r in raw:
        i = int(r['size_idx'])
        s = int(r['seed_idx'])
        if r['update'] == 'random':
            tau_metro_samples[i, s] = float(r['tau_int'])
        else:
            tau_wolff_samples[i, s] = float(r['tau_int'])

    def _summary(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute median and 16-84 percentile envelopes across seeds."""
        med = np.nanmedian(samples, axis=1)
        p16 = np.nanpercentile(samples, 16, axis=1)
        p84 = np.nanpercentile(samples, 84, axis=1)
        return med, p16, p84

    L_arr = np.asarray(sizes, dtype=float)
    tau_metro, tau_metro_p16, tau_metro_p84 = _summary(tau_metro_samples)
    tau_wolff, tau_wolff_p16, tau_wolff_p84 = _summary(tau_wolff_samples)

    metro_summary = summarize_replicate_samples(samples=tau_metro_samples)
    wolff_summary = summarize_replicate_samples(samples=tau_wolff_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, 'dynamic_exponent_z.npz')
    np.savez(
        npz_path,
        L_metro=L_arr,
        tau_metro=tau_metro,
        tau_metro_p16=tau_metro_p16,
        tau_metro_p84=tau_metro_p84,
        tau_metro_samples=tau_metro_samples,
        L_wolff=L_arr,
        tau_wolff=tau_wolff,
        tau_wolff_p16=tau_wolff_p16,
        tau_wolff_p84=tau_wolff_p84,
        tau_wolff_samples=tau_wolff_samples,
        Tc=TC_ISING,
        n_seeds=np.int64(n_seeds),
        tau_metro_value=metro_summary['value'],
        tau_metro_err=metro_summary['err'],
        tau_metro_ci_low=metro_summary['ci_low'],
        tau_metro_ci_high=metro_summary['ci_high'],
        tau_wolff_value=wolff_summary['value'],
        tau_wolff_err=wolff_summary['err'],
        tau_wolff_ci_low=wolff_summary['ci_low'],
        tau_wolff_ci_high=wolff_summary['ci_high'],
        uncertainty_method=UNCERTAINTY_METHOD_REPLICATE,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    )
    logger.info('Data saved to %s', npz_path)


if __name__ == '__main__':
    main()
