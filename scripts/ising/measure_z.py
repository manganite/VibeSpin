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
from utils.exceptions import ZeroVarianceAutocorrelationError
from utils.physics_helpers import calculate_autocorr
from utils.system_helpers import (
    adaptive_equilibrate,
    parallel_sweep,
    setup_logging,
)

#: Exact Onsager critical temperature for the 2D nearest-neighbour Ising model.
TC_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))


def _measure_tau_point(
    params: tuple[str, int, int, int, int],
) -> dict[str, Any]:
    """
    Worker: measure tau_int for a specific algorithm and lattice size at Tc.

    Parameters
    ----------
    params : tuple
        ``(update, L, eq_steps, meas_steps, seed)`` — update scheme, lattice size,
        minimum equilibration steps, measurement steps, and RNG seed.

    Returns
    -------
    dict
        Keys: ``update``, ``L``, ``tau_int``, ``wall_time``.
    """
    update, L, eq_steps, meas_steps, seed = params
    sim = IsingSimulation(size=L, temp=TC_ISING, update=update, seed=seed)

    # Thorough equilibration at Tc
    # factor=100.0 ensures probe window is much larger than tau_int
    adaptive_equilibrate(sim, min_steps=eq_steps, factor=100.0)

    t0 = time.perf_counter()
    mags, _ = sim.run(n_steps=meas_steps)
    wall_time = time.perf_counter() - t0

    mags_arr = np.array(mags)
    try:
        _, tau_int = calculate_autocorr(time_series=mags_arr)
    except ZeroVarianceAutocorrelationError:
        tau_int = float('nan')

    return {
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
        '--eq-steps', type=int, default=5000,
        help='Minimum equilibration steps at Tc (default: 5000)',
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
        '--output-dir', type=str, default='results/ising',
        help='Output directory (default: results/ising)',
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)

    sizes = sorted(args.sizes)
    logger.info(
        'Measuring dynamical exponent z at Tc=%.6f for L in %s.',
        TC_ISING, sizes,
    )

    sweep_params = []
    # Metropolis points (use 'random' for physical dynamics)
    for idx, L in enumerate(sizes):
        sweep_params.append(('random', L, args.eq_steps, args.meas_steps_metro, idx * 2000))

    # Wolff points
    for idx, L in enumerate(sizes):
        seed = (idx + len(sizes)) * 2000
        sweep_params.append(('wolff', L, args.eq_steps, args.meas_steps_wolff, seed))

    raw: list[dict[str, Any]] = parallel_sweep(
        worker_func=_measure_tau_point, params=sweep_params,
    )

    results_metro = [r for r in raw if r['update'] == 'random']
    results_wolff = [r for r in raw if r['update'] == 'wolff']

    L_metro = np.array([r['L'] for r in results_metro])
    tau_metro = np.array([r['tau_int'] for r in results_metro])

    L_wolff = np.array([r['L'] for r in results_wolff])
    tau_wolff = np.array([r['tau_int'] for r in results_wolff])

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, 'dynamic_exponent_z.npz')
    np.savez(
        npz_path,
        L_metro=L_metro,
        tau_metro=tau_metro,
        L_wolff=L_wolff,
        tau_wolff=tau_wolff,
        Tc=TC_ISING,
    )
    logger.info('Data saved to %s', npz_path)


if __name__ == '__main__':
    main()
