"""
Ordering kinetics analysis for the 2D Ising model.

Starting from a disordered state, quenches to T < T_c and tracks the growth of
domain size R(t) and the evolution of domain boundaries.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
from tqdm import tqdm

from models.ising_model import IsingSimulation
from utils.physics_helpers import compute_kinetics_metrics, power_fit
from utils.system_helpers import (
    _BAR_FORMAT,
    ensure_results_dir,
    plot_ordering_kinetics,
    setup_logging,
)


def compute_mean_intercept_length(sim: IsingSimulation) -> float:
    """Estimate domain size using the stereological mean intercept length (MIL)."""
    if sim.spins is None:
        return 0.0
    spins = sim.spins.astype(np.int8)
    N = sim.size
    row_walls = np.sum(spins[:, :-1] * spins[:, 1:] < 0)
    col_walls = np.sum(spins[:-1, :] * spins[1:, :] < 0)
    row_wrap = int(np.sum(spins[:, -1] * spins[:, 0] < 0))
    col_wrap = int(np.sum(spins[-1, :] * spins[0, :] < 0))
    total_walls = int(row_walls) + int(col_walls) + row_wrap + col_wrap
    mean_walls = total_walls / (2 * N)
    return float(N) / mean_walls if mean_walls != 0.0 else float(N)


def main() -> None:
    """Run the Ising ordering kinetics simulation."""
    parser = argparse.ArgumentParser(description='2D Ising Model Ordering Kinetics Analysis')
    parser.add_argument('--size', type=int, default=256, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.1, help='Quench temperature T')
    parser.add_argument('--max-steps', type=int, default=1000, help='Total MC steps')
    parser.add_argument('--samples', type=int, default=10, help='Number of measurement points')
    parser.add_argument('--fit-min', type=int, default=20, help='Min step for power-law fit')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    T = args.temp
    T_CRIT = 2.269

    logger.info(f'Ising ordering kinetics analysis (L={L}, T={T:.3f} < T_c={T_CRIT})')

    step_targets = np.unique(np.logspace(0, np.log10(args.max_steps), num=args.samples).astype(int))
    sim = IsingSimulation(size=L, temp=T, update='random')

    N_data = len(step_targets)
    t = np.zeros(N_data)
    R_sk = np.zeros(N_data)
    R_xi = np.zeros(N_data)
    R_mil = np.zeros(N_data)

    current_step = 0
    for i, target in enumerate(tqdm(step_targets, bar_format=_BAR_FORMAT, desc='Simulating')):
        steps_to_run = int(target) - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = int(target)

        metrics = compute_kinetics_metrics(sim=sim)
        t[i] = float(current_step)
        R_sk[i] = metrics['R_sk']
        R_xi[i] = metrics['xi']
        R_mil[i] = compute_mean_intercept_length(sim)

        logger.debug(f't={current_step}: R_sk={R_sk[i]:.2f}, xi={R_xi[i]:.2f}, MIL={R_mil[i]:.2f}')

    # Power law fits
    fit_mask = t >= args.fit_min
    exponents = {}
    prefactors = {}

    for key, data in [('R_sk', R_sk), ('xi', R_xi), ('third', R_mil)]:
        exp, pre = power_fit(t_arr=t, y_arr=data, mask=fit_mask)
        exponents[key], prefactors[key] = exp, pre
        if exp:
            logger.info(f'{key} exponent: {exp:.3f} (Allen-Cahn: 0.5)')

    plot_ordering_kinetics(
        t=t,
        R_sk=R_sk,
        R_xi=R_xi,
        third_metric=R_mil,
        third_metric_label='Mean Intercept Length $R_{MIL}$',
        exponents=exponents,
        prefactors=prefactors,
        fit_mask=fit_mask,
        title=f'2D Ising Ordering Kinetics — $T = {T}$ ($< T_c \\approx {T_CRIT}$), $L = {L}$',
        filename='ordering_kinetics.png',
        directory=ensure_results_dir(directory=args.output_dir),
        y_label='Domain Size Scale (lattice units)',
        left_title='Domain Coarsening',
        right_title='Boundary Wall Decay',
    )


if __name__ == '__main__':
    main()
