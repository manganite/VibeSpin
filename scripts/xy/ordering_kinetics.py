"""
Ordering kinetics analysis for the 2D XY model.

Quenches from a disordered state to T < T_BKT and records length scale
growth and vortex density decay over time.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
from tqdm import tqdm

from models.xy_model import XYSimulation
from utils.cli_helpers import parse_args_compat
from utils.physics_helpers import compute_kinetics_metrics, power_fit
from utils.plotting import ensure_results_dir, plot_ordering_kinetics
from utils.system_helpers import _BAR_FORMAT, setup_logging


def main() -> None:
    """Run the XY ordering kinetics simulation."""
    parser = argparse.ArgumentParser(description='2D XY Model Ordering Kinetics Analysis')
    parser.add_argument('--size', type=int, default=256, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.1, help='Quench temperature T')
    parser.add_argument('--max-steps', type=int, default=1000, help='Total MC steps')
    parser.add_argument('--samples', type=int, default=15, help='Number of measurement points')
    parser.add_argument('--fit-min', type=int, default=20, help='Min step for power-law fit')
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    T = args.temp
    T_BKT: float = 0.893

    logger.info(f'XY ordering kinetics analysis (L={L}, T={T:.3f} < T_BKT={T_BKT})')

    step_targets = np.unique(np.logspace(0, np.log10(args.max_steps), num=args.samples).astype(int))
    sim = XYSimulation(size=L, temp=T, update='random')

    N_data = len(step_targets)
    t = np.zeros(N_data)
    R_sk = np.zeros(N_data)
    R_xi = np.zeros(N_data)
    v_dens = np.zeros(N_data)

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
        v_dens[i] = sim._get_vortex_density()

        logger.debug(f't={current_step}: R_sk={R_sk[i]:.2f}, xi={R_xi[i]:.2f}, n_v={v_dens[i]:.4f}')

    # Power law fits
    fit_mask = t >= args.fit_min
    exponents = {}
    prefactors = {}

    for key, data in [('R_sk', R_sk), ('xi', R_xi), ('third', v_dens)]:
        exp, pre = power_fit(t_arr=t, y_arr=data, mask=fit_mask)
        exponents[key], prefactors[key] = exp, pre
        if exp:
            label = 'Growth' if key != 'third' else 'Decay'
            logger.info(f'{key} {label} exponent: {exp:.3f}')

    plot_ordering_kinetics(
        t=t,
        R_sk=R_sk,
        R_xi=R_xi,
        third_metric=v_dens,
        third_metric_label='Vortex Density $n_v(t)$',
        exponents=exponents,
        prefactors=prefactors,
        fit_mask=fit_mask,
        title=f'2D XY Ordering Kinetics - $T = {T}$ ($< T_{{BKT}} \\approx {T_BKT}$), $L = {L}$',
        filename='ordering_kinetics.png',
        directory=ensure_results_dir(directory=args.output_dir),
        left_title='Phase Ordering Dynamics',
        right_title='Vortex Decay',
    )


if __name__ == '__main__':
    main()
