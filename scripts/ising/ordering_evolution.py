"""
Domain ordering evolution visualisation for the 2D Ising model.

Quenches from a disordered state to T < T_c and records the spin configuration
at multiple time steps, plotting spin configurations, structure factors, and
radially averaged correlation functions G(r).
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from models.ising_model import IsingSimulation
from utils.cli_helpers import parse_args_compat
from utils.plotting import ensure_results_dir, plot_ordering_evolution
from utils.system_helpers import setup_logging


def main() -> None:
    """Run the simulation and generate a multi-row domain ordering figure."""
    parser = argparse.ArgumentParser(description='2D Ising Model Domain Ordering Visualisation')
    parser.add_argument('--size', type=int, default=512, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=2.0, help='Quench temperature T')
    parser.add_argument(
        '--targets',
        type=int,
        nargs='+',
        default=[1, 10, 100, 1000],
        help='MC steps at which to take snapshots',
    )
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    T = args.temp
    STEP_TARGETS = sorted(args.targets)
    T_CRIT: float = 2.269

    logger.info(f'Ising domain ordering evolution (L={L}, T={T})')
    logger.info(f'Recording snapshots at steps {STEP_TARGETS} ...')

    sim = IsingSimulation(size=L, temp=T, update='random')
    n_targets: int = len(STEP_TARGETS)

    # Storage for snapshots
    snapshots: list[np.ndarray] = []
    snapshots_gr: list[tuple[np.ndarray, np.ndarray]] = []

    current_step: int = 0

    for _i, target in enumerate(STEP_TARGETS):
        steps_to_run = target - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = target

        if sim.spins is not None:
            snapshots.append(sim.spins.copy())
            snapshots_gr.append(sim._calculate_correlation_function())
            logger.debug(f'Captured snapshot at step {target}')

    logger.info(f'Collected {n_targets} snapshots. Saving figure ...')

    title = f'2D Ising Ordering Evolution - T = {T} (< T_c ≈ {T_CRIT}), L = {L}'

    plot_ordering_evolution(
        targets=STEP_TARGETS,
        snapshots=snapshots,
        gr_data=snapshots_gr,
        vorticity_data=None,  # Ising uses Sk fallback in plot_ordering_evolution
        title=title,
        filename='ordering_evolution.png',
        directory=ensure_results_dir(directory=args.output_dir),
        is_vector=False,
    )


if __name__ == '__main__':
    main()
