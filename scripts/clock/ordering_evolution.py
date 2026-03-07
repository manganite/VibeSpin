"""
Domain snapshot visualisation for the 2D q-state Clock model.

Quenches from a disordered state to T < T_c and records the spin configuration
at multiple time steps, plotting phase configurations, vorticity maps, and
radially averaged correlation functions G(r).
"""

import argparse
import logging

import numpy as np

from models.clock_model import ClockSimulation
from utils.system_helpers import ensure_results_dir, plot_ordering_evolution, setup_logging


def main() -> None:
    """Run the snapshot simulation and generate a multi-row evolution figure."""
    parser = argparse.ArgumentParser(description='2D Clock Model Domain Snapshot Visualisation')
    parser.add_argument('--size', type=int, default=256, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.2, help='Quench temperature T')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    parser.add_argument('--aniso', type=float, default=0.5, help='Anisotropy strength A')
    parser.add_argument(
        '--targets',
        type=int,
        nargs='+',
        default=[1, 10, 100, 1000],
        help='MC steps at which to take snapshots',
    )
    parser.add_argument('--output-dir', type=str, default='results/clock', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    T = args.temp
    Q = args.q
    A = args.aniso
    STEP_TARGETS = sorted(args.targets)

    logger.info(f'Clock domain snapshots (L={L}, T={T}, q={Q}, A={A})')
    logger.info(f'Recording snapshots at steps {STEP_TARGETS} ...')

    sim = ClockSimulation(size=L, temp=T, q=Q, A=A)
    n_targets: int = len(STEP_TARGETS)

    # Storage for snapshots
    snapshots: list[np.ndarray] = []
    snapshots_vort: list[np.ndarray] = []
    snapshots_gr: list[tuple[np.ndarray, np.ndarray]] = []

    current_step: int = 0

    for _i, target in enumerate(STEP_TARGETS):
        steps_to_run = target - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = target

        if sim.spins is not None:
            snapshots.append(sim.spins.copy())
            snapshots_vort.append(sim._calculate_vorticity())
            snapshots_gr.append(sim._calculate_correlation_function())
            logger.debug(f'Captured snapshot at step {target}')

    logger.info(f'Collected {n_targets} snapshots. Saving figure ...')

    title = f'2D {Q}-state Clock Model Evolution — T = {T}, L = {L}, A = {A}'

    plot_ordering_evolution(
        targets=STEP_TARGETS,
        snapshots=snapshots,
        gr_data=snapshots_gr,
        vorticity_data=snapshots_vort,
        title=title,
        filename='ordering_evolution.png',
        directory=ensure_results_dir(args.output_dir),
        is_vector=True,
    )


if __name__ == '__main__':
    main()
