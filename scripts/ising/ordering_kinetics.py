"""
Ordering kinetics analysis for the 2D Ising model.

Starting from a disordered state, quenches to T < T_c and tracks the growth of
domain size R(t) and the evolution of domain boundaries.
"""
from __future__ import annotations

import argparse
import logging

from models.ising_model import IsingSimulation
from utils.kinetics_helpers import compute_mean_intercept_length, run_ordering_kinetics
from utils.plotting import ensure_results_dir
from utils.system import parse_args_compat, setup_logging


def main() -> None:
    """Run the Ising ordering kinetics simulation."""
    parser = argparse.ArgumentParser(description='2D Ising Model Ordering Kinetics Analysis')
    parser.add_argument('--size', type=int, default=256, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.1, help='Quench temperature T')
    parser.add_argument('--max-steps', type=int, default=400, help='Total MC steps')
    parser.add_argument('--samples', type=int, default=25, help='Number of measurement points')
    parser.add_argument('--fit-min', type=int, default=5, help='Min step for power-law fit')
    parser.add_argument(
        '--n-seeds', '--seeds', dest='seeds', type=int, default=4,
        help='Number of independent seeds for ensemble averaging',
    )
    parser.add_argument('--base-seed', type=int, default=42, help='Starting seed for the ensemble')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    T_CRIT: float = 2.269
    logger.info(
        f'Ising ordering kinetics analysis (L={args.size}, T={args.temp:.3f} < T_c={T_CRIT})'
    )

    run_ordering_kinetics(
        model_cls=IsingSimulation,
        model_kwargs={},
        third_metric_fn=lambda s: compute_mean_intercept_length(sim=s),
        third_metric_label='Mean Intercept Length $R_{MIL}$',
        title=(
            f'2D Ising Ordering Kinetics - $T = {args.temp}$'
            f' ($< T_c \\approx {T_CRIT}$), $L = {args.size}$'
        ),
        left_title='Domain Coarsening',
        right_title='Boundary Wall Decay',
        y_label='Domain Size Scale (lattice units)',
        size=args.size,
        temp=args.temp,
        max_steps=args.max_steps,
        samples=args.samples,
        fit_min=args.fit_min,
        output_dir=args.output_dir,
        logger=logger,
        npz_path=f'{ensure_results_dir(directory=args.output_dir)}/ordering_kinetics.npz',
        n_seeds=args.seeds,
        base_seed=args.base_seed,
    )


if __name__ == '__main__':
    main()
