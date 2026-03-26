"""
Ordering kinetics analysis for the 2D XY model.

Quenches from a disordered state to T < T_BKT and records length scale
growth and vortex density decay over time.
"""
from __future__ import annotations

import argparse
import logging

from models.xy_model import XYSimulation
from utils.kinetics_helpers import run_ordering_kinetics
from utils.system import parse_args_compat, setup_logging


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

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    T_BKT: float = 0.893
    logger.info(f'XY ordering kinetics analysis (L={args.size}, T={args.temp:.3f} < T_BKT={T_BKT})')

    run_ordering_kinetics(
        model_cls=XYSimulation,
        model_kwargs={},
        third_metric_fn=lambda sim: sim._get_vortex_density(),
        third_metric_label='Vortex Density $n_v(t)$',
        title=(
            f'2D XY Ordering Kinetics - $T = {args.temp}$'
            f' ($< T_{{BKT}} \\approx {T_BKT}$), $L = {args.size}$'
        ),
        left_title='Phase Ordering Dynamics',
        right_title='Vortex Decay',
        size=args.size,
        temp=args.temp,
        max_steps=args.max_steps,
        samples=args.samples,
        fit_min=args.fit_min,
        output_dir=args.output_dir,
        logger=logger,
    )


if __name__ == '__main__':
    main()
