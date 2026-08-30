"""
Domain ordering evolution visualisation for the 2D Ising model.

Quenches from a disordered state to T < T_c and records the spin configuration
at multiple time steps, plotting spin configurations, structure factors, and
radially averaged correlation functions G(r).
"""
from __future__ import annotations

import argparse
import logging

from models.ising_model import IsingSimulation
from utils.evolution_helpers import run_ordering_evolution
from utils.system import parse_args_compat, setup_logging


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
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for a reproducible quench (default: unseeded)',
    )
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    T_CRIT: float = 2.269
    logger.info(f'Ising domain ordering evolution (L={args.size}, T={args.temp})')
    logger.info(f'Recording snapshots at steps {sorted(args.targets)} ...')

    run_ordering_evolution(
        model_cls=IsingSimulation,
        model_kwargs={} if args.seed is None else {'seed': args.seed},
        capture_vorticity=False,
        title=(
            f'2D Ising Ordering Evolution - T = {args.temp}'
            f' (< T_c ≈ {T_CRIT}), L = {args.size}'
        ),
        size=args.size,
        temp=args.temp,
        step_targets=list(args.targets),
        output_dir=args.output_dir,
        logger=logger,
    )


if __name__ == '__main__':
    main()
