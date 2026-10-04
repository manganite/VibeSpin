"""
Phase ordering evolution visualisation for the 2D q-state Clock model.

Quenches from a disordered state to T < T_c and records the spin configuration
at multiple time steps, plotting phase configurations, vorticity maps, and
radially averaged correlation functions G(r).
"""
from __future__ import annotations

import argparse
import logging

from scripts.clock._model_choice import add_clock_model_arguments, resolve_clock_model
from utils.evolution_helpers import run_ordering_evolution
from utils.system import parse_args_compat, setup_logging

# Anisotropy of the continuous model when --continuous is given without
# --aniso; it keeps the value this script used before the discrete model
# became the default.
_DEFAULT_CONTINUOUS_ANISO = 0.5


def main() -> None:
    """Run the simulation and generate a multi-row phase ordering figure."""
    parser = argparse.ArgumentParser(description='2D Clock Model Phase Ordering Visualisation')
    parser.add_argument('--size', type=int, default=256, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.2, help='Quench temperature T')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    add_clock_model_arguments(parser=parser, default_aniso=_DEFAULT_CONTINUOUS_ANISO)
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
    parser.add_argument('--output-dir', type=str, default='results/clock', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)
    choice = resolve_clock_model(
        parser=parser, args=args, default_aniso=_DEFAULT_CONTINUOUS_ANISO,
    )

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    logger.info(
        f'Clock phase ordering evolution (L={args.size}, T={args.temp},'
        f' q={args.q}, {choice.variant})'
    )
    logger.info(f'Recording snapshots at steps {sorted(args.targets)} ...')

    run_ordering_evolution(
        model_cls=choice.model_cls,
        model_kwargs=(
            choice.model_kwargs
            | ({} if args.seed is None else {'seed': args.seed})
        ),
        capture_vorticity=True,
        title=(
            f'2D {args.q}-state Clock Model Evolution ({choice.variant}) -'
            f' T = {args.temp}, L = {args.size}'
        ),
        size=args.size,
        temp=args.temp,
        step_targets=list(args.targets),
        output_dir=args.output_dir,
        logger=logger,
    )


if __name__ == '__main__':
    main()
