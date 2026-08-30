"""
Regenerate every cached dataset the notebooks read.

Each notebook loads a NPZ file if it finds one and otherwise recomputes a
reduced version of the same physics inline, so a fresh checkout renders but
does so at fallback quality.  This script produces the full set in one
command, in dependency-free order, reporting what it ran and how long each
step took.

Two profiles are available.  The default runs every script at its own
production defaults, which is the data the published figures are built from
and takes roughly an hour on four cores.  ``--quick`` runs the same scripts
with sharply reduced lattices, temperature grids, and step counts; it is a
smoke test of the pipeline in a couple of minutes and its output is not
suitable for physics.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import NamedTuple

from utils.system import parse_args_compat, setup_logging


class DataProduct(NamedTuple):
    """
    One script invocation and the file it is expected to produce.

    Parameters
    ----------
    key : str
        Selector of the form ``model/script``, matched by ``--only`` and
        ``--skip`` and printed by ``--list``.
    module : str
        Importable module path run with ``python -m``.
    output : str
        Path of the NPZ file relative to the output root, used by
        ``--skip-existing`` and by the completion report.
    quick_args : tuple[str, ...]
        Arguments that replace the production defaults under ``--quick``.
    """

    key: str
    module: str
    output: str
    quick_args: tuple[str, ...]


#: Every dataset a notebook loads, in the order the pipeline table of
#: SCRIPTS.md lists them.  Scripts that only draw a figure are not included
#: because no notebook depends on their output.
DATA_PRODUCTS: tuple[DataProduct, ...] = (
    DataProduct(
        key='ising/temperature_sweep',
        module='scripts.ising.temperature_sweep',
        output='ising/temperature_sweep_data.npz',
        quick_args=(
            '--size', '16', '--t-points', '6', '--n-seeds', '1',
            '--meas-steps', '400', '--eq-max-steps', '2000',
        ),
    ),
    DataProduct(
        key='ising/measure_z',
        module='scripts.ising.measure_z',
        output='ising/dynamic_exponent_z.npz',
        quick_args=(
            '--sizes', '8', '12', '--n-seeds', '1', '--meas-steps-metro', '400',
            '--meas-steps-wolff', '200', '--eq-max-steps', '2000',
        ),
    ),
    DataProduct(
        key='ising/wolff_efficiency',
        module='scripts.ising.wolff_efficiency',
        output='ising/wolff_efficiency.npz',
        quick_args=(
            '--size', '16', '--t-points', '5', '--n-seeds', '1',
            '--meas-steps', '400', '--eq-max-steps', '2000',
        ),
    ),
    DataProduct(
        key='ising/correlation_comparison',
        module='scripts.ising.correlation_comparison',
        output='ising/correlation_comparison.npz',
        quick_args=('--size', '24', '--steps', '400', '--eq-max', '2000'),
    ),
    DataProduct(
        key='ising/correlation_divergence',
        module='scripts.ising.correlation_divergence',
        output='ising/correlation_divergence.npz',
        quick_args=('--size', '24', '--steps', '400', '--eq-steps', '200'),
    ),
    DataProduct(
        key='ising/ordering_kinetics',
        module='scripts.ising.ordering_kinetics',
        output='ising/ordering_kinetics.npz',
        quick_args=('--size', '32', '--max-steps', '200', '--samples', '12', '--n-seeds', '1'),
    ),
    DataProduct(
        key='ising/coarsening_analysis',
        module='scripts.ising.coarsening_analysis',
        output='ising/coarsening_analysis.npz',
        quick_args=(
            '--size', '32', '--quench-steps', '200', '--quench-seeds', '1',
            '--bridge-steps', '200', '--bridge-seeds', '1',
            '--ens-steps', '200', '--ens-seeds', '2',
            '--xi-eq-max', '2000', '--xi-eq-steps', '200',
        ),
    ),
    DataProduct(
        key='xy/temperature_sweep',
        module='scripts.xy.temperature_sweep',
        output='xy/temperature_sweep_data.npz',
        quick_args=(
            '--size', '16', '--t-points', '6', '--n-seeds', '1',
            '--meas-steps', '400', '--eq-max-steps', '2000',
        ),
    ),
    DataProduct(
        key='xy/bkt_transition',
        module='scripts.xy.bkt_transition',
        output='xy/bkt_transition.npz',
        quick_args=(
            '--size', '16', '--t-points', '5', '--meas-steps', '400',
            '--eq-max-steps', '2000',
        ),
    ),
    DataProduct(
        key='xy/helicity_modulus',
        module='scripts.xy.helicity_modulus',
        output='xy/helicity_modulus.npz',
        quick_args=(
            '--size', '16', '--t-points', '5', '--meas-steps', '400',
            '--eq-max-steps', '2000',
        ),
    ),
    DataProduct(
        key='xy/correlation_comparison',
        module='scripts.xy.correlation_comparison',
        output='xy/correlation_comparison.npz',
        quick_args=('--size', '24', '--steps', '400', '--eq-max', '2000'),
    ),
    DataProduct(
        key='xy/ordering_kinetics',
        module='scripts.xy.ordering_kinetics',
        output='xy/ordering_kinetics.npz',
        quick_args=('--size', '32', '--max-steps', '200', '--samples', '12', '--n-seeds', '1'),
    ),
    DataProduct(
        key='clock/temperature_sweep',
        module='scripts.clock.temperature_sweep',
        output='clock/temperature_sweep_data.npz',
        quick_args=(
            '--size', '16', '--t-points', '6', '--n-seeds', '1',
            '--meas-steps', '400', '--eq-max-steps', '2000',
        ),
    ),
    DataProduct(
        key='clock/correlation_comparison',
        module='scripts.clock.correlation_comparison',
        output='clock/correlation_comparison.npz',
        quick_args=('--size', '24', '--steps', '400', '--eq-max', '2000'),
    ),
    DataProduct(
        key='clock/ordering_kinetics',
        module='scripts.clock.ordering_kinetics',
        output='clock/ordering_kinetics.npz',
        quick_args=('--size', '32', '--max-steps', '200', '--samples', '12', '--n-seeds', '1'),
    ),
    DataProduct(
        key='benchmarks/throughput',
        module='scripts.benchmarks.throughput',
        output='benchmarks/scaling_benchmark.npz',
        quick_args=('--sizes', '16', '24', '--sweeps', '20'),
    ),
)


def select_products(
    *,
    products: tuple[DataProduct, ...],
    only: list[str] | None,
    skip: list[str] | None,
) -> list[DataProduct]:
    """
    Filter the product table by substring match on the selector key.

    Parameters
    ----------
    products : tuple[DataProduct, ...]
        Full table to filter.
    only : list[str] or None
        Keep a product if its key contains any of these substrings.  None or
        an empty list keeps everything.
    skip : list[str] or None
        Drop a product if its key contains any of these substrings, applied
        after ``only``.

    Returns
    -------
    list[DataProduct]
        Selected products in table order.

    Raises
    ------
    ValueError
        If a pattern in ``only`` or ``skip`` matches no key at all, which is
        almost always a typo rather than an intentionally empty selection.
    """
    for pattern in list(only or []) + list(skip or []):
        if not any(pattern in p.key for p in products):
            raise ValueError(
                f'Pattern {pattern!r} matches no data product. '
                f'Known keys: {", ".join(p.key for p in products)}'
            )

    selected = [p for p in products if not only or any(o in p.key for o in only)]
    return [p for p in selected if not any(s in p.key for s in (skip or []))]


def _format_duration(*, seconds: float) -> str:
    """Render a duration as minutes and seconds, or seconds alone under a minute."""
    if seconds < 60.0:
        return f'{seconds:.1f}s'
    return f'{int(seconds // 60)}m{seconds % 60:04.1f}s'


def run_product(
    *,
    product: DataProduct,
    output_root: str,
    quick: bool,
    dry_run: bool,
    logger: logging.Logger,
) -> tuple[bool, float]:
    """
    Run one script and report whether it succeeded.

    Parameters
    ----------
    product : DataProduct
        Entry to run.
    output_root : str
        Directory the per-model output directories live under.
    quick : bool
        If True, pass the entry's reduced arguments instead of relying on the
        script's production defaults.
    dry_run : bool
        If True, log the command and return without running it.
    logger : logging.Logger
        Logger for progress output.

    Returns
    -------
    tuple[bool, float]
        Success flag and wall-clock seconds spent.
    """
    model = product.key.split('/')[0]
    command = [
        sys.executable, '-m', product.module,
        '--output-dir', os.path.join(output_root, model),
    ]
    if quick:
        command.extend(product.quick_args)

    if dry_run:
        logger.info(f'  would run: {" ".join(command)}')
        return True, 0.0

    started = time.perf_counter()
    completed = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        logger.error(
            f'  {product.key} failed with exit code {completed.returncode} '
            f'after {_format_duration(seconds=elapsed)}'
        )
        return False, elapsed

    outpath = os.path.join(output_root, product.output)
    if not os.path.exists(outpath):
        logger.error(f'  {product.key} exited cleanly but did not write {outpath}')
        return False, elapsed

    size_mb = os.path.getsize(outpath) / 1e6
    logger.info(
        f'  wrote {outpath} ({size_mb:.2f} MB) in {_format_duration(seconds=elapsed)}'
    )
    return True, elapsed


def main() -> None:
    """
    Regenerate the cached notebook datasets and report the outcome.

    Raises
    ------
    SystemExit
        With a non-zero status if any script failed, so that the command can
        be used as a build step.
    """
    parser = argparse.ArgumentParser(
        description='Regenerate every cached dataset the notebooks read',
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run every script at sharply reduced parameters as a pipeline smoke test',
    )
    parser.add_argument(
        '--only', nargs='+', default=None, metavar='PATTERN',
        help='Run only products whose key contains one of these substrings',
    )
    parser.add_argument(
        '--skip', nargs='+', default=None, metavar='PATTERN',
        help='Skip products whose key contains one of these substrings',
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Leave products alone whose output file is already present',
    )
    parser.add_argument(
        '--fail-fast', action='store_true',
        help='Stop at the first failure instead of running the remaining products',
    )
    parser.add_argument(
        '--list', action='store_true',
        help='Print the product table and exit without running anything',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print the commands that would run without running them',
    )
    parser.add_argument(
        '--output-root', type=str, default='results',
        help='Directory holding the per-model output directories',
    )
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parse_args_compat(parser=parser)
    logger = setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file,
    )

    products = select_products(products=DATA_PRODUCTS, only=args.only, skip=args.skip)

    if args.list:
        for product in products:
            logger.info(f'{product.key:34s} -> {product.output}')
        return

    profile = 'quick' if args.quick else 'production'
    logger.info(f'Regenerating {len(products)} datasets ({profile} profile)')
    if args.quick:
        logger.warning('Quick profile: output exercises the pipeline but is not physics data.')

    failures: list[str] = []
    skipped: list[str] = []
    total_elapsed = 0.0

    for index, product in enumerate(products, start=1):
        outpath = os.path.join(args.output_root, product.output)
        if args.skip_existing and os.path.exists(outpath):
            logger.info(f'[{index}/{len(products)}] {product.key}: present, skipping')
            skipped.append(product.key)
            continue

        logger.info(f'[{index}/{len(products)}] {product.key}')
        ok, elapsed = run_product(
            product=product,
            output_root=args.output_root,
            quick=bool(args.quick),
            dry_run=bool(args.dry_run),
            logger=logger,
        )
        total_elapsed += elapsed
        if not ok:
            failures.append(product.key)
            if args.fail_fast:
                break

    produced = len(products) - len(failures) - len(skipped)
    logger.info(
        f'Done: {produced} produced, {len(skipped)} skipped, {len(failures)} failed '
        f'in {_format_duration(seconds=total_elapsed)}'
    )
    if failures:
        logger.error(f'Failed products: {", ".join(failures)}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
