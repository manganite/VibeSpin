"""
Shared command-line surface and orchestration for the temperature sweeps.

The Ising, XY, and Clock sweeps differ only in the model they instantiate, the
lattice and temperature window they default to, the transition overlay they
draw, and whether they prefer an ordered start below a known critical
temperature.  Everything between those choices, the argument surface, the
seed-retry collection loop, the uncertainty bundles, the NPZ schema, and the
figure, is identical and lives here, so that a change to the uncertainty
schema reaches all three models at once instead of drifting between them.
"""
from __future__ import annotations

import argparse
import logging
import os
from typing import Any, cast

import numpy as np

from utils.plotting import plot_temperature_sweep
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_BLOCKING,
    UNCERTAINTY_METHOD_BOOTSTRAP,
    summarize_entropy_observable,
)
from utils.sweep_helpers import (
    ThermoPoint,
    build_quality_flags,
    build_uncertainty_bundle,
    derive_point_seed,
    simulate_thermo_point,
    validate_sweep_uncertainty_args,
)
from utils.system import parallel_sweep, setup_logging

#: Observables carried through the full hierarchical uncertainty schema.  The
#: worker emits ``<key>_value``, ``<key>_err``, ``<key>_tau_int``, and
#: ``<key>_n_eff`` for each of them.
_BUNDLED_OBSERVABLES = ('avg_m', 'avg_e', 'susc', 'spec_h')


def add_temperature_sweep_arguments(
    *,
    parser: argparse.ArgumentParser,
    size: int,
    t_min: float,
    t_max: float,
    t_points: int,
    eq_max_steps: int,
    meas_steps: int,
    output_dir: str,
    transition_help: str,
) -> None:
    """
    Add the argument surface shared by all three temperature sweeps.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.  Model-specific arguments such as the clock state
        count are added by the caller, before or after this call.
    size : int
        Default linear lattice size L for this model.
    t_min : float
        Default lower end of the temperature window.
    t_max : float
        Default upper end of the temperature window.
    t_points : int
        Default number of temperature points.
    eq_max_steps : int
        Default hard cap on equilibration steps.  The vector models need a far
        higher cap than Ising because their relaxation is slower.
    meas_steps : int
        Default number of measurement sweeps per point.
    output_dir : str
        Default directory for the NPZ file and the figure.
    transition_help : str
        Model-specific help text for ``--transition-preset``, which names the
        overlay that ``auto`` draws.

    Returns
    -------
    None
        The parser is modified in place.
    """
    parser.add_argument('--size', type=int, default=size, help='Linear lattice size L')
    parser.add_argument(
        '--eq-probe-steps', type=int, default=500,
        help='Chunk size for convergence check during equilibration',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=eq_max_steps,
        help='Hard cap on total equilibration steps',
    )
    parser.add_argument(
        '--eq-qs-sigma-threshold', type=float, default=0.05,
        help='Tail-std threshold for quasi-steady stuck detection (0 disables)',
    )
    parser.add_argument(
        '--eq-qs-min-steps', type=int, default=1500,
        help='Minimum accumulated equilibration steps before stuck detection can fire',
    )
    parser.add_argument('--meas-steps', type=int, default=meas_steps, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=t_min, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=t_max, help='Maximum temperature')
    parser.add_argument(
        '--t-points', type=int, default=t_points, help='Number of temperature points',
    )
    parser.add_argument(
        '--n-seeds',
        type=int,
        default=5,
        help='Target number of fully converged seed replicas to retain',
    )
    parser.add_argument(
        '--max-seed-attempts',
        type=int,
        default=None,
        help='Per-temperature cap on seed attempts, including replacements; defaults to 10*n-seeds',
    )
    parser.add_argument(
        '--derived-uncertainty-method',
        type=str,
        default=UNCERTAINTY_METHOD_BLOCKING,
        choices=[UNCERTAINTY_METHOD_BLOCKING, UNCERTAINTY_METHOD_BOOTSTRAP],
        help='Uncertainty method for susceptibility and specific heat',
    )
    parser.add_argument(
        '--derived-bootstrap-resamples',
        type=int,
        default=0,
        help='Bootstrap resamples used when derived-uncertainty-method=bootstrap',
    )
    parser.add_argument(
        '--confidence-level',
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help='Two-sided confidence level for uncertainty intervals (0 < c < 1)',
    )
    parser.add_argument(
        '--entropy-uncertainty-method',
        type=str,
        default=UNCERTAINTY_METHOD_BOOTSTRAP,
        choices=[UNCERTAINTY_METHOD_BLOCKING, UNCERTAINTY_METHOD_BOOTSTRAP],
        help='Uncertainty method for entropy curves',
    )
    parser.add_argument(
        '--entropy-bootstrap-resamples',
        type=int,
        default=400,
        help='Bootstrap resamples used when entropy-uncertainty-method=bootstrap',
    )
    parser.add_argument(
        '--strict-uncertainty',
        action='store_true',
        help='Fail if undefined autocorrelation points exceed the allowed fraction',
    )
    parser.add_argument(
        '--max-undefined-fraction',
        type=float,
        default=0.25,
        help='Maximum allowed fraction of undefined tau_int points in strict mode',
    )
    parser.add_argument(
        '--min-effective-samples',
        type=float,
        default=20.0,
        help='Threshold used to flag low effective sample size points',
    )
    parser.add_argument(
        '--max-tau-relative-width',
        type=float,
        default=1.0,
        help='Relative width threshold to flag unstable tau intervals',
    )
    parser.add_argument(
        '--transition-preset',
        type=str,
        default='auto',
        choices=['auto', 'none', 'theory'],
        help=transition_help,
    )
    parser.add_argument('--output-dir', type=str, default=output_dir, help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')


def _collect_seed_ensemble(
    *,
    args: argparse.Namespace,
    temperatures: np.ndarray,
    model_cls: type,
    model_kwargs: dict[str, Any],
    ordered_start_below: float | None,
    target_n_seeds: int,
    max_seed_attempts: int,
) -> list[list[dict[str, float]]]:
    """
    Run the sweep until every temperature holds the target number of replicas.

    Each pass submits the points that are still short of the target, keeps the
    results that reached equilibrium, and repeats.  A temperature that keeps
    failing to converge is abandoned once it has used its attempt budget, so a
    single pathological point cannot stall the whole sweep.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed sweep arguments.
    temperatures : numpy.ndarray
        Temperature grid of the sweep.
    model_cls : type
        Simulation class to instantiate for every point.
    model_kwargs : dict[str, typing.Any]
        Extra constructor arguments beyond size, temperature, start, and seed.
    ordered_start_below : float or None
        Temperature under which a quasi-steady stuck state counts as converged
        and the ordered start may replace the random one.  None disables both.
    target_n_seeds : int
        Number of converged replicas wanted per temperature.
    max_seed_attempts : int
        Per-temperature cap on attempts including replacements.

    Returns
    -------
    list[list[dict[str, float]]]
        Converged worker results, indexed by temperature then by replica.
    """
    t_points = len(temperatures)
    results_grid: list[list[dict[str, float]]] = [[] for _ in range(t_points)]
    attempts_per_temp = np.zeros(t_points, dtype=int)

    while any(len(results_grid[t]) < target_n_seeds for t in range(t_points)):
        points_to_calculate = []
        for t in range(t_points):
            needed = target_n_seeds - len(results_grid[t])
            if needed <= 0 or attempts_per_temp[t] >= max_seed_attempts:
                continue
            for _ in range(needed):
                seed_idx = int(attempts_per_temp[t])
                t_val = float(temperatures[t])
                low_t = ordered_start_below is not None and t_val < ordered_start_below
                points_to_calculate.append(ThermoPoint(
                    temperature=t_val,
                    size=int(args.size),
                    meas_steps=int(args.meas_steps),
                    eq_probe_steps=int(args.eq_probe_steps),
                    eq_max_steps=int(args.eq_max_steps),
                    eq_qs_sigma_threshold=float(args.eq_qs_sigma_threshold),
                    eq_qs_min_steps=int(args.eq_qs_min_steps),
                    qs_allow_stuck=low_t,
                    prefer_ordered_start=low_t,
                    temperature_index=t,
                    seed_index=seed_idx,
                    seed=derive_point_seed(temperature_index=t, seed_index=seed_idx),
                    model_cls=model_cls,
                    model_kwargs=model_kwargs,
                    confidence=float(args.confidence_level),
                    derived_method=str(args.derived_uncertainty_method),
                    bootstrap_resamples=int(args.derived_bootstrap_resamples),
                ))
                attempts_per_temp[t] += 1

        if not points_to_calculate:
            break

        for res in parallel_sweep(
            worker_func=simulate_thermo_point,
            params=points_to_calculate,
        ):
            if res.get('equilibrated_flag', 0.0) > 0:
                results_grid[int(res['temperature_index'])].append(res)

    return results_grid


def run_temperature_sweep(
    *,
    args: argparse.Namespace,
    model_cls: type,
    model_kwargs: dict[str, Any],
    model_label: str,
    plot_title: str,
    metadata_note: str,
    transition_temperatures: dict[str, float] | None,
    variant_note: str = '',
    ordered_start_below: float | None = None,
) -> None:
    """
    Execute a temperature sweep and write its NPZ file and figure.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from a parser extended by
        ``add_temperature_sweep_arguments``.
    model_cls : type
        Simulation class to instantiate for every point.  It must be importable
        under its qualified name so that it survives the multiprocessing pickle.
    model_kwargs : dict[str, typing.Any]
        Extra constructor arguments beyond size, temperature, start, and seed.
    model_label : str
        Model name used in log messages, for example ``'Ising'``.
    plot_title : str
        Title of the thermodynamics figure.
    metadata_note : str
        Run provenance line printed under the figure.
    transition_temperatures : dict[str, float] or None
        Vertical overlays to draw, already resolved against
        ``--transition-preset`` by the caller, since which overlay is
        meaningful depends on the model.
    variant_note : str, optional
        Extra model configuration to mention in the opening log line, for
        example the clock variant and anisotropy.
    ordered_start_below : float, optional
        Temperature under which a quasi-steady stuck state counts as converged
        and the ordered start may replace the random one.  Set it to the
        critical temperature for models with true long-range order below it;
        leave it None where the low-temperature phase has none.

    Returns
    -------
    None
        Results are written to ``args.output_dir``.

    Raises
    ------
    RuntimeError
        If no temperature produced a converged replica, or if strict
        uncertainty mode is on and too many points have undefined
        autocorrelation.
    """
    validate_sweep_uncertainty_args(
        confidence_level=float(args.confidence_level),
        max_undefined_fraction=float(args.max_undefined_fraction),
        min_effective_samples=float(args.min_effective_samples),
        max_tau_relative_width=float(args.max_tau_relative_width),
        derived_uncertainty_method=str(args.derived_uncertainty_method),
        derived_bootstrap_resamples=int(args.derived_bootstrap_resamples),
        n_seeds=int(args.n_seeds),
    )

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)
    t_points = len(temperatures)

    target_n_seeds = int(args.n_seeds)
    max_seed_attempts = int(
        10 * target_n_seeds if args.max_seed_attempts is None else args.max_seed_attempts
    )

    logger.info(
        f'Starting {model_label} temperature sweep (L={args.size}, '
        f'{variant_note}target_n_seeds={target_n_seeds}, '
        f'max_seed_attempts={max_seed_attempts})...'
    )

    results_grid = _collect_seed_ensemble(
        args=args,
        temperatures=temperatures,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        ordered_start_below=ordered_start_below,
        target_n_seeds=target_n_seeds,
        max_seed_attempts=max_seed_attempts,
    )

    missing = sum(max(0, target_n_seeds - len(results_grid[t])) for t in range(t_points))
    if missing:
        logger.warning(
            f'Could not reach target_n_seeds={target_n_seeds} for all temperatures. '
            f'Missing points: {missing}'
        )

    # Temperatures that produced no converged replica at all are dropped rather
    # than carried as an all-NaN row, so downstream fits see only real points.
    valid_t_mask = np.array([len(results_grid[t]) > 0 for t in range(t_points)])
    valid_temperatures = temperatures[valid_t_mask]
    n_valid_t = int(valid_t_mask.sum())
    if n_valid_t == 0:
        raise RuntimeError('Failed to collect any converged results.')

    # Replica counts can differ between temperatures once attempts are
    # exhausted, so the grid is padded with NaN to a common width.
    max_seeds_retained = max(len(r) for r in results_grid)

    def extract_grid(key: str) -> np.ndarray:
        arr = np.full((n_valid_t, max_seeds_retained), np.nan)
        valid_idx = 0
        for t in range(t_points):
            if not valid_t_mask[t]:
                continue
            for s, res in enumerate(results_grid[t]):
                arr[valid_idx, s] = res.get(key, np.nan)
            valid_idx += 1
        return arr

    bundles = {
        name: build_uncertainty_bundle(
            values_by_seed=extract_grid(f'{name}_value'),
            errors_by_seed=extract_grid(f'{name}_err'),
            tau_by_seed=extract_grid(f'{name}_tau_int'),
            n_eff_by_seed=extract_grid(f'{name}_n_eff'),
            confidence=float(args.confidence_level),
        )
        for name in _BUNDLED_OBSERVABLES
    }
    mag_bundle = bundles['avg_m']
    spec_h_bundle = bundles['spec_h']

    entropy_res = summarize_entropy_observable(
        temperatures=valid_temperatures,
        specific_heat_samples=spec_h_bundle['samples'],
        specific_heat_err=spec_h_bundle['err'],
        method=str(args.entropy_uncertainty_method),
        confidence=float(args.confidence_level),
        bootstrap_resamples=int(args.entropy_bootstrap_resamples),
    )

    quality = build_quality_flags(
        tau_int=mag_bundle['tau_int'],
        ci_low=mag_bundle['ci_low'],
        ci_high=mag_bundle['ci_high'],
        n_eff=mag_bundle['n_eff'],
        min_effective_samples=float(args.min_effective_samples),
        max_tau_relative_width=float(args.max_tau_relative_width),
    )

    if args.strict_uncertainty:
        undef_frac = float(np.mean(quality['undefined_autocorr_flag']))
        if undef_frac > args.max_undefined_fraction:
            raise RuntimeError(
                f'Strict uncertainty failed: {undef_frac:.1%} of points have '
                'undefined autocorrelation.'
            )

    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, 'temperature_sweep_data.npz')
    np.savez(
        outpath,
        temperatures=valid_temperatures,
        # Legacy scalar arrays, kept for notebooks that predate the schema.
        avg_m=mag_bundle['value'],
        avg_e=bundles['avg_e']['value'],
        susc=bundles['susc']['value'],
        spec_h=spec_h_bundle['value'],
        entropy=entropy_res['value'],
        tau_int=mag_bundle['tau_int'],
        # Full uncertainty schema.
        **cast(Any, {
            f'{name}_{k}': v
            for name, bundle in bundles.items()
            for k, v in bundle.items()
        }),
        **cast(Any, {f'entropy_{k}': v for k, v in entropy_res.items()}),
        **cast(Any, quality),
        entropy_uncertainty_method=str(args.entropy_uncertainty_method),
        uncertainty_method=str(args.derived_uncertainty_method),
        confidence_level=float(args.confidence_level),
        n_seeds=max_seeds_retained,
        size=int(args.size),
        meas_steps=int(args.meas_steps),
        bootstrap_resamples=int(args.derived_bootstrap_resamples),
        requested_n_seeds=target_n_seeds,
        max_seed_attempts=max_seed_attempts,
        retained_n_seeds=max_seeds_retained,
        nan_or_undefined_count=int(np.sum(quality['undefined_autocorr_flag'])),
    )
    logger.info(f'Saved sweep data to {outpath}')

    plot_temperature_sweep(
        temperatures=valid_temperatures,
        avg_m=cast(Any, mag_bundle['value']),
        avg_m_err=mag_bundle['err'],
        avg_e=cast(Any, bundles['avg_e']['value']),
        avg_e_err=bundles['avg_e']['err'],
        susc=cast(Any, bundles['susc']['value']),
        susc_err=bundles['susc']['err'],
        spec_h=cast(Any, spec_h_bundle['value']),
        spec_h_err=spec_h_bundle['err'],
        entropy=entropy_res['value'],
        entropy_err=entropy_res['err'],
        entropy_ci_low=entropy_res['ci_low'],
        entropy_ci_high=entropy_res['ci_high'],
        tau_int=mag_bundle['tau_int'],
        tau_unstable_flag=quality['tau_interval_unstable_flag'],
        low_effective_sample_flag=quality['low_effective_sample_flag'],
        title=plot_title,
        filename='temperature_sweep.png',
        directory=args.output_dir,
        run_metadata_note=metadata_note,
        transition_temperatures=transition_temperatures,
    )
