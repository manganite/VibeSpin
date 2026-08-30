"""
Shared machinery for the Wolff-versus-Metropolis efficiency comparison.

The comparison asks the same question of every model here: at a given
temperature, how many statistically independent configurations per second of
wall-clock time does a local update deliver, and how many does the cluster
update deliver?  Nothing in that measurement is model-specific except the
simulation class and where the interesting temperature window lies, so the
worker, the figure, and the argument surface live here and each model script
supplies only its own physics.

Comparing raw autocorrelation times across the two algorithms would be
meaningless, because one Wolff step touches a cluster and one Metropolis sweep
touches the whole lattice.  The work-normalised time multiplies the Wolff
autocorrelation time by the mean cluster fraction, putting both on a
lattice-sweep footing; the independent-samples-per-second figure sidesteps the
question entirely by measuring against the clock.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np

from utils.equilibration import convergence_equilibrate
from utils.exceptions import ZeroVarianceAutocorrelationError
from utils.observables import calculate_thermodynamics
from utils.statistics import (
    DEFAULT_CONFIDENCE_LEVEL,
    UNCERTAINTY_METHOD_REPLICATE,
    calculate_autocorr,
    summarize_replicate_samples,
)
from utils.sweep_helpers import derive_point_seed
from utils.system import parallel_sweep, setup_logging

#: Quantities the worker reports per (temperature, seed) point.
_MEASURED_KEYS = (
    'tau_metro', 'tau_wolff', 'iss_metro', 'iss_wolff',
    'mean_cluster_frac', 'chi_metro', 'chi_wolff',
)

#: Cluster sizes are measured in a separate short pass so that the timing runs
#: stay free of the per-step bookkeeping the cluster record would add.
_CLUSTER_PASS_STEPS = 300


class EfficiencyPoint(NamedTuple):
    """
    Typed worker payload for one (temperature, seed) efficiency measurement.

    Parameters
    ----------
    temp_idx : int
        Index of this temperature in the sweep grid.
    seed_idx : int
        Replica index within the seed ensemble for this temperature.
    temperature : float
        Simulation temperature T.
    size : int
        Linear lattice dimension L.
    eq_probe_steps : int
        Chunk size for the two-start convergence check.
    eq_max_steps : int
        Hard cap on equilibration steps.
    meas_steps : int
        Timed measurement steps per algorithm.
    model_cls : type
        Simulation class to instantiate.  It must be importable under its
        qualified name so that it survives the multiprocessing pickle.
    model_kwargs : dict[str, typing.Any]
        Extra constructor arguments beyond size, temperature, update, start,
        and seed.
    """

    temp_idx: int
    seed_idx: int
    temperature: float
    size: int
    eq_probe_steps: int
    eq_max_steps: int
    meas_steps: int
    model_cls: type
    model_kwargs: dict[str, Any]


def _equilibrated_pair(*, point: EfficiencyPoint, update: str, seed: int) -> Any:
    """
    Build a two-start pair, equilibrate it, and return the random-start run.

    Parameters
    ----------
    point : EfficiencyPoint
        Payload describing the measurement.
    update : str
        Update scheme for both simulations.
    seed : int
        Seed shared by both starts of this pair.

    Returns
    -------
    typing.Any
        The equilibrated random-start simulation, ready to be measured.
    """
    common = dict(
        size=point.size, temp=point.temperature, update=update, **point.model_kwargs
    )
    sim_random = point.model_cls(init_state='random', seed=seed, **common)
    sim_ordered = point.model_cls(init_state='ordered', seed=seed, **common)
    convergence_equilibrate(
        sim_random=sim_random,
        sim_ordered=sim_ordered,
        chunk_size=point.eq_probe_steps,
        max_steps=point.eq_max_steps,
    )
    return sim_random


def _timed_measurement(
    *, sim: Any, meas_steps: int, temperature: float, size: int,
) -> tuple[float, float, float]:
    """
    Run a timed measurement pass and reduce it to the efficiency quantities.

    Parameters
    ----------
    sim : typing.Any
        Equilibrated simulation to measure.
    meas_steps : int
        Number of steps to run and time.
    temperature : float
        Temperature, needed for the susceptibility.
    size : int
        Linear lattice dimension, needed for the susceptibility.

    Returns
    -------
    tuple[float, float, float]
        Integrated autocorrelation time, independent samples per second, and
        magnetic susceptibility.  The first two are NaN where the
        magnetisation series has no usable variance.
    """
    started = time.perf_counter()
    mags, engs = sim.run(n_steps=meas_steps)
    elapsed = time.perf_counter() - started

    mags_arr = np.asarray(mags)
    engs_arr = np.asarray(engs)
    try:
        _, tau_int = calculate_autocorr(time_series=mags_arr)
    except ZeroVarianceAutocorrelationError:
        tau_int = float('nan')

    _, _, chi, _ = calculate_thermodynamics(
        mags=mags_arr, engs=engs_arr, T=temperature, L=size,
    )
    iss = (
        (meas_steps / elapsed) / tau_int
        if np.isfinite(tau_int) and tau_int > 0
        else float('nan')
    )
    return tau_int, iss, chi


def measure_efficiency_point(point: EfficiencyPoint) -> dict[str, float]:
    """
    Measure local-update and cluster-update efficiency at one grid point.

    Both algorithms start from their own independently equilibrated state, so
    that neither inherits the other's correlations, and each is timed over the
    same number of its own steps.

    Parameters
    ----------
    point : EfficiencyPoint
        Payload describing the measurement.

    Returns
    -------
    dict[str, float]
        Grid indices, temperature, and the seven measured quantities.
    """
    seed = derive_point_seed(
        temperature_index=point.temp_idx, seed_index=point.seed_idx,
    )

    sim_metro = _equilibrated_pair(point=point, update='checkerboard', seed=seed)
    tau_metro, iss_metro, chi_metro = _timed_measurement(
        sim=sim_metro, meas_steps=point.meas_steps,
        temperature=point.temperature, size=point.size,
    )

    sim_wolff = _equilibrated_pair(point=point, update='wolff', seed=seed + 1)
    tau_wolff, iss_wolff, chi_wolff = _timed_measurement(
        sim=sim_wolff, meas_steps=point.meas_steps,
        temperature=point.temperature, size=point.size,
    )

    sim_cluster = _equilibrated_pair(point=point, update='wolff', seed=seed + 2)
    _, _, cluster_sizes = sim_cluster.run_with_cluster_sizes(
        n_steps=min(point.meas_steps, _CLUSTER_PASS_STEPS),
    )
    mean_cluster_frac = float(np.mean(cluster_sizes)) / float(point.size**2)

    return {
        'temp_idx': float(point.temp_idx),
        'seed_idx': float(point.seed_idx),
        'T': float(point.temperature),
        'tau_metro': tau_metro,
        'tau_wolff': tau_wolff,
        'iss_metro': iss_metro,
        'iss_wolff': iss_wolff,
        'mean_cluster_frac': mean_cluster_frac,
        'chi_metro': chi_metro,
        'chi_wolff': chi_wolff,
    }


def _summarize(*, samples: np.ndarray) -> dict[str, np.ndarray]:
    """Reduce a (temperature, seed) sample grid to a median and a 16-84 band."""
    summary = summarize_replicate_samples(samples=samples)
    return {
        'value': np.asarray(summary['value']),
        'p16': np.asarray(summary['ci_low']),
        'p84': np.asarray(summary['ci_high']),
    }


def _band(
    *, ax: plt.Axes, temperatures: np.ndarray, summary: dict[str, np.ndarray],
    color: str, marker: str, label: str | None, log: bool,
) -> None:
    """Draw one median curve with its percentile band on the given axes."""
    plot = ax.semilogy if log else ax.plot
    plot(temperatures, summary['value'], marker, color=color, ms=4, label=label)
    ax.fill_between(
        temperatures, summary['p16'], summary['p84'], color=color, alpha=0.15,
    )


def plot_efficiency(
    *,
    temperatures: np.ndarray,
    summaries: dict[str, dict[str, np.ndarray]],
    title: str,
    directory: str,
    transitions: dict[str, float] | None,
) -> None:
    """
    Produce and save the four-panel efficiency comparison figure.

    Parameters
    ----------
    temperatures : numpy.ndarray
        Sorted temperature grid.
    summaries : dict[str, dict[str, numpy.ndarray]]
        Median and band for each plotted quantity, keyed by
        ``tau_metro_norm``, ``tau_wolff_norm``, ``iss_metro``, ``iss_wolff``,
        ``mean_cluster_frac``, ``chi_metro``, and ``chi_wolff``.
    title : str
        Figure suptitle.
    directory : str
        Output directory for the saved PNG.
    transitions : dict[str, float] or None
        Vertical reference lines to draw in every panel, keyed by legend
        label, or None to omit them.

    Returns
    -------
    None
        The figure is written to ``wolff_efficiency.png`` in ``directory``.
    """
    metro, wolff = '#4878CF', '#D65F5F'
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(title, fontsize=13)

    def mark_transition(ax: plt.Axes) -> None:
        for offset, (label, temperature) in enumerate(sorted((transitions or {}).items())):
            ax.axvline(
                temperature, color='0.4', ls='--' if offset == 0 else ':', lw=1, label=label,
            )

    ax = axes[0, 0]
    _band(ax=ax, temperatures=temperatures, summary=summaries['tau_metro_norm'],
          color=metro, marker='-o', label='Metropolis', log=True)
    _band(ax=ax, temperatures=temperatures, summary=summaries['tau_wolff_norm'],
          color=wolff, marker='-s', label='Wolff (normalised)', log=True)
    mark_transition(ax)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\tau^{\mathrm{norm}}_{\mathrm{int}}$ ($L^2$-sweep equiv.)')
    ax.set_title('Integrated autocorrelation time\n(work-normalised; median with 16–84% band)')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    _band(ax=ax, temperatures=temperatures, summary=summaries['iss_metro'],
          color=metro, marker='-o', label='Metropolis', log=True)
    _band(ax=ax, temperatures=temperatures, summary=summaries['iss_wolff'],
          color=wolff, marker='-s', label='Wolff', log=True)
    mark_transition(ax)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('Independent samples / s')
    ax.set_title('Sampling efficiency ISS\n(median with 16–84% band)')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    _band(ax=ax, temperatures=temperatures, summary=summaries['mean_cluster_frac'],
          color=wolff, marker='-^', label=None, log=False)
    mark_transition(ax)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\langle C \rangle \,/\, N^2$')
    ax.set_title(
        'Mean cluster size fraction (Wolff)\n'
        r'= normalisation factor $\langle C\rangle/L^2$',
    )
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    _band(ax=ax, temperatures=temperatures, summary=summaries['chi_metro'],
          color=metro, marker='-o', label='Metropolis', log=False)
    _band(ax=ax, temperatures=temperatures, summary=summaries['chi_wolff'],
          color=wolff, marker='-s', label='Wolff', log=False)
    mark_transition(ax)
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\chi$')
    ax.set_title('Susceptibility (agreement validates correctness)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(directory, exist_ok=True)
    fig.savefig(os.path.join(directory, 'wolff_efficiency.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def add_wolff_efficiency_arguments(
    *,
    parser: argparse.ArgumentParser,
    size: int,
    t_min: float,
    t_max: float,
    output_dir: str,
) -> None:
    """
    Add the argument surface shared by the per-model efficiency scripts.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    size : int
        Default linear lattice size L.
    t_min : float
        Default lower end of the temperature window.
    t_max : float
        Default upper end of the temperature window.
    output_dir : str
        Default directory for the NPZ file and the figure.

    Returns
    -------
    None
        The parser is modified in place.
    """
    parser.add_argument('--size', type=int, default=size, help='Lattice size L')
    parser.add_argument(
        '--eq-probe-steps', type=int, default=500,
        help='Chunk size for convergence check during equilibration',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=200_000,
        help='Hard cap on equilibration steps',
    )
    parser.add_argument(
        '--meas-steps', type=int, default=2000,
        help='Measurement steps per algorithm per temperature point',
    )
    parser.add_argument('--t-min', type=float, default=t_min, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=t_max, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=20, help='Temperature grid points')
    parser.add_argument(
        '--n-seeds', type=int, default=10,
        help='Independent seed replicas per temperature point',
    )
    parser.add_argument('--output-dir', type=str, default=output_dir, help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')


def run_wolff_efficiency(
    *,
    args: argparse.Namespace,
    model_cls: type,
    model_kwargs: dict[str, Any],
    model_label: str,
    transitions: dict[str, float] | None,
) -> None:
    """
    Run the efficiency comparison and write its NPZ file and figure.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments parsed from a parser extended by
        ``add_wolff_efficiency_arguments``.
    model_cls : type
        Simulation class to compare the two update schemes on.
    model_kwargs : dict[str, typing.Any]
        Extra constructor arguments beyond size, temperature, update, start,
        and seed.
    model_label : str
        Model name for the figure title and the NPZ metadata.
    transitions : dict[str, float] or None
        Temperatures to mark in every panel, keyed by legend label, or None to
        omit the markers.  The first by label is drawn dashed, the rest dotted.

    Returns
    -------
    None
        Results are written to ``args.output_dir``.
    """
    logger = setup_logging(
        level=logging.DEBUG if args.verbose else logging.INFO,
        log_file=args.log_file,
    )

    size = int(args.size)
    n_seeds = int(args.n_seeds)
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)

    logger.info(
        f'{model_label} Wolff efficiency: L={size}, '
        f'T in [{args.t_min:.2f}, {args.t_max:.2f}], {args.t_points} points, '
        f'{n_seeds} seed replicas, {args.meas_steps} meas steps.'
    )

    points = [
        EfficiencyPoint(
            temp_idx=temp_idx,
            seed_idx=seed_idx,
            temperature=float(temperature),
            size=size,
            eq_probe_steps=int(args.eq_probe_steps),
            eq_max_steps=int(args.eq_max_steps),
            meas_steps=int(args.meas_steps),
            model_cls=model_cls,
            model_kwargs=model_kwargs,
        )
        for temp_idx, temperature in enumerate(temperatures)
        for seed_idx in range(n_seeds)
    ]
    raw = parallel_sweep(worker_func=measure_efficiency_point, params=points)

    samples = {
        key: np.full((len(temperatures), n_seeds), np.nan) for key in _MEASURED_KEYS
    }
    for record in raw:
        i, s = int(record['temp_idx']), int(record['seed_idx'])
        for key in _MEASURED_KEYS:
            samples[key][i, s] = float(record[key])

    # One Wolff step touches a cluster while one Metropolis sweep touches the
    # whole lattice, so the Wolff autocorrelation time is scaled by the mean
    # cluster fraction to put both algorithms on a lattice-sweep footing.
    samples['tau_metro_norm'] = samples['tau_metro']
    samples['tau_wolff_norm'] = samples['tau_wolff'] * samples['mean_cluster_frac']

    summaries = {key: _summarize(samples=array) for key, array in samples.items()}

    marked = transitions or {}
    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, 'wolff_efficiency.npz')
    np.savez(
        npz_path,
        temperatures=temperatures,
        n_seeds=np.int64(n_seeds),
        uncertainty_method=UNCERTAINTY_METHOD_REPLICATE,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        nan_or_undefined_count=int(sum(
            int(np.isnan(samples[key]).sum()) for key in _MEASURED_KEYS
        )),
        model=model_label,
        transition_labels=np.array(sorted(marked)),
        transition_temperatures=np.array([marked[k] for k in sorted(marked)], dtype=float),
        meas_steps=np.int64(args.meas_steps),
        L=np.int64(size),
        size=np.int64(size),
        **cast(Any, {f'{key}_samples': samples[key] for key in _MEASURED_KEYS}),
        **cast(Any, {key: summaries[key]['value'] for key in _MEASURED_KEYS}),
        **cast(Any, {f'{key}_p16': summaries[key]['p16'] for key in _MEASURED_KEYS}),
        **cast(Any, {f'{key}_p84': summaries[key]['p84'] for key in _MEASURED_KEYS}),
    )
    logger.info(f'Data saved to {npz_path}')

    plot_efficiency(
        temperatures=temperatures,
        summaries=summaries,
        title=(
            f'Wolff vs. Metropolis Efficiency — {model_label}  '
            f'(L = {size}, seeds = {n_seeds})'
        ),
        directory=args.output_dir,
        transitions=transitions,
    )
