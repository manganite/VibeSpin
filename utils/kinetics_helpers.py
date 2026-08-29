"""
Shared infrastructure for ordering kinetics analysis scripts.

Provides the simulation loop, power-law fitting, and result plotting
shared by the Ising, XY, and Clock ordering-kinetics scripts. The
``compute_mean_intercept_length`` function is also placed here as a
reusable domain-size estimator for scalar-spin models.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm import tqdm

from utils.observables import compute_kinetics_metrics
from utils.plotting import ensure_results_dir, plot_ordering_kinetics
from utils.statistics import power_fit

# tqdm bar format that always shows rate as iterations/s (never inverts to s/it).
_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]'


def compute_mean_intercept_length(sim: Any) -> float:
    """Estimate domain size via the stereological mean intercept length (MIL).

    Counts domain-wall crossings along every row and column of the lattice,
    including the periodic wrap-around crossing, then divides the total line
    length by the crossing count to obtain the mean intercept length.

    Parameters
    ----------
    sim : Any
        A simulation instance with a ``spins`` attribute (an integer-sign
        array) and a ``size`` property giving the linear lattice dimension.

    Returns
    -------
    float
        Estimated domain size in lattice units. Returns ``float(size)`` when
        there are no domain walls (fully ordered state).
    """
    if sim.spins is None:
        return 0.0
    spins = sim.spins.astype(np.int8)
    N = sim.size
    row_walls = np.sum(spins[:, :-1] * spins[:, 1:] < 0)
    col_walls = np.sum(spins[:-1, :] * spins[1:, :] < 0)
    row_wrap = int(np.sum(spins[:, -1] * spins[:, 0] < 0))
    col_wrap = int(np.sum(spins[-1, :] * spins[0, :] < 0))
    total_walls = int(row_walls) + int(col_walls) + row_wrap + col_wrap
    mean_walls = total_walls / (2 * N)
    return float(N) / mean_walls if mean_walls != 0.0 else float(N)


def _run_single_kinetics(
    *,
    model_cls: type,
    model_kwargs: dict[str, Any],
    third_metric_fn: Callable[[Any], float],
    size: int,
    temp: float,
    step_targets: np.ndarray,
    seed: int | None,
    logger: logging.Logger,
    desc: str = 'Simulating',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a single kinetics trajectory and return (t, R_sk, R_xi, third).

    Parameters
    ----------
    model_cls : type
        Simulation class to instantiate.
    model_kwargs : dict[str, Any]
        Extra keyword arguments beyond ``size``, ``temp``, ``update``.
    third_metric_fn : Callable[[Any], float]
        Model-specific third metric function.
    size : int
        Linear lattice size.
    temp : float
        Quench temperature.
    step_targets : np.ndarray
        Sorted array of MC step targets to measure at.
    seed : int or None
        Random seed for reproducibility.
    logger : logging.Logger
        Logger instance.
    desc : str
        Description for the progress bar.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays (t, R_sk, R_xi, third_metric).
    """
    seed_kwargs: dict[str, Any] = {}
    if seed is not None:
        seed_kwargs['seed'] = seed
    sim = model_cls(size=size, temp=temp, update='random', **model_kwargs, **seed_kwargs)

    N_data = len(step_targets)
    t = np.zeros(N_data)
    R_sk = np.zeros(N_data)
    R_xi = np.zeros(N_data)
    third = np.zeros(N_data)

    current_step = 0
    for i, target in enumerate(tqdm(step_targets, bar_format=_BAR_FORMAT, desc=desc)):
        steps_to_run = int(target) - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = int(target)

        metrics = compute_kinetics_metrics(sim=sim)
        t[i] = float(current_step)
        R_sk[i] = metrics['R_sk']
        R_xi[i] = metrics['xi']
        third[i] = third_metric_fn(sim)
        logger.debug(
            f't={current_step}: R_sk={R_sk[i]:.2f}, xi={R_xi[i]:.2f}, third={third[i]:.4f}'
        )

    return t, R_sk, R_xi, third


def run_ordering_kinetics(
    *,
    model_cls: type,
    model_kwargs: dict[str, Any],
    third_metric_fn: Callable[[Any], float],
    third_metric_label: str,
    title: str,
    left_title: str,
    right_title: str,
    size: int,
    temp: float,
    max_steps: int,
    samples: int,
    fit_min: int,
    output_dir: str,
    y_label: str = 'Characteristic Length Scale $L(t)$ (lattice units)',
    logger: logging.Logger | None = None,
    npz_path: str | None = None,
    n_seeds: int = 1,
    base_seed: int = 42,
) -> None:
    """Run an ordering kinetics simulation and save the two-panel kinetics figure.

    Quenches the model to ``temp``, steps forward to each of ``samples``
    logarithmically spaced targets up to ``max_steps``, measures growth
    scales and a model-specific third metric, fits power laws, and writes
    the figure to ``output_dir``.

    When ``n_seeds > 1``, the simulation is repeated for an ensemble of
    seeds.  The median trajectory is used for fitting and plotting, and
    per-seed arrays together with IQR-based error bars are saved to the
    NPZ file.

    Parameters
    ----------
    model_cls : type
        Simulation class to instantiate (e.g. ``IsingSimulation``).
    model_kwargs : dict[str, Any]
        Extra keyword arguments forwarded to the constructor beyond ``size``,
        ``temp``, and ``update='random'``.
    third_metric_fn : Callable[[Any], float]
        Called with the live simulation instance at each measurement step;
        returns a scalar third metric (e.g. vortex density or MIL).
    third_metric_label : str
        Y-axis label for the third-metric panel.
    title : str
        Figure-level suptitle.
    left_title : str
        Title for the left (growth-scale) subplot.
    right_title : str
        Title for the right (third-metric) subplot.
    size : int
        Linear lattice size L.
    temp : float
        Quench temperature T.
    max_steps : int
        Total MC steps to run.
    samples : int
        Number of logarithmically spaced measurement points.
    fit_min : int
        Minimum MC step used for power-law fitting.
    output_dir : str
        Directory in which to write the output figure.
    y_label : str, optional
        Y-axis label for the growth-scale panel.
    logger : logging.Logger, optional
        Logger to use; creates a module-level logger if not provided.
    npz_path : str, optional
        If given, save the kinetics data arrays to this ``.npz`` file for
        notebook consumption.
    n_seeds : int, optional
        Number of independent seeds to run.  Defaults to 1 (legacy
        single-trajectory behavior).
    base_seed : int, optional
        Starting seed; seeds are ``base_seed, base_seed+1, ...``.
    """
    _log = logger or logging.getLogger(__name__)

    step_targets = np.unique(
        np.logspace(0, np.log10(max_steps), num=samples).astype(int)
    )

    if n_seeds < 1:
        raise ValueError(f'n_seeds must be >= 1, got {n_seeds}')

    # --- Run ensemble ---
    N_data = len(step_targets)
    all_R_sk = np.zeros((n_seeds, N_data))
    all_R_xi = np.zeros((n_seeds, N_data))
    all_third = np.zeros((n_seeds, N_data))

    for s in range(n_seeds):
        seed = base_seed + s
        desc = f'Seed {s + 1}/{n_seeds}' if n_seeds > 1 else 'Simulating'
        t, R_sk_s, R_xi_s, third_s = _run_single_kinetics(
            model_cls=model_cls, model_kwargs=model_kwargs,
            third_metric_fn=third_metric_fn,
            size=size, temp=temp, step_targets=step_targets,
            seed=seed, logger=_log, desc=desc,
        )
        all_R_sk[s] = R_sk_s
        all_R_xi[s] = R_xi_s
        all_third[s] = third_s

    # Summary statistics: median for robustness against outlier seeds.
    R_sk = np.median(all_R_sk, axis=0)
    R_xi = np.median(all_R_xi, axis=0)
    third = np.median(all_third, axis=0)

    # IQR-based error: half the interquartile range as a robust ±1σ analogue.
    if n_seeds > 1:
        R_sk_err = 0.5 * (np.percentile(all_R_sk, 75, axis=0)
                          - np.percentile(all_R_sk, 25, axis=0))
        R_xi_err = 0.5 * (np.percentile(all_R_xi, 75, axis=0)
                          - np.percentile(all_R_xi, 25, axis=0))
        third_err = 0.5 * (np.percentile(all_third, 75, axis=0)
                           - np.percentile(all_third, 25, axis=0))
    else:
        R_sk_err = None
        R_xi_err = None
        third_err = None

    # --- Fit power laws to median trajectory ---
    fit_mask = t >= fit_min
    exponents: dict[str, float | None] = {}
    prefactors: dict[str, float | None] = {}

    for key, data in [('R_sk', R_sk), ('xi', R_xi), ('third', third)]:
        exp, pre = power_fit(t_arr=t, y_arr=data, mask=fit_mask)
        exponents[key], prefactors[key] = exp, pre
        if exp is not None:
            _log.info(f'{key} exponent: {exp:.3f}')

    # --- Save NPZ ---
    if npz_path is not None:
        _nan = float('nan')
        npz_data: dict[str, Any] = {
            # Legacy keys (single-trajectory shape, backward compatible).
            't': t,
            'R_sk': R_sk,
            'R_xi': R_xi,
            'third_metric': third,
            'third_metric_label': third_metric_label,
            'exponent_R_sk': exponents.get('R_sk') or _nan,
            'exponent_xi': exponents.get('xi') or _nan,
            'exponent_third': exponents.get('third') or _nan,
            'prefactor_R_sk': prefactors.get('R_sk') or _nan,
            'prefactor_xi': prefactors.get('xi') or _nan,
            'prefactor_third': prefactors.get('third') or _nan,
            'fit_min': fit_min,
            'size': size,
            'temp': temp,
            'n_seeds': n_seeds,
            'base_seed': base_seed,
        }
        # Additive ensemble keys (only when n_seeds > 1).
        if n_seeds > 1:
            npz_data['R_sk_err'] = R_sk_err
            npz_data['R_xi_err'] = R_xi_err
            npz_data['third_metric_err'] = third_err
            npz_data['R_sk_seeds'] = all_R_sk
            npz_data['R_xi_seeds'] = all_R_xi
            npz_data['third_metric_seeds'] = all_third

        np.savez_compressed(npz_path, **npz_data)
        _log.info(f'Kinetics data saved to {npz_path} (n_seeds={n_seeds})')

    # --- Plot ---
    plot_ordering_kinetics(
        t=t,
        R_sk=R_sk,
        R_xi=R_xi,
        third_metric=third,
        third_metric_label=third_metric_label,
        exponents=exponents,
        prefactors=prefactors,
        fit_mask=fit_mask,
        title=title,
        filename='ordering_kinetics.png',
        directory=ensure_results_dir(directory=output_dir),
        y_label=y_label,
        left_title=left_title,
        right_title=right_title,
    )
