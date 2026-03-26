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
) -> None:
    """Run an ordering kinetics simulation and save the two-panel kinetics figure.

    Quenches the model to ``temp``, steps forward to each of ``samples``
    logarithmically spaced targets up to ``max_steps``, measures growth
    scales and a model-specific third metric, fits power laws, and writes
    the figure to ``output_dir``.

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
    """
    _log = logger or logging.getLogger(__name__)

    step_targets = np.unique(
        np.logspace(0, np.log10(max_steps), num=samples).astype(int)
    )
    sim = model_cls(size=size, temp=temp, update='random', **model_kwargs)

    N_data = len(step_targets)
    t = np.zeros(N_data)
    R_sk = np.zeros(N_data)
    R_xi = np.zeros(N_data)
    third = np.zeros(N_data)

    current_step = 0
    for i, target in enumerate(tqdm(step_targets, bar_format=_BAR_FORMAT, desc='Simulating')):
        steps_to_run = int(target) - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = int(target)

        metrics = compute_kinetics_metrics(sim=sim)
        t[i] = float(current_step)
        R_sk[i] = metrics['R_sk']
        R_xi[i] = metrics['xi']
        third[i] = third_metric_fn(sim)
        _log.debug(
            f't={current_step}: R_sk={R_sk[i]:.2f}, xi={R_xi[i]:.2f}, third={third[i]:.4f}'
        )

    fit_mask = t >= fit_min
    exponents: dict[str, float | None] = {}
    prefactors: dict[str, float | None] = {}

    for key, data in [('R_sk', R_sk), ('xi', R_xi), ('third', third)]:
        exp, pre = power_fit(t_arr=t, y_arr=data, mask=fit_mask)
        exponents[key], prefactors[key] = exp, pre
        if exp:
            _log.info(f'{key} exponent: {exp:.3f}')

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
