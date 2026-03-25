"""
Shared infrastructure for ordering evolution visualisation scripts.

Provides the snapshot loop and figure generation shared by the Ising, XY,
and Clock ordering-evolution scripts.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from utils.plotting import ensure_results_dir, plot_ordering_evolution


def run_ordering_evolution(
    *,
    model_cls: type,
    model_kwargs: dict[str, Any],
    capture_vorticity: bool,
    title: str,
    size: int,
    temp: float,
    step_targets: list[int],
    output_dir: str,
    logger: logging.Logger | None = None,
) -> None:
    """Run an ordering evolution simulation and save the multi-panel figure.

    Quenches the model to ``temp``, advances to each step in
    ``step_targets``, captures spin configurations and correlation functions,
    optionally captures vorticity maps, and writes the figure to
    ``output_dir``.

    Parameters
    ----------
    model_cls : type
        Simulation class to instantiate (e.g. ``XYSimulation``).
    model_kwargs : dict[str, Any]
        Extra keyword arguments forwarded to the constructor beyond ``size``,
        ``temp``, and ``update='random'``.
    capture_vorticity : bool
        Whether to record vorticity maps at each snapshot. Pass ``True`` for
        vector-spin models (XY, Clock) and ``False`` for scalar models
        (Ising). Controls ``is_vector`` in the output figure.
    title : str
        Figure-level suptitle.
    size : int
        Linear lattice size L.
    temp : float
        Quench temperature T.
    step_targets : list[int]
        MC step counts at which to capture snapshots. Need not be sorted;
        the function sorts them internally.
    output_dir : str
        Directory in which to write the output figure.
    logger : logging.Logger, optional
        Logger to use; creates a module-level logger if not provided.
    """
    _log = logger or logging.getLogger(__name__)

    sim = model_cls(size=size, temp=temp, update='random', **model_kwargs)
    targets = sorted(step_targets)
    n_targets: int = len(targets)

    snapshots: list[np.ndarray] = []
    snapshots_vort: list[np.ndarray] = []
    snapshots_gr: list[tuple[np.ndarray, np.ndarray]] = []

    current_step: int = 0
    for target in targets:
        steps_to_run = target - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = target

        if sim.spins is not None:
            snapshots.append(sim.spins.copy())
            snapshots_gr.append(sim._calculate_correlation_function())
            if capture_vorticity:
                snapshots_vort.append(sim._calculate_vorticity())
                _log.debug(
                    f'Captured snapshot at step {target}'
                    f' (n_v={sim._get_vortex_density():.4f})'
                )
            else:
                _log.debug(f'Captured snapshot at step {target}')

    _log.info(f'Collected {n_targets} snapshots. Saving figure ...')

    plot_ordering_evolution(
        targets=targets,
        snapshots=snapshots,
        gr_data=snapshots_gr,
        vorticity_data=snapshots_vort if capture_vorticity else None,
        title=title,
        filename='ordering_evolution.png',
        directory=ensure_results_dir(directory=output_dir),
        is_vector=capture_vorticity,
    )
