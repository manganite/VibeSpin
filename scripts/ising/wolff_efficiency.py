"""
Wolff cluster algorithm efficiency demonstration for the 2D Ising model.

Compares integrated autocorrelation time (tau_int), independent samples per
second (ISS), mean cluster size fraction, and susceptibility between the
Metropolis checkerboard and Wolff cluster algorithms across a temperature range
centred on the critical point T_c ~= 2.269.

Results are saved to ``results/ising/wolff_efficiency.npz`` for notebook
re-use and ``results/ising/wolff_efficiency.png`` as a 4-panel figure.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from models.ising_model import IsingSimulation
from utils.exceptions import ZeroVarianceAutocorrelationError
from utils.physics_helpers import calculate_autocorr, calculate_thermodynamics
from utils.system_helpers import adaptive_equilibrate, parallel_sweep, setup_logging

#: Exact Onsager critical temperature for the 2D nearest-neighbour Ising model.
TC_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))


def _measure_efficiency_point(
    params: tuple[float, int, int, int, int],
) -> dict[str, float]:
    """
    Worker: measure algorithmic efficiency at one temperature point.

    Runs the Metropolis checkerboard and Wolff cluster algorithms from
    independent equilibrated states at temperature *T*.  For each algorithm
    the worker records wall-clock time for *meas_steps* steps via
    ``sim.run()``, computes tau_int from the magnetisation series, and derives
    ISS = (steps / wall_time) / tau_int.  Cluster sizes are measured in a
    separate, short Wolff pass to keep the timing paths clean.

    Parameters
    ----------
    params : tuple
        ``(T, L, eq_steps, meas_steps, seed)`` — temperature, lattice size,
        minimum equilibration steps, measurement steps, and base RNG seed.

    Returns
    -------
    dict
        Keys: ``T``, ``tau_metro``, ``tau_wolff``, ``iss_metro``,
        ``iss_wolff``, ``mean_cluster_frac``, ``chi_metro``, ``chi_wolff``.
    """
    T, L, eq_steps, meas_steps, seed = params

    # ---- Metropolis checkerboard ----
    sim_m = IsingSimulation(size=L, temp=T, update='checkerboard', seed=seed)
    adaptive_equilibrate(sim_m, min_steps=eq_steps)
    t0 = time.perf_counter()
    mags_m, engs_m = sim_m.run(n_steps=meas_steps)
    t_metro = time.perf_counter() - t0
    mags_m_arr = np.array(mags_m)
    engs_m_arr = np.array(engs_m)
    try:
        _, tau_metro = calculate_autocorr(time_series=mags_m_arr)
    except ZeroVarianceAutocorrelationError:
        tau_metro = float('nan')
    _, _, chi_metro, _ = calculate_thermodynamics(
        mags=mags_m_arr, engs=engs_m_arr, T=T, L=L,
    )
    iss_metro = (
        (meas_steps / t_metro) / tau_metro
        if np.isfinite(tau_metro) and tau_metro > 0
        else float('nan')
    )

    # ---- Wolff cluster (tau_int and ISS via sim.run for clean timing) ----
    sim_w = IsingSimulation(size=L, temp=T, update='wolff', seed=seed + 1)
    adaptive_equilibrate(sim_w, min_steps=eq_steps)
    t0 = time.perf_counter()
    mags_w, engs_w = sim_w.run(n_steps=meas_steps)
    t_wolff = time.perf_counter() - t0
    mags_w_arr = np.array(mags_w)
    engs_w_arr = np.array(engs_w)
    try:
        _, tau_wolff = calculate_autocorr(time_series=mags_w_arr)
    except ZeroVarianceAutocorrelationError:
        tau_wolff = float('nan')
    _, _, chi_wolff, _ = calculate_thermodynamics(
        mags=mags_w_arr, engs=engs_w_arr, T=T, L=L,
    )
    iss_wolff = (
        (meas_steps / t_wolff) / tau_wolff
        if np.isfinite(tau_wolff) and tau_wolff > 0
        else float('nan')
    )

    # ---- Cluster size: separate short pass to keep timing clean ----
    cluster_steps = min(meas_steps, 300)
    sim_c = IsingSimulation(size=L, temp=T, update='wolff', seed=seed + 2)
    sim_c.equilibrate(n_steps=eq_steps)
    _, _, cluster_sizes_arr = sim_c.run_with_cluster_sizes(n_steps=cluster_steps)
    mean_cluster_frac = float(np.mean(cluster_sizes_arr)) / (L * L)

    return {
        'T': T,
        'tau_metro': tau_metro,
        'tau_wolff': tau_wolff,
        'iss_metro': iss_metro,
        'iss_wolff': iss_wolff,
        'mean_cluster_frac': mean_cluster_frac,
        'chi_metro': chi_metro,
        'chi_wolff': chi_wolff,
    }


def _plot_efficiency(
    *,
    temperatures: np.ndarray,
    tau_metro: np.ndarray,
    tau_wolff: np.ndarray,
    iss_metro: np.ndarray,
    iss_wolff: np.ndarray,
    mean_cluster_frac: np.ndarray,
    chi_metro: np.ndarray,
    chi_wolff: np.ndarray,
    L: int,
    directory: str,
) -> None:
    """
    Produce and save the 4-panel efficiency comparison figure.

    Parameters
    ----------
    temperatures : np.ndarray
        Sorted temperature array.
    tau_metro : np.ndarray
        Integrated autocorrelation time for Metropolis checkerboard (steps).
    tau_wolff : np.ndarray
        Integrated autocorrelation time for Wolff cluster (steps).
    iss_metro : np.ndarray
        Independent samples per second — Metropolis.
    iss_wolff : np.ndarray
        Independent samples per second — Wolff.
    mean_cluster_frac : np.ndarray
        Mean cluster size fraction per Wolff step, ``<C> / N^2``.
    chi_metro : np.ndarray
        Magnetic susceptibility from Metropolis.
    chi_wolff : np.ndarray
        Magnetic susceptibility from Wolff.
    L : int
        Lattice size used in the simulation.
    directory : str
        Output directory for the saved PNG.
    """
    palette = {'metro': '#4878CF', 'wolff': '#D65F5F'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f'Wolff vs. Metropolis Efficiency — 2D Ising Model  (L = {L})',
        fontsize=13,
    )

    # Panel 1: tau_int(T)
    ax = axes[0, 0]
    ax.semilogy(
        temperatures, tau_metro, '-o', color=palette['metro'], ms=4, label='Metropolis',
    )
    ax.semilogy(
        temperatures, tau_wolff, '-s', color=palette['wolff'], ms=4, label='Wolff',
    )
    ax.axvline(TC_ISING, color='0.4', ls='--', lw=1, label=r'$T_c$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\tau_{\mathrm{int}}$ (steps)')
    ax.set_title('Integrated autocorrelation time')
    ax.legend(fontsize=8)

    # Panel 2: ISS(T)
    ax = axes[0, 1]
    ax.plot(temperatures, iss_metro, '-o', color=palette['metro'], ms=4, label='Metropolis')
    ax.plot(temperatures, iss_wolff, '-s', color=palette['wolff'], ms=4, label='Wolff')
    ax.axvline(TC_ISING, color='0.4', ls='--', lw=1, label=r'$T_c$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('Independent samples / s')
    ax.set_title('Sampling efficiency (ISS)')
    ax.legend(fontsize=8)

    # Panel 3: mean cluster size fraction <C>/N^2
    ax = axes[1, 0]
    ax.plot(temperatures, mean_cluster_frac, '-^', color=palette['wolff'], ms=4)
    ax.axvline(TC_ISING, color='0.4', ls='--', lw=1, label=r'$T_c$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\langle C \rangle \,/\, N^2$')
    ax.set_title('Mean cluster size fraction (Wolff)')
    ax.legend(fontsize=8)

    # Panel 4: chi(T) consistency check
    ax = axes[1, 1]
    ax.plot(temperatures, chi_metro, '-o', color=palette['metro'], ms=4, label='Metropolis')
    ax.plot(temperatures, chi_wolff, '-s', color=palette['wolff'], ms=4, label='Wolff')
    ax.axvline(TC_ISING, color='0.4', ls='--', lw=1, label=r'$T_c$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\chi$')
    ax.set_title(r'Susceptibility $\chi$ (consistency check)')
    ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(directory, exist_ok=True)
    out_path = os.path.join(directory, 'wolff_efficiency.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    """
    Execute the Wolff efficiency comparison sweep and save figure and data.

    Runs ``_measure_efficiency_point`` in parallel across the requested
    temperature range, then writes a 4-panel PNG and an ``.npz`` data file
    to ``--output-dir``.  The ``.npz`` is consumed by
    ``Wolff_Efficiency.ipynb`` to avoid re-running the sweep.
    """
    parser = argparse.ArgumentParser(
        description='Wolff vs. Metropolis efficiency demo for the 2D Ising model.',
    )
    parser.add_argument('--size', type=int, default=64, help='Lattice size L (default: 64)')
    parser.add_argument(
        '--eq-steps', type=int, default=500,
        help='Minimum equilibration steps per temperature point (default: 500)',
    )
    parser.add_argument(
        '--meas-steps', type=int, default=2000,
        help='Measurement steps per algorithm per temperature point (default: 2000)',
    )
    parser.add_argument('--t-min', type=float, default=1.8, help='Minimum T (default: 1.8)')
    parser.add_argument('--t-max', type=float, default=3.2, help='Maximum T (default: 3.2)')
    parser.add_argument(
        '--t-points', type=int, default=20, help='Temperature grid points (default: 20)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/ising',
        help='Output directory (default: results/ising)',
    )
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)
    logger.info(
        'Wolff efficiency demo: L=%d, T in [%.2f, %.2f], %d points, %d meas steps.',
        L, args.t_min, args.t_max, args.t_points, args.meas_steps,
    )

    sweep_params = [
        (T, L, args.eq_steps, args.meas_steps, idx * 1000)
        for idx, T in enumerate(temperatures)
    ]
    raw: list[dict[str, float]] = parallel_sweep(
        worker_func=_measure_efficiency_point, params=sweep_params,
    )

    tau_metro = np.array([r['tau_metro'] for r in raw])
    tau_wolff = np.array([r['tau_wolff'] for r in raw])
    iss_metro = np.array([r['iss_metro'] for r in raw])
    iss_wolff = np.array([r['iss_wolff'] for r in raw])
    mean_cluster_frac = np.array([r['mean_cluster_frac'] for r in raw])
    chi_metro = np.array([r['chi_metro'] for r in raw])
    chi_wolff = np.array([r['chi_wolff'] for r in raw])

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, 'wolff_efficiency.npz')
    np.savez(
        npz_path,
        temperatures=temperatures,
        tau_metro=tau_metro,
        tau_wolff=tau_wolff,
        iss_metro=iss_metro,
        iss_wolff=iss_wolff,
        mean_cluster_frac=mean_cluster_frac,
        chi_metro=chi_metro,
        chi_wolff=chi_wolff,
        L=np.int64(L),
    )
    logger.info('Data saved to %s', npz_path)

    _plot_efficiency(
        temperatures=temperatures,
        tau_metro=tau_metro,
        tau_wolff=tau_wolff,
        iss_metro=iss_metro,
        iss_wolff=iss_wolff,
        mean_cluster_frac=mean_cluster_frac,
        chi_metro=chi_metro,
        chi_wolff=chi_wolff,
        L=L,
        directory=args.output_dir,
    )
    logger.info('Figure saved to %s', os.path.join(args.output_dir, 'wolff_efficiency.png'))


if __name__ == '__main__':
    main()
