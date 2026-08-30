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
from utils.system import parallel_sweep, parse_args_compat, setup_logging

#: Exact Onsager critical temperature for the 2D nearest-neighbour Ising model.
TC_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))


def _measure_efficiency_point(
    params: tuple[int, int, float, int, int, int, int],
) -> dict[str, float | int]:
    """
    Worker: measure algorithmic efficiency at one temperature point.

    Runs the Metropolis checkerboard and Wolff cluster algorithms from
    independent equilibrated states at temperature *T*. For each algorithm
    the worker records wall-clock time for *meas_steps* steps via
    ``sim.run()``, computes tau_int from the magnetisation series, and derives
    ISS = (steps / wall_time) / tau_int. Cluster sizes are measured in a
    separate, short Wolff pass to keep the timing paths clean.

    Parameters
    ----------
    params : tuple
        ``(temp_idx, seed_idx, T, L, eq_probe_steps, eq_max_steps, meas_steps)``
        where ``temp_idx`` and ``seed_idx`` identify the grid position and
        random-seed replicate. A deterministic base seed is derived from these
        indices so aggregation is reproducible.

    Returns
    -------
    dict
        Keys: ``T``, ``tau_metro``, ``tau_wolff``, ``iss_metro``,
        ``iss_wolff``, ``mean_cluster_frac``, ``chi_metro``, ``chi_wolff``.
    """
    temp_idx, seed_idx, T, L, eq_probe_steps, eq_max_steps, meas_steps = params
    seed = derive_point_seed(temperature_index=temp_idx, seed_index=seed_idx)

    # ---- Metropolis checkerboard ----
    sim_m_r = IsingSimulation(
        size=L, temp=T, update='checkerboard', init_state='random', seed=seed
    )
    sim_m_o = IsingSimulation(
        size=L, temp=T, update='checkerboard', init_state='ordered', seed=seed
    )
    convergence_equilibrate(
        sim_random=sim_m_r, sim_ordered=sim_m_o,
        chunk_size=eq_probe_steps, max_steps=eq_max_steps,
    )

    t0 = time.perf_counter()
    mags_m, engs_m = sim_m_r.run(n_steps=meas_steps)
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
    sim_w_r = IsingSimulation(size=L, temp=T, update='wolff', init_state='random', seed=seed + 1)
    sim_w_o = IsingSimulation(size=L, temp=T, update='wolff', init_state='ordered', seed=seed + 1)
    convergence_equilibrate(
        sim_random=sim_w_r, sim_ordered=sim_w_o,
        chunk_size=eq_probe_steps, max_steps=eq_max_steps,
    )

    t0 = time.perf_counter()
    mags_w, engs_w = sim_w_r.run(n_steps=meas_steps)
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
    sim_c_r = IsingSimulation(size=L, temp=T, update='wolff', init_state='random', seed=seed + 2)
    sim_c_o = IsingSimulation(size=L, temp=T, update='wolff', init_state='ordered', seed=seed + 2)
    convergence_equilibrate(
        sim_random=sim_c_r, sim_ordered=sim_c_o,
        chunk_size=eq_probe_steps, max_steps=eq_max_steps,
    )
    _, _, cluster_sizes_arr = sim_c_r.run_with_cluster_sizes(n_steps=cluster_steps)
    mean_cluster_frac = float(np.mean(cluster_sizes_arr)) / (L * L)

    return {
        'temp_idx': int(temp_idx),
        'seed_idx': int(seed_idx),
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
    tau_metro_norm: np.ndarray,
    tau_metro_norm_p16: np.ndarray,
    tau_metro_norm_p84: np.ndarray,
    tau_wolff_norm: np.ndarray,
    tau_wolff_norm_p16: np.ndarray,
    tau_wolff_norm_p84: np.ndarray,
    iss_metro: np.ndarray,
    iss_metro_p16: np.ndarray,
    iss_metro_p84: np.ndarray,
    iss_wolff: np.ndarray,
    iss_wolff_p16: np.ndarray,
    iss_wolff_p84: np.ndarray,
    mean_cluster_frac: np.ndarray,
    mean_cluster_frac_p16: np.ndarray,
    mean_cluster_frac_p84: np.ndarray,
    chi_metro: np.ndarray,
    chi_metro_p16: np.ndarray,
    chi_metro_p84: np.ndarray,
    chi_wolff: np.ndarray,
    chi_wolff_p16: np.ndarray,
    chi_wolff_p84: np.ndarray,
    n_seeds: int,
    L: int,
    directory: str,
) -> None:
    """
    Produce and save the 4-panel efficiency comparison figure.

    Parameters
    ----------
    temperatures : np.ndarray
        Sorted temperature array.
    tau_metro_norm : np.ndarray
        Work-normalized integrated autocorrelation time for Metropolis.
    tau_metro_norm_p16 : np.ndarray
        16th percentile band for ``tau_metro_norm``.
    tau_metro_norm_p84 : np.ndarray
        84th percentile band for ``tau_metro_norm``.
    tau_wolff_norm : np.ndarray
        Work-normalized integrated autocorrelation time for Wolff.
    tau_wolff_norm_p16 : np.ndarray
        16th percentile band for ``tau_wolff_norm``.
    tau_wolff_norm_p84 : np.ndarray
        84th percentile band for ``tau_wolff_norm``.
    iss_metro : np.ndarray
        Independent samples per second — Metropolis.
    iss_metro_p16 : np.ndarray
        16th percentile band for ``iss_metro``.
    iss_metro_p84 : np.ndarray
        84th percentile band for ``iss_metro``.
    iss_wolff : np.ndarray
        Independent samples per second — Wolff.
    iss_wolff_p16 : np.ndarray
        16th percentile band for ``iss_wolff``.
    iss_wolff_p84 : np.ndarray
        84th percentile band for ``iss_wolff``.
    mean_cluster_frac : np.ndarray
        Mean cluster size fraction per Wolff step, ``<C> / N^2``.
    mean_cluster_frac_p16 : np.ndarray
        16th percentile band for ``mean_cluster_frac``.
    mean_cluster_frac_p84 : np.ndarray
        84th percentile band for ``mean_cluster_frac``.
    chi_metro : np.ndarray
        Magnetic susceptibility from Metropolis.
    chi_metro_p16 : np.ndarray
        16th percentile band for ``chi_metro``.
    chi_metro_p84 : np.ndarray
        84th percentile band for ``chi_metro``.
    chi_wolff : np.ndarray
        Magnetic susceptibility from Wolff.
    chi_wolff_p16 : np.ndarray
        16th percentile band for ``chi_wolff``.
    chi_wolff_p84 : np.ndarray
        84th percentile band for ``chi_wolff``.
    n_seeds : int
        Number of seed replicas used for uncertainty estimation.
    L : int
        Lattice size used in the simulation.
    directory : str
        Output directory for the saved PNG.
    """
    palette = {'metro': '#4878CF', 'wolff': '#D65F5F'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f'Wolff vs. Metropolis Efficiency — 2D Ising Model  (L = {L}, seeds = {n_seeds})',
        fontsize=13,
    )

    # Panel 1: work-normalized tau_int(T)
    ax = axes[0, 0]
    ax.semilogy(
        temperatures, tau_metro_norm, '-o', color=palette['metro'], ms=4, label='Metropolis',
    )
    ax.semilogy(
        temperatures,
        tau_wolff_norm,
        '-s',
        color=palette['wolff'],
        ms=4,
        label='Wolff (normalised)',
    )
    ax.fill_between(
        temperatures, tau_metro_norm_p16, tau_metro_norm_p84,
        color=palette['metro'], alpha=0.15,
    )
    ax.fill_between(
        temperatures, tau_wolff_norm_p16, tau_wolff_norm_p84,
        color=palette['wolff'], alpha=0.15,
    )
    ax.axvline(TC_ISING, color='0.4', ls='--', lw=1, label=r'$T_c$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\tau^{\mathrm{norm}}_{\mathrm{int}}$ ($L^2$-sweep equiv.)')
    ax.set_title('Integrated autocorrelation time\n(work-normalised; median with 16–84% band)')
    ax.legend(fontsize=8)

    # Panel 2: ISS(T)
    ax = axes[0, 1]
    ax.semilogy(temperatures, iss_metro, '-o', color=palette['metro'], ms=4, label='Metropolis')
    ax.semilogy(temperatures, iss_wolff, '-s', color=palette['wolff'], ms=4, label='Wolff')
    ax.fill_between(
        temperatures, iss_metro_p16, iss_metro_p84,
        color=palette['metro'], alpha=0.15,
    )
    ax.fill_between(
        temperatures, iss_wolff_p16, iss_wolff_p84,
        color=palette['wolff'], alpha=0.15,
    )
    ax.axvline(TC_ISING, color='0.4', ls='--', lw=1, label=r'$T_c$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel('Independent samples / s')
    ax.set_title('Sampling efficiency ISS\n(median with 16–84% band)')
    ax.legend(fontsize=8)

    # Panel 3: mean cluster size fraction <C>/N^2
    ax = axes[1, 0]
    ax.plot(temperatures, mean_cluster_frac, '-^', color=palette['wolff'], ms=4)
    ax.fill_between(
        temperatures, mean_cluster_frac_p16, mean_cluster_frac_p84,
        color=palette['wolff'], alpha=0.15,
    )
    ax.axvline(TC_ISING, color='0.4', ls='--', lw=1, label=r'$T_c$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\langle C \rangle \,/\, N^2$')
    ax.set_title(
        r'Mean cluster size fraction (Wolff)'
        + '\n'
        + r'= normalisation factor $\langle C\rangle/L^2$',
    )
    ax.legend(fontsize=8)

    # Panel 4: chi(T) consistency check
    ax = axes[1, 1]
    ax.plot(temperatures, chi_metro, '-o', color=palette['metro'], ms=4, label='Metropolis')
    ax.plot(temperatures, chi_wolff, '-s', color=palette['wolff'], ms=4, label='Wolff')
    ax.fill_between(
        temperatures, chi_metro_p16, chi_metro_p84,
        color=palette['metro'], alpha=0.15,
    )
    ax.fill_between(
        temperatures, chi_wolff_p16, chi_wolff_p84,
        color=palette['wolff'], alpha=0.15,
    )
    ax.axvline(TC_ISING, color='0.4', ls='--', lw=1, label=r'$T_c$')
    ax.set_xlabel('Temperature $T$')
    ax.set_ylabel(r'$\chi$')
    ax.set_title(r'Susceptibility (agreement validates correctness)')
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
        '--eq-probe-steps', type=int, default=500,
        help='Chunk size for convergence check during equilibration (default: 500)',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=200000,
        help='Hard cap on equilibration steps (default: 200000)',
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
        '--n-seeds', type=int, default=10,
        help='Independent seed replicas per temperature point (default: 10)',
    )
    parser.add_argument(
        '--output-dir', type=str, default='results/ising',
        help='Output directory (default: results/ising)',
    )
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parse_args_compat(parser=parser)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    temperatures: np.ndarray = np.linspace(args.t_min, args.t_max, args.t_points)
    logger.info(
        (
            'Wolff efficiency demo: L=%d, T in [%.2f, %.2f], %d points, '
            '%d seed replicas, %d meas steps.'
        ),
        L, args.t_min, args.t_max, args.t_points, args.n_seeds, args.meas_steps,
    )

    sweep_params = [
        (temp_idx, seed_idx, T, L, args.eq_probe_steps, args.eq_max_steps, args.meas_steps)
        for temp_idx, T in enumerate(temperatures)
        for seed_idx in range(args.n_seeds)
    ]
    raw: list[dict[str, float | int]] = parallel_sweep(
        worker_func=_measure_efficiency_point, params=sweep_params,
    )

    n_temp = len(temperatures)
    n_seed = int(args.n_seeds)

    tau_metro_samples = np.full((n_temp, n_seed), np.nan)
    tau_wolff_samples = np.full((n_temp, n_seed), np.nan)
    iss_metro_samples = np.full((n_temp, n_seed), np.nan)
    iss_wolff_samples = np.full((n_temp, n_seed), np.nan)
    mean_cluster_frac_samples = np.full((n_temp, n_seed), np.nan)
    chi_metro_samples = np.full((n_temp, n_seed), np.nan)
    chi_wolff_samples = np.full((n_temp, n_seed), np.nan)

    for r in raw:
        i = int(r['temp_idx'])
        s = int(r['seed_idx'])
        tau_metro_samples[i, s] = float(r['tau_metro'])
        tau_wolff_samples[i, s] = float(r['tau_wolff'])
        iss_metro_samples[i, s] = float(r['iss_metro'])
        iss_wolff_samples[i, s] = float(r['iss_wolff'])
        mean_cluster_frac_samples[i, s] = float(r['mean_cluster_frac'])
        chi_metro_samples[i, s] = float(r['chi_metro'])
        chi_wolff_samples[i, s] = float(r['chi_wolff'])

    def _summary(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Median and 16-84 percentile envelope via the shared replicate summarizer."""
        summary = summarize_replicate_samples(samples=samples)
        return (
            np.asarray(summary['value']),
            np.asarray(summary['ci_low']),
            np.asarray(summary['ci_high']),
        )

    tau_metro, tau_metro_p16, tau_metro_p84 = _summary(tau_metro_samples)
    tau_wolff, tau_wolff_p16, tau_wolff_p84 = _summary(tau_wolff_samples)
    iss_metro, iss_metro_p16, iss_metro_p84 = _summary(iss_metro_samples)
    iss_wolff, iss_wolff_p16, iss_wolff_p84 = _summary(iss_wolff_samples)
    (
        mean_cluster_frac,
        mean_cluster_frac_p16,
        mean_cluster_frac_p84,
    ) = _summary(mean_cluster_frac_samples)
    chi_metro, chi_metro_p16, chi_metro_p84 = _summary(chi_metro_samples)
    chi_wolff, chi_wolff_p16, chi_wolff_p84 = _summary(chi_wolff_samples)

    # Work-normalized tau for cross-algorithm comparability.
    tau_metro_norm_samples = tau_metro_samples
    tau_wolff_norm_samples = tau_wolff_samples * mean_cluster_frac_samples
    tau_metro_norm, tau_metro_norm_p16, tau_metro_norm_p84 = _summary(tau_metro_norm_samples)
    tau_wolff_norm, tau_wolff_norm_p16, tau_wolff_norm_p84 = _summary(tau_wolff_norm_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    npz_path = os.path.join(args.output_dir, 'wolff_efficiency.npz')
    nan_or_undefined_count = int(sum(
        int(np.isnan(arr).sum())
        for arr in (
            tau_metro_samples, tau_wolff_samples, iss_metro_samples,
            iss_wolff_samples, mean_cluster_frac_samples,
            chi_metro_samples, chi_wolff_samples,
        )
    ))
    np.savez(
        npz_path,
        temperatures=temperatures,
        n_seeds=np.int64(n_seed),
        uncertainty_method=UNCERTAINTY_METHOD_REPLICATE,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
        nan_or_undefined_count=nan_or_undefined_count,
        tau_metro=tau_metro,
        tau_metro_p16=tau_metro_p16,
        tau_metro_p84=tau_metro_p84,
        tau_metro_samples=tau_metro_samples,
        tau_wolff=tau_wolff,
        tau_wolff_p16=tau_wolff_p16,
        tau_wolff_p84=tau_wolff_p84,
        tau_wolff_samples=tau_wolff_samples,
        iss_metro=iss_metro,
        iss_metro_p16=iss_metro_p16,
        iss_metro_p84=iss_metro_p84,
        iss_metro_samples=iss_metro_samples,
        iss_wolff=iss_wolff,
        iss_wolff_p16=iss_wolff_p16,
        iss_wolff_p84=iss_wolff_p84,
        iss_wolff_samples=iss_wolff_samples,
        mean_cluster_frac=mean_cluster_frac,
        mean_cluster_frac_p16=mean_cluster_frac_p16,
        mean_cluster_frac_p84=mean_cluster_frac_p84,
        mean_cluster_frac_samples=mean_cluster_frac_samples,
        chi_metro=chi_metro,
        chi_metro_p16=chi_metro_p16,
        chi_metro_p84=chi_metro_p84,
        chi_metro_samples=chi_metro_samples,
        chi_wolff=chi_wolff,
        chi_wolff_p16=chi_wolff_p16,
        chi_wolff_p84=chi_wolff_p84,
        chi_wolff_samples=chi_wolff_samples,
        L=np.int64(L),
    )
    logger.info('Data saved to %s', npz_path)

    _plot_efficiency(
        temperatures=temperatures,
        tau_metro_norm=tau_metro_norm,
        tau_metro_norm_p16=tau_metro_norm_p16,
        tau_metro_norm_p84=tau_metro_norm_p84,
        tau_wolff_norm=tau_wolff_norm,
        tau_wolff_norm_p16=tau_wolff_norm_p16,
        tau_wolff_norm_p84=tau_wolff_norm_p84,
        iss_metro=iss_metro,
        iss_metro_p16=iss_metro_p16,
        iss_metro_p84=iss_metro_p84,
        iss_wolff=iss_wolff,
        iss_wolff_p16=iss_wolff_p16,
        iss_wolff_p84=iss_wolff_p84,
        mean_cluster_frac=mean_cluster_frac,
        mean_cluster_frac_p16=mean_cluster_frac_p16,
        mean_cluster_frac_p84=mean_cluster_frac_p84,
        chi_metro=chi_metro,
        chi_metro_p16=chi_metro_p16,
        chi_metro_p84=chi_metro_p84,
        chi_wolff=chi_wolff,
        chi_wolff_p16=chi_wolff_p16,
        chi_wolff_p84=chi_wolff_p84,
        n_seeds=n_seed,
        L=L,
        directory=args.output_dir,
    )
    logger.info('Figure saved to %s', os.path.join(args.output_dir, 'wolff_efficiency.png'))


if __name__ == '__main__':
    main()
