"""
Supplementary coarsening analysis for the 2D Ising model.

Precomputes three data sets used by the Correlation_and_Coarsening notebook:

1. **Quench-depth sensitivity**: Multiple quench temperatures, multi-seed
   median R_xi(t) at each, confirming that the growth exponent n = 1/2 is
   independent of quench depth while the prefactor depends on temperature.

2. **Equilibrium crossover**: Coarsening R_xi(t) trajectory on a small lattice
   long enough to saturate at the equilibrium correlation length xi_eq,
   demonstrating the bridge between non-equilibrium coarsening and
   equilibrium structure.

3. **Stochastic ensemble**: Multiple independent seeds at one quench
   temperature, storing per-seed traces to visualize run-to-run variability
   and the ensemble median + IQR.
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
from tqdm import tqdm

from models.ising_model import IsingSimulation
from utils.equilibration import convergence_equilibrate
from utils.observables import get_averaged_correlation
from utils.plotting import ensure_results_dir
from utils.system import parse_args_compat, setup_logging

TC_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))

# tqdm bar format matching the project convention.
_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]'


def _correlation_length_1e(r: np.ndarray, G: np.ndarray) -> float:
    """Extract correlation length as the first r where G(r) < G(0)/e."""
    threshold = G[0] / np.e
    idx = np.where(G < threshold)[0]
    return float(r[idx[0]]) if len(idx) > 0 else float(r[-1])


def _run_coarsening_traces(
    *,
    size: int,
    temp: float,
    n_steps: int,
    sample_interval: int,
    n_seeds: int,
    base_seed: int,
    logger: logging.Logger,
    desc: str = '',
) -> tuple[np.ndarray, np.ndarray]:
    """Run multi-seed coarsening and return (times, seed_traces) arrays.

    Parameters
    ----------
    size : int
        Linear lattice size.
    temp : float
        Quench temperature.
    n_steps : int
        Total MC steps.
    sample_interval : int
        Steps between R_xi measurements.
    n_seeds : int
        Number of independent seeds.
    base_seed : int
        First seed value.
    logger : logging.Logger
        Logger instance.
    desc : str, optional
        Progress bar description prefix.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``times`` of shape ``(n_samples,)`` and ``traces`` of shape
        ``(n_seeds, n_samples)``.
    """
    times = np.arange(sample_interval, n_steps + 1, sample_interval)
    traces = np.zeros((n_seeds, len(times)))

    for s in tqdm(range(n_seeds), bar_format=_BAR_FORMAT, desc=desc or 'Seeds'):
        seed = base_seed + s
        sim = IsingSimulation(
            size=size, temp=temp, update='random', init_state='random', seed=seed,
        )
        rec = 0
        for t in range(1, n_steps + 1):
            sim.step()
            if t % sample_interval == 0:
                r_arr, G_arr = get_averaged_correlation(
                    sim=sim, total_steps=1, sample_interval=1,
                )
                traces[s, rec] = _correlation_length_1e(r_arr, G_arr)
                rec += 1
        logger.debug(f'{desc} seed={seed}: final R_xi={traces[s, -1]:.2f}')

    return times, traces


def _measure_xi_eq(
    *,
    size: int,
    temp: float,
    seed: int,
    eq_probe: int,
    eq_max: int,
    meas_steps: int,
    meas_interval: int,
    logger: logging.Logger,
) -> float:
    """Measure the equilibrium correlation length at a given temperature.

    Uses two-start convergence equilibration before measuring.

    Parameters
    ----------
    size : int
        Linear lattice size.
    temp : float
        Temperature.
    seed : int
        Random seed.
    eq_probe : int
        Convergence probe chunk size.
    eq_max : int
        Maximum equilibration steps.
    meas_steps : int
        Measurement steps after equilibration.
    meas_interval : int
        Sample interval during measurement.
    logger : logging.Logger
        Logger.

    Returns
    -------
    float
        Equilibrium correlation length (1/e criterion).
    """
    logger.info(f'Measuring xi_eq at T={temp:.4f} (L={size})...')
    sim_r = IsingSimulation(
        size=size, temp=temp, update='checkerboard', init_state='random', seed=seed,
    )
    sim_o = IsingSimulation(
        size=size, temp=temp, update='checkerboard', init_state='ordered', seed=seed,
    )
    convergence_equilibrate(sim_r, sim_o, chunk_size=eq_probe, max_steps=eq_max)
    r_eq, G_eq = get_averaged_correlation(
        sim=sim_r, total_steps=meas_steps, sample_interval=meas_interval,
    )
    xi = _correlation_length_1e(r_eq, G_eq)
    logger.info(f'xi_eq = {xi:.2f} lattice spacings')
    return xi


def main() -> None:
    """Run the supplementary coarsening analysis."""
    parser = argparse.ArgumentParser(
        description='Ising Supplementary Coarsening Analysis',
    )
    # Shared parameters.
    parser.add_argument('--size', type=int, default=64, help='Linear lattice size L')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    # Quench-depth parameters.
    parser.add_argument(
        '--quench-fracs', type=float, nargs='+', default=[0.1, 0.5, 0.8],
        help='Quench temperatures as fractions of T_c',
    )
    parser.add_argument('--quench-steps', type=int, default=200, help='MC steps per quench run')
    parser.add_argument('--quench-interval', type=int, default=5, help='Sample interval for quench')
    parser.add_argument('--quench-seeds', type=int, default=4, help='Seeds per quench temperature')

    # Crossover parameters.
    parser.add_argument(
        '--bridge-frac', type=float, default=0.5,
        help='Crossover quench temp as fraction of T_c',
    )
    parser.add_argument('--bridge-steps', type=int, default=2000, help='MC steps for crossover run')
    parser.add_argument(
        '--bridge-interval', type=int, default=5,
        help='Sample interval for crossover',
    )
    parser.add_argument('--bridge-seeds', type=int, default=4, help='Seeds for crossover averaging')

    # Ensemble parameters.
    parser.add_argument(
        '--ens-frac', type=float, default=0.1,
        help='Ensemble quench temp as fraction of T_c',
    )
    parser.add_argument('--ens-steps', type=int, default=100, help='MC steps per ensemble trace')
    parser.add_argument('--ens-interval', type=int, default=2, help='Sample interval for ensemble')
    parser.add_argument('--ens-seeds', type=int, default=8, help='Number of ensemble seeds')

    # Seed control.
    parser.add_argument('--base-seed', type=int, default=42, help='Starting seed')

    args = parse_args_compat(parser)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    npz_data: dict[str, Any] = {
        'size': args.size,
        'Tc': TC_ISING,
    }

    # ===== 1. Quench-depth sensitivity =====
    logger.info('=== Quench-depth sensitivity ===')
    quench_temps = [f * TC_ISING for f in args.quench_fracs]
    quench_all_traces: dict[float, np.ndarray] = {}

    for frac, T_q in zip(args.quench_fracs, quench_temps, strict=True):
        logger.info(f'Quench fraction={frac} -> T={T_q:.4f}')
        times_q, traces_q = _run_coarsening_traces(
            size=args.size, temp=T_q,
            n_steps=args.quench_steps, sample_interval=args.quench_interval,
            n_seeds=args.quench_seeds, base_seed=args.base_seed,
            logger=logger, desc=f'Quench f={frac}',
        )
        quench_all_traces[frac] = traces_q

    npz_data['quench_fracs'] = np.array(args.quench_fracs)
    npz_data['quench_temps'] = np.array(quench_temps)
    npz_data['quench_times'] = times_q
    npz_data['quench_n_seeds'] = args.quench_seeds
    for i, frac in enumerate(args.quench_fracs):
        traces = quench_all_traces[frac]
        npz_data[f'quench_{i}_seeds'] = traces
        npz_data[f'quench_{i}_median'] = np.median(traces, axis=0)
        npz_data[f'quench_{i}_q25'] = np.percentile(traces, 25, axis=0)
        npz_data[f'quench_{i}_q75'] = np.percentile(traces, 75, axis=0)

    # ===== 2. Equilibrium crossover =====
    logger.info('=== Equilibrium crossover ===')
    T_bridge = args.bridge_frac * TC_ISING

    xi_eq = _measure_xi_eq(
        size=args.size, temp=T_bridge, seed=args.base_seed + 200,
        eq_probe=150, eq_max=6000, meas_steps=2000, meas_interval=20,
        logger=logger,
    )

    times_b, traces_b = _run_coarsening_traces(
        size=args.size, temp=T_bridge,
        n_steps=args.bridge_steps, sample_interval=args.bridge_interval,
        n_seeds=args.bridge_seeds, base_seed=args.base_seed,
        logger=logger, desc='Crossover',
    )

    npz_data['bridge_frac'] = args.bridge_frac
    npz_data['bridge_temp'] = T_bridge
    npz_data['bridge_xi_eq'] = xi_eq
    npz_data['bridge_times'] = times_b
    npz_data['bridge_seeds'] = traces_b
    npz_data['bridge_median'] = np.median(traces_b, axis=0)
    npz_data['bridge_n_seeds'] = args.bridge_seeds

    # ===== 3. Stochastic ensemble =====
    logger.info('=== Stochastic ensemble ===')
    T_ens = args.ens_frac * TC_ISING

    times_e, traces_e = _run_coarsening_traces(
        size=args.size, temp=T_ens,
        n_steps=args.ens_steps, sample_interval=args.ens_interval,
        n_seeds=args.ens_seeds, base_seed=args.base_seed + 100,
        logger=logger, desc='Ensemble',
    )

    npz_data['ens_frac'] = args.ens_frac
    npz_data['ens_temp'] = T_ens
    npz_data['ens_times'] = times_e
    npz_data['ens_seeds'] = traces_e
    npz_data['ens_median'] = np.median(traces_e, axis=0)
    npz_data['ens_q25'] = np.percentile(traces_e, 25, axis=0)
    npz_data['ens_q75'] = np.percentile(traces_e, 75, axis=0)
    npz_data['ens_n_seeds'] = args.ens_seeds

    # ===== Save =====
    output_dir = ensure_results_dir(directory=args.output_dir)
    npz_path = f'{output_dir}/coarsening_analysis.npz'
    np.savez_compressed(npz_path, **npz_data)
    logger.info(f'All coarsening analysis data saved to {npz_path}')


if __name__ == '__main__':
    main()
