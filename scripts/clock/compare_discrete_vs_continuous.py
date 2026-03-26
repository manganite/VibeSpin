"""
Side-by-side temperature sweep: continuous clock (XY + anisotropy) vs discrete clock.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from utils.analysis import calculate_thermodynamics
from utils.equilibration import convergence_equilibrate
from utils.system import setup_logging


def sweep_model(
    *,
    model_cls: type,
    temperatures: np.ndarray,
    L: int,
    q: int,
    eq_probe_steps: int,
    eq_max_steps: int,
    meas_steps: int,
    extra_kwargs: dict,
) -> tuple[list[float], list[float], list[float], list[float]]:
    avg_m_list: list[float] = []
    avg_e_list: list[float] = []
    susc_list: list[float] = []
    spec_h_list: list[float] = []

    for T in temperatures:
        sim_r = model_cls(size=L, temp=T, q=q, init_state='random', **extra_kwargs)
        sim_o = model_cls(size=L, temp=T, q=q, init_state='ordered', **extra_kwargs)
        convergence_equilibrate(sim_r, sim_o, chunk_size=eq_probe_steps, max_steps=eq_max_steps)

        mags, engs = sim_r.run(n_steps=meas_steps)
        avg_m, avg_e, susc, spec_h = calculate_thermodynamics(
            mags=np.array(mags), engs=np.array(engs), T=T, L=L,
        )
        avg_m_list.append(avg_m)
        avg_e_list.append(avg_e)
        susc_list.append(susc)
        spec_h_list.append(spec_h)
    return avg_m_list, avg_e_list, susc_list, spec_h_list


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare Discrete vs Continuous Clock Models')
    parser.add_argument('--size', type=int, default=32, help='Lattice size L')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    parser.add_argument(
        '--eq-probe-steps', type=int, default=500,
        help='Chunk size for convergence check during equilibration',
    )
    parser.add_argument(
        '--eq-max-steps', type=int, default=100000,
        help='Hard cap on equilibration steps per temperature point',
    )
    parser.add_argument('--meas-steps', type=int, default=5000, help='Measurement steps')
    parser.add_argument('--t-min', type=float, default=0.1, help='Minimum temperature')
    parser.add_argument('--t-max', type=float, default=2.0, help='Maximum temperature')
    parser.add_argument('--t-points', type=int, default=30, help='Number of temperature points')
    parser.add_argument('--output-dir', type=str, default='results/clock', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)

    L = args.size
    q = args.q
    eq_probe_steps = args.eq_probe_steps
    eq_max_steps = args.eq_max_steps
    meas_steps = args.meas_steps
    temperatures = np.linspace(args.t_min, args.t_max, args.t_points)

    logger.info(f'Sweeping q={q} clock models, L={L}, {len(temperatures)} temperatures')
    logger.info(
        f'  eq_probe={eq_probe_steps}, eq_max={eq_max_steps}, '
        f'meas={meas_steps} steps each\n'
    )

    # --- Continuous clock (XY + anisotropy A=0.1) ---
    logger.info('Running continuous clock (A=0.1)...')
    t0 = time.perf_counter()
    cm, ce, cs, cc = sweep_model(
        model_cls=ClockSimulation,
        temperatures=temperatures,
        L=L, q=q,
        eq_probe_steps=eq_probe_steps,
        eq_max_steps=eq_max_steps,
        meas_steps=meas_steps,
        extra_kwargs={'A': 0.1},
    )
    logger.info(f'  Done in {time.perf_counter() - t0:.1f}s')

    # --- Discrete clock ---
    logger.info('Running discrete clock...')
    t0 = time.perf_counter()
    dm, de, ds, dc = sweep_model(
        model_cls=DiscreteClockSimulation,
        temperatures=temperatures,
        L=L, q=q,
        eq_probe_steps=eq_probe_steps,
        eq_max_steps=eq_max_steps,
        meas_steps=meas_steps,
        extra_kwargs={},
    )
    logger.info(f'  Done in {time.perf_counter() - t0:.1f}s')

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f'{q}-state Clock Model Comparison (L={L})',
        fontsize=14,
    )

    labels = [
        ('Magnetization $\\langle |M| \\rangle$', cm, dm),
        ('Energy $\\langle E \\rangle$', ce, de),
        ('Susceptibility $\\chi$', cs, ds),
        ('Specific Heat $C$', cc, dc),
    ]

    for ax, (ylabel, cont_data, disc_data) in zip(axes.flat, labels, strict=True):
        ax.plot(temperatures, cont_data, 'o-', ms=4, label='Continuous (A=0.1)')
        ax.plot(temperatures, disc_data, 's--', ms=4, label='Discrete')
        ax.set_xlabel('Temperature $T$')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    os.makedirs(args.output_dir, exist_ok=True)
    outpath = os.path.join(args.output_dir, 'discrete_vs_continuous.png')
    fig.savefig(outpath, dpi=150)
    logger.info(f'\nPlot saved to {outpath}')


if __name__ == '__main__':
    main()
