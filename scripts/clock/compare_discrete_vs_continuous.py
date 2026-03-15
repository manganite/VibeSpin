"""
Side-by-side temperature sweep: continuous clock (XY + anisotropy) vs discrete clock.
"""
from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from utils.physics_helpers import calculate_thermodynamics


def sweep_model(
    *,
    model_cls: type,
    temperatures: np.ndarray,
    L: int,
    q: int,
    eq_steps: int,
    meas_steps: int,
    extra_kwargs: dict,
) -> tuple[list[float], list[float], list[float], list[float]]:
    avg_m_list: list[float] = []
    avg_e_list: list[float] = []
    susc_list: list[float] = []
    spec_h_list: list[float] = []

    for T in temperatures:
        sim = model_cls(size=L, temp=T, q=q, **extra_kwargs)
        sim.equilibrate(n_steps=eq_steps)
        mags, engs = sim.run(n_steps=meas_steps)
        avg_m, avg_e, susc, spec_h = calculate_thermodynamics(
            mags=np.array(mags), engs=np.array(engs), T=T, L=L,
        )
        avg_m_list.append(avg_m)
        avg_e_list.append(avg_e)
        susc_list.append(susc)
        spec_h_list.append(spec_h)
    return avg_m_list, avg_e_list, susc_list, spec_h_list


def main() -> None:
    L = 32
    q = 6
    eq_steps = 5000
    meas_steps = 5000
    temperatures = np.linspace(0.1, 2.0, 30)

    print(f'Sweeping q={q} clock models, L={L}, {len(temperatures)} temperatures')
    print(f'  eq={eq_steps}, meas={meas_steps} steps each\n')

    # --- Continuous clock (XY + anisotropy A=0.1) ---
    print('Running continuous clock (A=0.1)...')
    t0 = time.perf_counter()
    cm, ce, cs, cc = sweep_model(
        model_cls=ClockSimulation,
        temperatures=temperatures,
        L=L, q=q, eq_steps=eq_steps, meas_steps=meas_steps,
        extra_kwargs={'A': 0.1},
    )
    print(f'  Done in {time.perf_counter() - t0:.1f}s')

    # --- Discrete clock ---
    print('Running discrete clock...')
    t0 = time.perf_counter()
    dm, de, ds, dc = sweep_model(
        model_cls=DiscreteClockSimulation,
        temperatures=temperatures,
        L=L, q=q, eq_steps=eq_steps, meas_steps=meas_steps,
        extra_kwargs={},
    )
    print(f'  Done in {time.perf_counter() - t0:.1f}s')

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
    outdir = 'results/clock'
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, 'discrete_vs_continuous.png')
    fig.savefig(outpath, dpi=150)
    print(f'\nPlot saved to {outpath}')


if __name__ == '__main__':
    main()
