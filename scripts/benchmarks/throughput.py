"""
Comprehensive scaling benchmark for the VibeSpin Simulation Project.

Measures performance of MC sweeps and analysis functions across different
lattice sizes and saves results to results/benchmarks/scaling_benchmark.npz
for use by notebooks/Performance_Benchmarks.ipynb.

Usage
-----
    python scripts/benchmarks/throughput.py
    python scripts/benchmarks/throughput.py --sizes 64 128 256 512 --sweeps 200
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from models.ising_model import IsingSimulation
from models.simulation_base import MonteCarloSimulation
from models.xy_model import XYSimulation
from utils.system_helpers import ensure_results_dir, save_plot


def measure_performance(
    sim: MonteCarloSimulation, sweeps: int = 100, analysis_iters: int = 50
) -> dict[str, float]:
    """Measure sweeps/sec and analysis times for a simulation instance."""
    # Warm-up: trigger JIT
    sim.step()
    sim._get_energy()
    sim._get_magnetization()
    sim._calculate_correlation_function()
    if hasattr(sim, '_calculate_vorticity'):
        sim._calculate_vorticity()
    if hasattr(sim, '_get_vortex_density'):
        sim._get_vortex_density()
    if hasattr(sim, '_get_helicity_data'):
        sim._get_helicity_data()

    # 1. Sweep speed
    start = time.perf_counter()
    for _ in range(sweeps):
        sim.step()
    sweep_duration = time.perf_counter() - start
    sps = sweeps / sweep_duration

    # 2. Thermodynamic measurements (Energy + Mag)
    start = time.perf_counter()
    for _ in range(analysis_iters):
        sim._get_energy()
        sim._get_magnetization()
    thermo_ms = (time.perf_counter() - start) / analysis_iters * 1000

    # 3. Correlation function G(r)
    start = time.perf_counter()
    for _ in range(analysis_iters):
        sim._calculate_correlation_function()
    corr_ms = (time.perf_counter() - start) / analysis_iters * 1000

    # 4. Vorticity
    vort_ms = 0.0
    if hasattr(sim, '_calculate_vorticity'):
        start = time.perf_counter()
        for _ in range(analysis_iters):
            sim._calculate_vorticity()
        vort_ms = (time.perf_counter() - start) / analysis_iters * 1000

    # 5. Vortex Density
    vden_ms = 0.0
    if hasattr(sim, '_get_vortex_density'):
        start = time.perf_counter()
        for _ in range(analysis_iters):
            sim._get_vortex_density()
        vden_ms = (time.perf_counter() - start) / analysis_iters * 1000

    # 6. Helicity
    heli_ms = 0.0
    if hasattr(sim, '_get_helicity_data'):
        start = time.perf_counter()
        for _ in range(analysis_iters):
            sim._get_helicity_data()
        heli_ms = (time.perf_counter() - start) / analysis_iters * 1000

    return {
        'sps': sps,
        'thermo_ms': thermo_ms,
        'corr_ms': corr_ms,
        'vort_ms': vort_ms,
        'vden_ms': vden_ms,
        'heli_ms': heli_ms,
    }


def run_scaling_benchmark() -> None:
    parser = argparse.ArgumentParser(description='Scaling Benchmark')
    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[32, 64, 128, 256, 512, 1024],
        help='Lattice sizes',
    )
    parser.add_argument('--sweeps', type=int, default=100, help='Sweeps per point')
    parser.add_argument(
        '--output-dir', type=str, default='results/benchmarks', help='Output directory'
    )
    args = parser.parse_args()

    sizes = sorted(args.sizes)
    model_configs = [
        (
            'Ising (Checkerboard)',
            lambda L: IsingSimulation(size=L, temp=2.269, update='checkerboard'),
        ),
        ('Ising (Random)', lambda L: IsingSimulation(size=L, temp=2.269, update='random')),
        ('XY (Checkerboard)', lambda L: XYSimulation(size=L, temp=0.89, update='checkerboard')),
        ('XY (Random)', lambda L: XYSimulation(size=L, temp=0.89, update='random')),
        (
            'Clock Continuous (Checkerboard)',
            lambda L: ClockSimulation(size=L, temp=0.5, q=6, update='checkerboard'),
        ),
        (
            'Clock Continuous (Random)',
            lambda L: ClockSimulation(size=L, temp=0.5, q=6, update='random'),
        ),
        (
            'Clock Discrete (Checkerboard)',
            lambda L: DiscreteClockSimulation(size=L, temp=0.5, q=6, update='checkerboard'),
        ),
        (
            'Clock Discrete (Random)',
            lambda L: DiscreteClockSimulation(size=L, temp=0.5, q=6, update='random'),
        ),
    ]

    all_results: dict[str, dict[int, dict[str, float]]] = {name: {} for name, _ in model_configs}

    print(f'Starting scaling benchmark for sizes: {sizes}\n')

    for L in sizes:
        print(f'--- Lattice Size L = {L} (N = {L*L}) ---')
        for name, constructor in model_configs:
            print(f'Benchmarking {name}...', end=' ', flush=True)
            sim = constructor(L)
            metrics = measure_performance(sim, sweeps=args.sweeps)
            all_results[name][L] = metrics
            print(f"{metrics['sps']:.1f} sweeps/s")
        print()

    # --- Visualization ---
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Comprehensive Performance Scaling Analysis', fontsize=16)
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    for name in all_results:
        res = all_results[name]
        L_vals = sorted(list(res.keys()))
        N_vals = np.array([L * L for L in L_vals])

        # Panel 1: Sweeps per second vs N
        ax1.loglog(N_vals, [res[L]['sps'] for L in L_vals], 'o-', label=name)

        # Panel 2: Nanoseconds per site update
        ns_per_update = []
        for L in L_vals:
            t_sweep = 1.0 / res[L]['sps']
            ns_per_update.append(t_sweep / (L * L) * 1e9)
        ax2.semilogx(N_vals, ns_per_update, 's-', label=name)

        # Panel 3: Correlation function time vs N
        ax3.loglog(N_vals, [res[L]['corr_ms'] for L in L_vals], '^-', label=name)

        # Panel 4: Topological analysis cost vs N
        topo_ms = [res[L]['vort_ms'] + res[L]['vden_ms'] + res[L]['heli_ms'] for L in L_vals]
        ax4.loglog(N_vals, topo_ms, 'D-', label=name)

        # Panel 5: Thermodynamic measurement time vs N
        ax5.loglog(N_vals, [res[L]['thermo_ms'] for L in L_vals], 'v-', label=name)

        # Panel 6: Total Analysis Overhead Ratio
        overhead = []
        for L in L_vals:
            sweep_time_ms = (1.0 / res[L]['sps']) * 1000
            total_analysis_ms = (
                res[L]['thermo_ms']
                + res[L]['corr_ms']
                + res[L]['vort_ms']
                + res[L]['vden_ms']
                + res[L]['heli_ms']
            )
            overhead.append(total_analysis_ms / sweep_time_ms)
        ax6.plot(L_vals, overhead, 'P-', label=name)

    # Styling
    ax1.set_title('Throughput (Sweeps/sec)')
    ax1.set_xlabel('Number of sites N ($L^2$)')
    ax1.set_ylabel('Sweeps / sec')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend(fontsize='small', ncol=2)

    ax2.set_title('Cost per Spin Update')
    ax2.set_xlabel('Number of sites N')
    ax2.set_ylabel('Time (ns)')
    ax2.grid(True, which='both', alpha=0.3)

    ax3.set_title('Correlation Function Cost $G(r)$')
    ax3.set_xlabel('Number of sites N')
    ax3.set_ylabel('Time (ms)')
    ax3.grid(True, which='both', alpha=0.3)

    ax4.set_title('Topological Analysis Cost (Vort + Heli)')
    ax4.set_xlabel('Number of sites N')
    ax4.set_ylabel('Time (ms)')
    ax4.grid(True, which='both', alpha=0.3)

    ax5.set_title('Thermodynamic Measurement Cost (E + M)')
    ax5.set_xlabel('Number of sites N')
    ax5.set_ylabel('Time (ms)')
    ax5.grid(True, which='both', alpha=0.3)

    ax6.set_title('Total Analysis Overhead Ratio')
    ax6.set_xlabel('Lattice Size L')
    ax6.set_ylabel('Ratio (Analysis / Sweep)')
    ax6.axhline(1.0, color='black', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    ensure_results_dir(directory=args.output_dir)
    save_plot(filename='scaling_benchmark.png', directory=args.output_dir)

    # Save all metrics to NPZ for use by notebooks/Performance_Benchmarks.ipynb.
    # Flat layout: sizes (1D int64), model_names (1D str), and one 2D float64
    # array per metric with shape (n_models, n_sizes), rows ordered to match
    # model_names. This mirrors the pattern used by wolff_efficiency.py.
    model_names_list = list(all_results.keys())
    metric_keys = ('sps', 'thermo_ms', 'corr_ms', 'vort_ms', 'vden_ms', 'heli_ms')
    n_models = len(model_names_list)
    n_sizes = len(sizes)
    metric_arrays: dict[str, np.ndarray] = {
        m: np.zeros((n_models, n_sizes), dtype=np.float64) for m in metric_keys
    }
    for i, name in enumerate(model_names_list):
        for j, L in enumerate(sizes):
            for key in metric_keys:
                metric_arrays[key][i, j] = all_results[name][L][key]
    npz_path = os.path.join(args.output_dir, 'scaling_benchmark.npz')
    np.savez(
        npz_path,
        sizes=np.array(sizes, dtype=np.int64),
        model_names=np.array(model_names_list),
        sps=metric_arrays['sps'],
        thermo_ms=metric_arrays['thermo_ms'],
        corr_ms=metric_arrays['corr_ms'],
        vort_ms=metric_arrays['vort_ms'],
        vden_ms=metric_arrays['vden_ms'],
        heli_ms=metric_arrays['heli_ms'],
    )
    print(f'Data saved to {npz_path}')

    # Print summary table for largest size
    L_max = sizes[-1]
    print(f'\nFinal Performance Table (L={L_max}):')
    print('=' * 125)
    row_fmt = '{:<32} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}'
    print(row_fmt.format(
        'Model', 'Sweeps/s', 'ns/site', 'Thermo(ms)', 'Corr(ms)', 'Topo(ms)', 'Overhead'
    ))
    print('-' * 125)
    for name in all_results:
        m = all_results[name][L_max]
        sw_ms = (1.0 / m['sps']) * 1000
        ns_site = sw_ms / (L_max * L_max) * 1e6
        topo_total_ms: float = m['vort_ms'] + m['vden_ms'] + m['heli_ms']
        ratio = (m['thermo_ms'] + m['corr_ms'] + topo_total_ms) / sw_ms
        print(
            f'{name:<32} | {m["sps"]:>10.1f} | {ns_site:>10.2f} | '
            f'{m["thermo_ms"]:>10.3f} | {m["corr_ms"]:>10.3f} | '
            f'{topo_total_ms:>10.3f} | {ratio:>10.2f}x'
        )
    print('=' * 125)


if __name__ == '__main__':
    run_scaling_benchmark()
