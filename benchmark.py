"""
Comprehensive scaling benchmark for the VibeSpin Simulation Project.
Measures performance of MC sweeps and analysis functions across different lattice sizes.
"""

import argparse
import time

import matplotlib.pyplot as plt

from models.clock_model import ClockSimulation
from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation
from utils.system_helpers import ensure_results_dir, save_plot


def measure_performance(sim, sweeps=100, analysis_iters=50):
    """Measure sweeps/sec and analysis times for a simulation instance."""
    # Warm-up
    sim.step()
    sim._get_energy()
    sim._calculate_correlation_function()
    if hasattr(sim, '_calculate_vorticity'):
        sim._calculate_vorticity()

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

    # 4. Vorticity (if applicable)
    vort_ms = 0.0
    if hasattr(sim, '_calculate_vorticity'):
        start = time.perf_counter()
        for _ in range(analysis_iters):
            sim._calculate_vorticity()
        vort_ms = (time.perf_counter() - start) / analysis_iters * 1000

    return {'sps': sps, 'thermo_ms': thermo_ms, 'corr_ms': corr_ms, 'vort_ms': vort_ms}


def run_scaling_benchmark():
    parser = argparse.ArgumentParser(description='Scaling Benchmark')
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=[32, 64, 128, 256], help='Lattice sizes'
    )
    parser.add_argument('--sweeps', type=int, default=200, help='Sweeps per point')
    parser.add_argument(
        '--output-dir', type=str, default='results/benchmarks', help='Output directory'
    )
    args = parser.parse_args()

    sizes = sorted(args.sizes)
    model_configs = [
        ('Ising (Checker)', lambda L: IsingSimulation(size=L, temp=2.269, update='checkerboard')),
        ('Ising (Random)', lambda L: IsingSimulation(size=L, temp=2.269, update='random')),
        ('XY (Checker)', lambda L: XYSimulation(size=L, temp=0.89, update='checkerboard')),
        ('XY (Random)', lambda L: XYSimulation(size=L, temp=0.89, update='random')),
        (
            'Clock (Checker)',
            lambda L: ClockSimulation(size=L, temp=0.5, q=6, update='checkerboard'),
        ),
        ('Clock (Random)', lambda L: ClockSimulation(size=L, temp=0.5, q=6, update='random')),
    ]

    # Results structure: results[model_name][size] = metrics_dict
    all_results = {name: {} for name, _ in model_configs}

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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Scaling Analysis', fontsize=16)

    for name in all_results:
        res = all_results[name]
        L_vals = sorted(list(res.keys()))
        N_vals = [L * L for L in L_vals]

        # Panel 1: Sweeps per second vs N
        ax1.loglog(N_vals, [res[L]['sps'] for L in L_vals], 'o-', label=name)

        # Panel 2: Thermodynamic measurement time vs N
        ax2.loglog(N_vals, [res[L]['thermo_ms'] for L in L_vals], 's-', label=name)

        # Panel 3: Correlation function time vs N
        ax3.loglog(N_vals, [res[L]['corr_ms'] for L in L_vals], '^-', label=name)

        # Panel 4: Analysis overhead ratio (Analysis time / Sweep time)
        overhead = []
        for L in L_vals:
            sweep_time_ms = (1.0 / res[L]['sps']) * 1000
            total_analysis_ms = res[L]['thermo_ms'] + res[L]['corr_ms'] + res[L]['vort_ms']
            overhead.append(total_analysis_ms / sweep_time_ms)
        ax4.plot(L_vals, overhead, 'D-', label=name)

    # Styling
    ax1.set_title('Throughput (Sweeps/sec)')
    ax1.set_xlabel('Number of sites N ($L^2$)')
    ax1.set_ylabel('Sweeps / sec')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    ax2.set_title('Thermodynamic Measurement Cost')
    ax2.set_xlabel('Number of sites N')
    ax2.set_ylabel('Time (ms)')
    ax2.grid(True, which='both', alpha=0.3)

    ax3.set_title('Correlation Function Cost $G(r)$')
    ax3.set_xlabel('Number of sites N')
    ax3.set_ylabel('Time (ms)')
    ax3.grid(True, which='both', alpha=0.3)

    ax4.set_title('Analysis Overhead Ratio')
    ax4.set_xlabel('Lattice Size L')
    ax4.set_ylabel('Ratio (Analysis Time / Sweep Time)')
    ax4.grid(True, alpha=0.3)

    ensure_results_dir(directory=args.output_dir)
    save_plot(filename='scaling_benchmark.png', directory=args.output_dir)

    # Print summary table for largest size
    L_max = sizes[-1]
    print(f'\nFinal Performance Table (L={L_max}):')
    print('=' * 85)
    row_fmt = '{:<20} | {:>12} | {:>12} | {:>12} | {:>10}'
    print(row_fmt.format('Model', 'Sweeps/s', 'Thermo (ms)', 'G(r) (ms)', 'Overhead'))
    print('-' * 85)
    for name in all_results:
        m = all_results[name][L_max]
        sw_ms = (1.0 / m['sps']) * 1000
        ratio = (m['thermo_ms'] + m['corr_ms'] + m['vort_ms']) / sw_ms
        print(
            f'{name:<20} | {m["sps"]:>12.1f} | {m["thermo_ms"]:>12.3f} | '
            f'{m["corr_ms"]:>12.3f} | {ratio:>10.2f}x'
        )
    print('=' * 85)


if __name__ == '__main__':
    run_scaling_benchmark()
