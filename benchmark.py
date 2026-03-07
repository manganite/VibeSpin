"""
Benchmark script for Multiferroic Simulation Project.
Measures performance of Monte Carlo sweeps and analysis functions.
"""

import time
import numpy as np
from models.ising_model import IsingSimulation
from models.xy_model import XYSimulation
from models.clock_model import ClockSimulation

def run_benchmark():
    L = 128
    steps = 1000
    analysis_iters = 100
    
    print(f"Starting Benchmark (L={L}, sweeps={steps})...\n")
    
    models = [
        ("Ising (Checkerboard)", IsingSimulation(L, 2.269, update='checkerboard')),
        ("Ising (Random)", IsingSimulation(L, 2.269, update='random')),
        ("XY Model", XYSimulation(L, 0.89)),
        ("Clock Model (q=6)", ClockSimulation(L, 0.5, q=6))
    ]
    
    results = []
    
    for name, sim in models:
        print(f"Benchmarking {name}...")
        
        # Warm-up (ensure JIT compilation)
        sim.step()
        sim._calculate_correlation_function()
        if hasattr(sim, '_calculate_vorticity'):
            sim._calculate_vorticity()
            
        # 1. Measure Sweep Speed
        start = time.perf_counter()
        for _ in range(steps):
            sim.step()
        end = time.perf_counter()
        duration = end - start
        sps = steps / duration
        
        # 2. Measure Correlation Function Speed
        start_corr = time.perf_counter()
        for _ in range(analysis_iters):
            sim._calculate_correlation_function()
        end_corr = time.perf_counter()
        corr_ms = ((end_corr - start_corr) / analysis_iters) * 1000
        
        # 3. Measure Vorticity Speed (if applicable)
        vort_ms = 0.0
        if hasattr(sim, '_calculate_vorticity'):
            start_vort = time.perf_counter()
            for _ in range(analysis_iters):
                sim._calculate_vorticity()
            end_vort = time.perf_counter()
            vort_ms = ((end_vort - start_vort) / analysis_iters) * 1000
            
        results.append({
            "name": name,
            "sps": sps,
            "corr_ms": corr_ms,
            "vort_ms": vort_ms
        })

    print("\n" + "="*65)
    print(f"{'Model':<25} | {'Sweeps/sec':<12} | {'G(r) (ms)':<10} | {'Vort (ms)':<10}")
    print("-"*65)
    for res in results:
        vort_str = f"{res['vort_ms']:>9.2f}" if res['vort_ms'] > 0 else f"{'N/A':>9}"
        print(f"{res['name']:<25} | {res['sps']:>11.1f} | {res['corr_ms']:>9.2f} | {vort_str}")
    print("="*65)

if __name__ == "__main__":
    run_benchmark()
