"""
Topological analysis of the 2D XY model, focusing on the BKT transition.
Calculates vortex proliferation as a function of temperature.
"""

import numpy as np
import matplotlib.pyplot as plt

from models.xy_model import XYSimulation
from utils.system_helpers import parallel_sweep, save_plot


# Simulation Parameters
L: int = 64
EQUILIBRATION_STEPS: int = 5000
MEASUREMENT_STEPS: int = 2000

# Sweep Parameters
T_MIN: float = 0.1
T_MAX: float = 1.5
T_POINTS: int = 30
T_BKT_THEORETICAL: float = 0.89

def simulate_bkt_point(T: float) -> float:
    """
    Worker function to simulate a single temperature and count average vortex density.
    
    Args:
        T: Temperature.
        
    Returns:
        Average number of vortices found in the lattice.
    """
    sim = XYSimulation(L, T)
    sim.equilibrate(EQUILIBRATION_STEPS)
    
    total_vortex_count: int = 0
    for _ in range(MEASUREMENT_STEPS):
        sim.step()
        # Calculate vorticity map and accumulate vortex count
        vorticity: np.ndarray = sim._calculate_vorticity()
        total_vortex_count += int(np.count_nonzero(vorticity))

    return total_vortex_count / MEASUREMENT_STEPS

def run_bkt_sweep() -> None:
    """
    Run parallel sweep to analyze vortex proliferation across the BKT transition.
    """
    # Generate temperature points
    temperatures: np.ndarray = np.linspace(T_MIN, T_MAX, T_POINTS)
    
    print(f"Starting BKT transition study (Vortex counting) for L={L}...")
    print(f"Range: [{T_MIN}, {T_MAX}] with {T_POINTS} points.")
    
    vortex_counts: list[float] = parallel_sweep(simulate_bkt_point, temperatures)
        
    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, vortex_counts, 'o-', markersize=5, label='Total Vortex Count')
    
    plt.axvline(x=T_BKT_THEORETICAL, color='r', linestyle='--', alpha=0.7, label=f'Theoretical $T_{{BKT}} \\approx {T_BKT_THEORETICAL}$')
    plt.xlabel('Temperature (T)')
    plt.ylabel('Average Number of Vortices')
    plt.title('Vortex Proliferation across BKT Transition')
    plt.grid(True)
    plt.legend()
    
    save_plot('bkt_transition.png', directory='results/xy')

if __name__ == "__main__":
    run_bkt_sweep()
