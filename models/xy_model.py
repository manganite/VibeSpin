"""
2D XY Model simulation using the Metropolis-Hastings algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm
from numba import njit
from .simulation_base import MonteCarloSimulation, calculate_vorticity_numba, get_helicity_data_numba

@njit
def xy_step_numba(spins: np.ndarray, beta: float, J: float) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the XY lattice.
    Uses a checkerboard update pattern for better Numba optimization.

    Args:
        spins: (N, N, 2) array of unit vectors.
        beta: Inverse temperature 1/kT.
        J: Coupling constant.

    Returns:
        Updated spins array.
    """
    N = spins.shape[0]
    for parity in range(2):
        for i in range(N):
            # Use striding to avoid 'if' condition in the inner loop
            start_j = (parity + i) % 2
            for j in range(start_j, N, 2):
                i_prev = N - 1 if i == 0 else i - 1
                i_next = 0 if i == N - 1 else i + 1
                j_prev = N - 1 if j == 0 else j - 1
                j_next = 0 if j == N - 1 else j + 1

                # Neighbor sum vector
                nx = (spins[i_prev, j, 0] + spins[i_next, j, 0] +
                      spins[i, j_prev, 0] + spins[i, j_next, 0])
                ny = (spins[i_prev, j, 1] + spins[i_next, j, 1] +
                      spins[i, j_prev, 1] + spins[i, j_next, 1])

                # Current spin
                sx = spins[i, j, 0]
                sy = spins[i, j, 1]

                # Propose update
                delta = np.random.uniform(-0.5, 0.5)
                c, s = np.cos(delta), np.sin(delta)

                sx_new = sx * c - sy * s
                sy_new = sx * s + sy * c

                # Normalize
                norm = np.sqrt(sx_new**2 + sy_new**2)
                sx_new /= norm
                sy_new /= norm

                # Energy change: -J * (s_new - s_old) . neighbors
                dE = -J * ((sx_new * nx + sy_new * ny) - (sx * nx + sy * ny))

                if dE <= 0 or np.random.random() < np.exp(-dE * beta):
                    spins[i, j, 0] = sx_new
                    spins[i, j, 1] = sy_new
    return spins

@njit
def xy_energy_numba(spins: np.ndarray, J: float) -> float:
    """
    Calculate the total energy of the XY lattice.

    Args:
        spins: (N, N, 2) array of unit vectors.
        J: Coupling constant.

    Returns:
        energy: Total energy per site.
    """
    N = spins.shape[0]
    energy = 0.0
    for i in range(N):
        for j in range(N):
            i_next = 0 if i == N - 1 else i + 1
            j_next = 0 if j == N - 1 else j + 1
            # Sum unique pairs (right and down)
            dot_right = spins[i, j, 0]*spins[i, j_next, 0] + spins[i, j, 1]*spins[i, j_next, 1]
            dot_down = spins[i, j, 0]*spins[i_next, j, 0] + spins[i, j, 1]*spins[i_next, j, 1]
            energy -= J * (dot_right + dot_down)
    return energy / (N * N)

class XYSimulation(MonteCarloSimulation):
    """
    Simulation of the 2D XY model on a square lattice.
    """
    def __init__(self, size: int, temp: float, J: float = 1.0):
        """
        Initialize the XY simulation.

        Args:
            size: Linear dimension L of the L x L lattice.
            temp: Temperature T.
            J: Coupling constant (default 1.0).
        """
        super().__init__(size, temp)
        self.J = J

        # Initialize random spins as 2D unit vectors
        # spin = (spin_x, spin_y)
        angles = np.random.uniform(0, 2*np.pi, size=(size, size))
        self.spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def step(self) -> None:
        """Perform one Monte Carlo step using Numba."""
        if self.spins is not None:
            self.spins = xy_step_numba(self.spins, self.beta, self.J)
        self.steps += 1

    def _get_magnetization(self) -> float:
        """Calculate magnetization magnitude per spin."""
        if self.spins is not None:
            total_spin = np.sum(self.spins, axis=(0, 1))
            return float(np.linalg.norm(total_spin) / (self.size**2))
        return 0.0

    def _get_energy(self) -> float:
        """Calculate energy per spin."""
        if self.spins is not None:
            return xy_energy_numba(self.spins, self.J)
        return 0.0

    def _calculate_vorticity(self) -> np.ndarray:
        """Calculate the vorticity (winding number) of each plaquette."""
        if self.spins is not None:
            return calculate_vorticity_numba(self.spins)
        return np.array([])

    def _get_helicity_data(self) -> tuple[float, float]:
        """Calculate sum of cos and sin of angle differences in x-direction."""
        if self.spins is not None:
            return get_helicity_data_numba(self.spins)
        return 0.0, 0.0

    def _get_structure_factor_squared_unshifted(self) -> np.ndarray:
        """Calculate the unshifted squared magnitude of the Fourier transform."""
        if self.spins is not None:
            sx = self.spins[..., 0]
            sy = self.spins[..., 1]
            Sk_x = np.fft.fft2(sx)
            Sk_y = np.fft.fft2(sy)
            return np.abs(Sk_x)**2 + np.abs(Sk_y)**2
        return np.array([])

if __name__ == "__main__":
    # Parameters
    L = 50      # Lattice size (L x L)
    T = 0.89    # Temperature (BKT transition approx 0.89)
    STEPS = 1000

    print(f"Initializing XY Model (L={L}, T={T})...")
    sim = XYSimulation(L, T)

    print(f"Running for {STEPS} steps...")
    mag_history, energy_history = sim.run(STEPS)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Final Configuration (Phase angle)
    if sim.spins is not None:
        angles = np.arctan2(sim.spins[..., 1], sim.spins[..., 0])
        im = ax1.imshow(angles, cmap='hsv', interpolation='none')
        ax1.set_title(f'Spin Phase after {STEPS} steps')
        ax1.axis('off')
        fig.colorbar(im, ax=ax1, label='Phase (radians)', shrink=0.8)

    # Vorticity map
    vorticity = sim._calculate_vorticity()
    im_vortex = ax2.imshow(vorticity, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
    ax2.set_title('Vorticity Map')
    ax2.axis('off')
    fig.colorbar(im_vortex, ax=ax2, ticks=[-1, 0, 1], label='Winding Number', shrink=0.8)

    # Spin-Spin Correlation Function
    r, G_r = sim._calculate_correlation_function()
    ax3.plot(r, G_r)
    ax3.set_title('Spin-Spin Correlation G(r)')
    ax3.set_xlabel('Distance r (pixels)')
    ax3.set_ylabel('G(r)')
    ax3.grid(True)
    ax3.set_yscale('log')

    # Structure Factor
    S_k = sim._calculate_structure_factor()
    im_sk = ax4.imshow(S_k, cmap='viridis', norm=LogNorm())
    ax4.set_title('Structure Factor S(k)')
    ax4.set_xlabel('$k_x$')
    ax4.set_ylabel('$k_y$')
    fig.colorbar(im_sk, ax=ax4, label='S(k)', shrink=0.8)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'xy_simulation.png')
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Simulation finished. Plot saved to {output_file}")
