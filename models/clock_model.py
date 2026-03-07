"""
2D q-state Clock Model simulation using the Metropolis-Hastings algorithm.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from .simulation_base import (
    MonteCarloSimulation,
    calculate_vorticity_numba,
    get_helicity_data_numba,
)


@njit(cache=True)
def clock_step_numba(
    spins: np.ndarray, beta: float, J: float, A: float, q: int, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the Clock Model lattice.
    Uses a checkerboard update pattern for better Numba optimization.

    Args:
        spins: (N, N, 2) array of unit vectors.
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        A: Anisotropy strength.
        q: Number of clock states.
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

    Returns:
        Updated spins array.
    """
    N = spins.shape[0]
    for parity in range(2):
        for i in range(N):
            # Use striding to avoid 'if' condition in the inner loop
            start_j = (parity + i) % 2
            inxt = idx_next[i]
            iprv = idx_prev[i]
            for j in range(start_j, N, 2):
                jnxt = idx_next[j]
                jprv = idx_prev[j]

                # Neighbor sum
                nx = (
                    spins[iprv, j, 0]
                    + spins[inxt, j, 0]
                    + spins[i, jprv, 0]
                    + spins[i, jnxt, 0]
                )
                ny = (
                    spins[iprv, j, 1]
                    + spins[inxt, j, 1]
                    + spins[i, jprv, 1]
                    + spins[i, jnxt, 1]
                )

                sx, sy = spins[i, j, 0], spins[i, j, 1]

                # Propose update
                delta = np.random.uniform(-0.5, 0.5)
                c, s = np.cos(delta), np.sin(delta)
                sx_new = sx * c - sy * s
                sy_new = sx * s + sy * c
                norm = np.sqrt(sx_new**2 + sy_new**2)
                sx_new /= norm
                sy_new /= norm

                # Interaction Energy Change
                dE_inter = -J * ((sx_new * nx + sy_new * ny) - (sx * nx + sy * ny))

                # Anisotropy Energy Change
                phi_old = np.arctan2(sy, sx)
                phi_new = np.arctan2(sy_new, sx_new)
                dE_aniso = -A * (np.cos(q * phi_new) - np.cos(q * phi_old))

                dE = dE_inter + dE_aniso

                if dE <= 0 or np.random.random() < np.exp(-dE * beta):
                    spins[i, j, 0] = sx_new
                    spins[i, j, 1] = sy_new
    return spins


@njit(cache=True)
def clock_energy_numba(spins: np.ndarray, J: float, A: float, q: int, idx_next: np.ndarray) -> float:
    """
    Calculate the total energy of the Clock Model lattice.

    Args:
        spins: (N, N, 2) array of unit vectors.
        J: Coupling constant.
        A: Anisotropy strength.
        q: Number of clock states.
        idx_next: Pre-calculated next-neighbor indices.

    Returns:
        energy: Total energy per site.
    """
    N = spins.shape[0]
    energy = 0.0
    for i in range(N):
        inxt = idx_next[i]
        for j in range(N):
            jnxt = idx_next[j]
            # Interaction
            dot_right = (
                spins[i, j, 0] * spins[i, jnxt, 0] + spins[i, j, 1] * spins[i, jnxt, 1]
            )
            dot_down = (
                spins[i, j, 0] * spins[inxt, j, 0] + spins[i, j, 1] * spins[inxt, j, 1]
            )
            energy -= J * (dot_right + dot_down)

            # Anisotropy
            phi = np.arctan2(spins[i, j, 1], spins[i, j, 0])
            energy -= A * np.cos(q * phi)
    return energy / (N * N)


class ClockSimulation(MonteCarloSimulation):
    """
    Simulation of the 2D q-state clock model on a square lattice.
    """

    def __init__(
        self, size: int, temp: float, J: float = 1.0, A: float = 1.0, q: int = 6, seed: int | None = None
    ):
        """
        Initialize the Clock Model simulation.

        Args:
            size: Linear dimension L of the L x L lattice.
            temp: Temperature T.
            J: Coupling constant (default 1.0).
            A: Anisotropy strength (default 1.0).
            q: Number of clock states (default 6). Must be ≥ 2.
            seed: Optional random seed for reproducibility.

        Raises:
            ValueError: If ``q`` is less than 2.
        """
        super().__init__(size, temp, seed=seed)
        if q < 2:
            raise ValueError(f"q must be >= 2 (number of clock states), got {q}")
        self.J = J
        self.A = A
        self.q = q

        # Initialize random spins as 2D unit vectors
        # spin = (spin_x, spin_y)
        angles = self.rng.uniform(0, 2 * np.pi, size=(size, size))
        self.spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def step(self) -> None:
        """Perform one Monte Carlo step using Numba."""
        if self.spins is not None:
            if self.seed is not None:
                from .simulation_base import _seed_numba

                _seed_numba(self.seed + self.steps)
            self.spins = clock_step_numba(
                self.spins, self.beta, self.J, self.A, self.q, self.idx_next, self.idx_prev
            )
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
            return float(
                clock_energy_numba(self.spins, self.J, self.A, self.q, self.idx_next)
            )  # type: ignore[no-any-return]
        return 0.0

    def _calculate_vorticity(self) -> np.ndarray:
        """Calculate the vorticity (winding number) of each plaquette."""
        if self.spins is not None:
            return np.asarray(calculate_vorticity_numba(self.spins))  # type: ignore[no-any-return]
        return np.array([])

    def _get_helicity_data(self) -> tuple[float, float]:
        """Calculate sum of cos and sin of angle differences in x-direction."""
        if self.spins is not None:
            cos_sum, sin_sum = get_helicity_data_numba(self.spins)
            return float(cos_sum), float(sin_sum)  # type: ignore[no-any-return]
        return 0.0, 0.0

    def _get_structure_factor_squared_unshifted(self) -> np.ndarray:
        """Calculate the unshifted squared magnitude of the Fourier transform."""
        if self.spins is not None:
            sx = self.spins[..., 0]
            sy = self.spins[..., 1]
            Sk_x = np.fft.fft2(sx)
            Sk_y = np.fft.fft2(sy)
            return np.asarray(np.abs(Sk_x) ** 2 + np.abs(Sk_y) ** 2)  # type: ignore[no-any-return]
        return np.array([])


if __name__ == "__main__":
    # Parameters
    L = 50  # Lattice size (L x L)
    T = 0.5  # Temperature
    STEPS = 1000
    A = 1.0  # Anisotropy strength
    q = 6  # q-state clock model

    print(f"Initializing {q}-state Clock Model (L={L}, T={T}, A={A})...")
    sim = ClockSimulation(L, T, A=A, q=q)

    print(f"Running for {STEPS} steps...")
    mag_history, energy_history = sim.run(STEPS)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Final Configuration (Phase angle)
    if sim.spins is not None:
        angles = np.arctan2(sim.spins[..., 1], sim.spins[..., 0])
        im = ax1.imshow(angles, cmap='hsv', interpolation='none')
        ax1.set_title(f'Spin Phase after {STEPS} steps')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, label='Phase (radians)')

    # Magnetization and Energy
    ax2.plot(mag_history, label='Magnetization |M|')
    ax2.plot(energy_history, label='Energy')
    ax2.set_title('Observables vs Time')
    ax2.set_xlabel('Monte Carlo Steps')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    ax2.legend()

    # Vorticity map
    vorticity = sim._calculate_vorticity()
    im_vortex = ax3.imshow(vorticity, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
    ax3.set_title('Vorticity Map')
    ax3.axis('off')
    plt.colorbar(im_vortex, ax=ax3, ticks=[-1, 0, 1], label='Winding Number')

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'clock_simulation.png')
    plt.savefig(output_file)
    print(f"Simulation finished. Plot saved to {output_file}")


