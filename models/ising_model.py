"""
2D Ising Model simulation using the Metropolis-Hastings algorithm.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from .simulation_base import MonteCarloSimulation


@njit
def ising_step_numba(spins: np.ndarray, beta: float, J: float) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the Ising lattice.
    Uses a checkerboard update pattern for better Numba optimization.

    Args:
        spins: (N, N) array of spins (+1 or -1).
        beta: Inverse temperature 1/kT.
        J: Coupling constant.

    Returns:
        Updated spins array.
    """
    N = spins.shape[0]
    # Pre-calculate transition probabilities for dE > 0
    # dE can be 4J or 8J
    prob4 = np.exp(-4.0 * J * beta)
    prob8 = np.exp(-8.0 * J * beta)

    for parity in range(2):
        for i in range(N):
            # Use striding to avoid 'if' condition in the inner loop
            start_j = (parity + i) % 2
            for j in range(start_j, N, 2):
                i_prev = N - 1 if i == 0 else i - 1
                i_next = 0 if i == N - 1 else i + 1
                j_prev = N - 1 if j == 0 else j - 1
                j_next = 0 if j == N - 1 else j + 1

                neighbor_sum = (spins[i_prev, j] + spins[i_next, j] +
                                spins[i, j_prev] + spins[i, j_next])

                dE = 2 * J * spins[i, j] * neighbor_sum

                if dE <= 0:
                    spins[i, j] *= -1
                else:
                    # Optimized probability check
                    p = prob4 if dE == 4.0 * J else prob8
                    if np.random.random() < p:
                        spins[i, j] *= -1
    return spins

@njit
def ising_energy_numba(spins: np.ndarray, J: float) -> float:
    """
    Calculate the total energy of the Ising lattice.

    Args:
        spins: (N, N) array of spins.
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
            # Sum unique pairs (right and down) to avoid double counting
            energy -= J * spins[i, j] * (spins[i_next, j] + spins[i, j_next])
    return energy / (N * N)

@njit
def ising_step_random_numba(spins: np.ndarray, beta: float, J: float) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the Ising lattice using random
    sequential updates (N^2 randomly chosen single-spin flip attempts).

    Unlike the checkerboard sweep, sites are chosen uniformly at random with
    replacement, giving a more physical stochastic dynamics with no spatial
    bias.  This is the standard Metropolis single-spin flip algorithm.

    Boltzmann factors for the two possible positive energy changes (4J, 8J)
    are pre-calculated outside the inner loop to avoid repeated calls to
    ``np.exp``, matching the optimisation used in ``ising_step_numba``.

    Args:
        spins: (N, N) array of spins (+1 or -1).
        beta: Inverse temperature 1/kT.
        J: Coupling constant.

    Returns:
        Updated spins array.
    """
    N = spins.shape[0]
    # Pre-calculate the two possible acceptance probabilities for dE > 0.
    # On a 2-D square lattice dE = 2*J*s_i*Σneighbours ∈ {-8J,-4J,0,4J,8J},
    # so only prob4 and prob8 are ever needed.
    prob4 = np.exp(-4.0 * J * beta)
    prob8 = np.exp(-8.0 * J * beta)

    for _ in range(N * N):
        idx = np.random.randint(0, N * N)
        i = idx // N
        j = idx % N
        i_prev = N - 1 if i == 0 else i - 1
        i_next = 0 if i == N - 1 else i + 1
        j_prev = N - 1 if j == 0 else j - 1
        j_next = 0 if j == N - 1 else j + 1

        neighbor_sum = (spins[i_prev, j] + spins[i_next, j] +
                        spins[i, j_prev] + spins[i, j_next])
        dE = 2 * J * spins[i, j] * neighbor_sum

        if dE <= 0:
            spins[i, j] *= -1
        else:
            p = prob4 if dE == 4.0 * J else prob8
            if np.random.random() < p:
                spins[i, j] *= -1
    return spins


class IsingSimulation(MonteCarloSimulation):
    """
    Simulation of the 2D Ising model on a square lattice.
    """
    _VALID_UPDATES: frozenset = frozenset({'checkerboard', 'random'})

    def __init__(self, size: int, temp: float, J: float = 1.0,
                 update: str = 'checkerboard'):
        """
        Initialize the Ising simulation.

        Args:
            size: Linear dimension L of the L x L lattice.
            temp: Temperature T.
            J: Coupling constant (default 1.0).
            update: Update scheme — ``'checkerboard'`` (default, faster) or
                ``'random'`` (random sequential Metropolis, more physical
                stochastic dynamics for coarsening studies).

        Raises:
            ValueError: If ``update`` is not one of the recognised schemes.
        """
        super().__init__(size, temp)
        if update not in self._VALID_UPDATES:
            raise ValueError(
                f"Unknown update scheme {update!r}. "
                f"Valid options: {sorted(self._VALID_UPDATES)}"
            )
        self.J = J
        self.update = update
        # Initialize random spins +1 or -1
        self.spins = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))

    def step(self) -> None:
        """Perform one Monte Carlo sweep using the configured update scheme."""
        if self.spins is not None:
            if self.update == 'random':
                self.spins = ising_step_random_numba(self.spins, self.beta, self.J)
            else:
                self.spins = ising_step_numba(self.spins, self.beta, self.J)
        self.steps += 1

    def _get_magnetization(self) -> float:
        """Calculate magnetization per spin."""
        if self.spins is not None:
            return float(np.abs(np.sum(self.spins)) / (self.size**2))
        return 0.0

    def _get_energy(self) -> float:
        """Calculate energy per spin of the lattice."""
        if self.spins is not None:
            return ising_energy_numba(self.spins, self.J)
        return 0.0

    def _get_structure_factor_squared_unshifted(self) -> np.ndarray:
        """Calculate the unshifted squared magnitude of the Fourier transform."""
        if self.spins is not None:
            Sk = np.fft.fft2(self.spins)
            return np.abs(Sk)**2
        return np.array([])

if __name__ == "__main__":
    # Parameters
    L = 50      # Lattice size (L x L)
    T = 2.269    # Temperature (critical point approx 2.269)
    STEPS = 1000

    print(f"Initializing Ising Model (L={L}, T={T})...")
    sim = IsingSimulation(L, T)

    print(f"Running for {STEPS} steps...")
    mag_history, energy_history = sim.run(STEPS)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Final Configuration
    if sim.spins is not None:
        ax1.imshow(sim.spins, cmap='gray', interpolation='none')
    ax1.set_title(f'Spin Configuration after {STEPS} steps')
    ax1.axis('off')

    # Magnetization and Energy
    ax2.plot(mag_history, label='Magnetization')
    ax2.plot(energy_history, label='Energy')
    ax2.set_title('Observables vs Time')
    ax2.set_xlabel('Monte Carlo Steps')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    ax2.legend()

    # Correlation
    r, G_r = sim._calculate_correlation_function()
    ax3.plot(r, G_r)
    ax3.set_title('Spin-Spin Correlation G(r)')
    ax3.set_xlabel('Distance r')
    ax3.set_ylabel('G(r)')
    ax3.grid(True)
    ax3.set_yscale('log')

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ising_simulation.png')
    plt.savefig(output_file)
    print(f"Simulation finished. Plot saved to {output_file}")
