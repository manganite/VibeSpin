from __future__ import annotations
"""
2D Ising Model simulation using the Metropolis-Hastings algorithm.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from .simulation_base import MonteCarloSimulation


@njit(cache=True, fastmath=True)
def ising_step_numba(
    *, spins: np.ndarray, beta: float, J: float, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the Ising lattice.
    Uses a checkerboard update pattern for better Numba optimization.

    Args:
        spins: (N, N) array of spins (+1 or -1).
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

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
                # Using pre-calculated indices for PBCs
                inxt = idx_next[i]
                iprv = idx_prev[i]
                jnxt = idx_next[j]
                jprv = idx_prev[j]

                neighbor_sum = spins[iprv, j] + spins[inxt, j] + spins[i, jprv] + spins[i, jnxt]

                dE = 2 * J * spins[i, j] * neighbor_sum

                if dE <= 0:
                    spins[i, j] *= -1
                else:
                    # Optimized probability check
                    p = prob4 if dE == 4.0 * J else prob8
                    if np.random.random() < p:
                        spins[i, j] *= -1
    return spins


@njit(cache=True, fastmath=True)
def ising_energy_numba(*, spins: np.ndarray, J: float, idx_next: np.ndarray) -> float:
    """
    Calculate the total energy of the Ising lattice.

    Args:
        spins: (N, N) array of spins.
        J: Coupling constant.
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
            # Sum unique pairs (right and down) to avoid double counting
            energy -= J * spins[i, j] * (spins[inxt, j] + spins[i, jnxt])
    return energy / (N * N)


@njit(cache=True, fastmath=True)
def ising_step_random_numba(
    *, spins: np.ndarray, beta: float, J: float, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
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
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

    Returns:
        Updated spins array.
    """
    N = spins.shape[0]
    # Pre-calculate the two possible acceptance probabilities for dE > 0.
    prob4 = np.exp(-4.0 * J * beta)
    prob8 = np.exp(-8.0 * J * beta)

    for _ in range(N * N):
        idx = np.random.randint(0, N * N)
        i = idx // N
        j = idx % N

        inxt = idx_next[i]
        iprv = idx_prev[i]
        jnxt = idx_next[j]
        jprv = idx_prev[j]

        neighbor_sum = spins[iprv, j] + spins[inxt, j] + spins[i, jprv] + spins[i, jnxt]
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

    def __init__(
        self,
        *,
        size: int,
        temp: float,
        J: float = 1.0,
        update: str = 'checkerboard',
        seed: int | None = None,
    ):
        """
        Initialize the Ising simulation.

        Args:
            size: Linear dimension L of the L x L lattice.
            temp: Temperature T.
            J: Coupling constant (default 1.0).
            update: Update scheme — ``'checkerboard'`` (default, faster) or
                ``'random'`` (random sequential Metropolis, more physical
                stochastic dynamics for coarsening studies).
            seed: Optional random seed for reproducibility.

        Raises:
            ValueError: If ``update`` is not one of the recognised schemes.
        """
        super().__init__(size=size, temp=temp, seed=seed)
        if update not in self._VALID_UPDATES:
            valid_opts = sorted(self._VALID_UPDATES)
            raise ValueError(f'Unknown update scheme {update!r}. Valid options: {valid_opts}')
        self.J = J
        self.update = update
        # Initialize random spins +1 or -1
        self.spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))

    def step(self) -> None:
        """Perform one Monte Carlo sweep using the configured update scheme."""
        if self.spins is not None:
            # For Numba compatibility with reproducibility, we seed numba's
            # random generator if a seed was provided.
            if self.seed is not None:
                from .simulation_base import _seed_numba

                _seed_numba(seed=self.seed + self.steps)

            if self.update == 'random':
                self.spins = ising_step_random_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
            else:
                self.spins = ising_step_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
        self.steps += 1

    def _get_magnetization(self) -> float:
        """Calculate magnetization per spin."""
        if self.spins is not None:
            return float(np.abs(np.sum(self.spins)) / (self.size**2))
        return 0.0

    def _get_energy(self) -> float:
        """Calculate energy per spin of the lattice."""
        if self.spins is not None:
            return float(
                ising_energy_numba(spins=self.spins, J=self.J, idx_next=self.idx_next)
            )
        return 0.0

    def _get_structure_factor_squared_unshifted(self) -> np.ndarray:
        """Calculate the unshifted squared magnitude of the Fourier transform."""
        if self.spins is not None:
            Sk = np.fft.fft2(self.spins)
            return np.abs(Sk) ** 2
        return np.array([])


if __name__ == '__main__':
    import argparse
    import logging

    from utils.system_helpers import setup_logging

    parser = argparse.ArgumentParser(description='Ising Model Quick Example')
    parser.add_argument('--size', type=int, default=128, help='Lattice size L')
    parser.add_argument('--temp', type=float, default=2.269, help='Temperature T')
    parser.add_argument('--steps', type=int, default=500, help='MC steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)

    logger.info(f'Initializing Ising Model (L={args.size}, T={args.temp})...')
    sim = IsingSimulation(size=args.size, temp=args.temp, seed=args.seed)

    logger.info(f'Running for {args.steps} steps...')
    mag_history, energy_history = sim.run(n_steps=args.steps)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'2D Ising Model — $L={args.size}, T={args.temp}$', fontsize=14)

    # Final Configuration
    if sim.spins is not None:
        ax1.imshow(sim.spins, cmap='gray', interpolation='none')
    ax1.set_title('Final Spin Configuration')
    ax1.axis('off')

    # Magnetization and Energy
    ax2.plot(mag_history, label='|M|')
    ax2.plot(energy_history, label='Energy')
    ax2.set_title('Thermodynamics vs Time')
    ax2.set_xlabel('Monte Carlo Steps')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Correlation
    r, G_r = sim._calculate_correlation_function()
    ax3.plot(r, G_r, 'o-', markersize=3)
    ax3.set_title('Spin-Spin Correlation G(r)')
    ax3.set_xlabel('Distance r')
    ax3.set_ylabel('G(r)')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    plt.tight_layout()
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'ising_example.png')
    plt.savefig(output_file)
    logger.info(f'Simulation finished. Plot saved to {output_file}')
