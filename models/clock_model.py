"""
2D q-state Clock Model simulation using the Metropolis-Hastings algorithm.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

from .simulation_base import (
    MonteCarloSimulation,
    calculate_vortex_density_numba,
    calculate_vorticity_numba,
    get_helicity_data_numba,
)


@njit(cache=True, fastmath=True)
def clock_step_numba(
    *,
    spins: np.ndarray,
    beta: float,
    J: float,
    A: float,
    q: int,
    idx_next: np.ndarray,
    idx_prev: np.ndarray,
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
                nx = spins[iprv, j, 0] + spins[inxt, j, 0] + spins[i, jprv, 0] + spins[i, jnxt, 0]
                ny = spins[iprv, j, 1] + spins[inxt, j, 1] + spins[i, jprv, 1] + spins[i, jnxt, 1]

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
                phi_new = phi_old + delta
                dE_aniso = -A * (np.cos(q * phi_new) - np.cos(q * phi_old))

                dE = dE_inter + dE_aniso

                if dE <= 0 or np.random.random() < np.exp(-dE * beta):
                    spins[i, j, 0] = sx_new
                    spins[i, j, 1] = sy_new
    return spins


@njit(cache=True, fastmath=True)
def clock_step_random_numba(
    *,
    spins: np.ndarray,
    beta: float,
    J: float,
    A: float,
    q: int,
    idx_next: np.ndarray,
    idx_prev: np.ndarray,
) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the Clock Model lattice using random sequential updates.

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
    for _ in range(N * N):
        idx = np.random.randint(0, N * N)
        i = idx // N
        j = idx % N

        inxt = idx_next[i]
        iprv = idx_prev[i]
        jnxt = idx_next[j]
        jprv = idx_prev[j]

        # Neighbor sum
        nx = spins[iprv, j, 0] + spins[inxt, j, 0] + spins[i, jprv, 0] + spins[i, jnxt, 0]
        ny = spins[iprv, j, 1] + spins[inxt, j, 1] + spins[i, jprv, 1] + spins[i, jnxt, 1]

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
        phi_new = phi_old + delta
        dE_aniso = -A * (np.cos(q * phi_new) - np.cos(q * phi_old))

        dE = dE_inter + dE_aniso

        if dE <= 0 or np.random.random() < np.exp(-dE * beta):
            spins[i, j, 0] = sx_new
            spins[i, j, 1] = sy_new
    return spins


@njit(cache=True, fastmath=True)
def clock_energy_numba(
    *, spins: np.ndarray, J: float, A: float, q: int, idx_next: np.ndarray
) -> float:
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
            dot_right = spins[i, j, 0] * spins[i, jnxt, 0] + spins[i, j, 1] * spins[i, jnxt, 1]
            dot_down = spins[i, j, 0] * spins[inxt, j, 0] + spins[i, j, 1] * spins[inxt, j, 1]
            energy -= J * (dot_right + dot_down)

            # Anisotropy
            phi = np.arctan2(spins[i, j, 1], spins[i, j, 0])
            energy -= A * np.cos(q * phi)
    return energy / (N * N)


class ClockSimulation(MonteCarloSimulation):
    """
    Simulation of the 2D q-state clock model on a square lattice.
    """

    _VALID_UPDATES: frozenset = frozenset({'checkerboard', 'random'})

    def __init__(
        self,
        *,
        size: int,
        temp: float,
        J: float = 1.0,
        A: float = 1.0,
        q: int = 6,
        update: str = 'checkerboard',
        seed: int | None = None,
    ):
        """
        Initialize the Clock Model simulation.

        Args:
            size: Linear dimension L of the L x L lattice.
            temp: Temperature T.
            J: Coupling constant (default 1.0).
            A: Anisotropy strength (default 1.0).
            q: Number of clock states (default 6). Must be ≥ 2.
            update: Update scheme — ``'checkerboard'`` (default, faster) or
                ``'random'`` (random sequential Metropolis, more physical
                stochastic dynamics for kinetics studies).
            seed: Optional random seed for reproducibility.

        Raises:
            ValueError: If ``q`` is less than 2 or update scheme is unknown.
        """
        super().__init__(size=size, temp=temp, seed=seed)
        if q < 2:
            raise ValueError(f'q must be >= 2 (number of clock states), got {q}')
        if update not in self._VALID_UPDATES:
            valid_opts = sorted(self._VALID_UPDATES)
            raise ValueError(f'Unknown update scheme {update!r}. Valid options: {valid_opts}')
        self.J = J
        self.A = A
        self.q = q
        self.update = update

        # Initialize random spins as 2D unit vectors
        # spin = (spin_x, spin_y)
        angles = self.rng.uniform(0, 2 * np.pi, size=(size, size))
        self.spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def step(self) -> None:
        """Perform one Monte Carlo step using Numba."""
        if self.spins is not None:
            if self.seed is not None:
                from .simulation_base import _seed_numba

                _seed_numba(seed=self.seed + self.steps)

            if self.update == 'random':
                self.spins = clock_step_random_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    A=self.A,
                    q=self.q,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
            else:
                self.spins = clock_step_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    A=self.A,
                    q=self.q,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
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
                clock_energy_numba(
                    spins=self.spins, J=self.J, A=self.A, q=self.q, idx_next=self.idx_next
                )
            )
        return 0.0


    def _calculate_vorticity(self) -> np.ndarray:
        """Calculate the vorticity (winding number) of each plaquette."""
        if self.spins is not None:
            return np.asarray(calculate_vorticity_numba(spins=self.spins, idx_next=self.idx_next))
        return np.array([])

    def _get_vortex_density(self) -> float:
        """Calculate vortex density n_v, the fraction of plaquettes with non-zero winding."""
        if self.spins is not None:
            return float(calculate_vortex_density_numba(spins=self.spins, idx_next=self.idx_next))
        return 0.0

    def _get_helicity_data(self) -> tuple[float, float]:
        """Calculate sum of cos and sin of angle differences in x-direction."""
        if self.spins is not None:
            cos_sum, sin_sum = get_helicity_data_numba(spins=self.spins)
            return float(cos_sum), float(sin_sum)
        return 0.0, 0.0

    def _get_structure_factor_squared_unshifted(self) -> np.ndarray:
        """Calculate the unshifted squared magnitude of the Fourier transform."""
        if self.spins is not None:
            sx = self.spins[..., 0]
            sy = self.spins[..., 1]
            Sk_x = np.fft.fft2(sx)
            Sk_y = np.fft.fft2(sy)
            return np.asarray(np.abs(Sk_x) ** 2 + np.abs(Sk_y) ** 2)
        return np.array([])


if __name__ == '__main__':
    import argparse
    import logging

    from utils.system_helpers import setup_logging

    parser = argparse.ArgumentParser(description='Clock Model Quick Example')
    parser.add_argument('--size', type=int, default=128, help='Lattice size L')
    parser.add_argument('--temp', type=float, default=0.2, help='Temperature T')
    parser.add_argument('--q', type=int, default=6, help='Clock states q')
    parser.add_argument('--steps', type=int, default=500, help='MC steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)

    logger.info(f'Initializing {args.q}-state Clock Model (L={args.size}, T={args.temp})...')
    sim = ClockSimulation(size=args.size, temp=args.temp, q=args.q, seed=args.seed)

    logger.info(f'Running for {args.steps} steps...')
    mag_history, energy_history = sim.run(n_steps=args.steps)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'2D {args.q}-state Clock Model — $L={args.size}, T={args.temp}$', fontsize=14)

    # Final Phase Configuration
    if sim.spins is not None:
        angles = np.arctan2(sim.spins[..., 1], sim.spins[..., 0])
        im1 = ax1.imshow(angles, cmap='hsv', interpolation='none', vmin=-np.pi, vmax=np.pi)
        ax1.set_title('Final Spin Phase')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Phase (rad)', shrink=0.8)

    # Magnetization and Energy
    ax2.plot(mag_history, label='|M|')
    ax2.plot(energy_history, label='Energy')
    ax2.set_title('Thermodynamics vs Time')
    ax2.set_xlabel('Monte Carlo Steps')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Vorticity map
    vort = sim._calculate_vorticity()
    im2 = ax3.imshow(vort, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
    ax3.set_title(f'Vorticity (Total: {int(np.sum(np.abs(vort)))})')
    ax3.axis('off')
    plt.colorbar(im2, ax=ax3, ticks=[-1, 0, 1], label='Winding No.', shrink=0.8)

    plt.tight_layout()
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'clock_example.png')
    plt.savefig(output_file)
    logger.info(f'Simulation finished. Plot saved to {output_file}')

