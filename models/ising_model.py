"""
2D Ising Model simulation using Metropolis-Hastings and Wolff cluster algorithms.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

from .simulation_base import MonteCarloSimulation


@njit(cache=True, fastmath=True)
def ising_step_numba(
    *, spins: np.ndarray, beta: float, J: float, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the Ising lattice.
    Uses a checkerboard update pattern for better Numba optimization.

    Parameters
    ----------
        spins: (N, N) array of spins (+1 or -1).
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

    Returns
    -------
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


@njit(cache=True, fastmath=True, parallel=True)
def ising_step_parallel_numba(
    *, spins: np.ndarray, beta: float, J: float, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
    """
    Parallel version of ising_step_numba using numba.prange.

    Parameters
    ----------
        spins: (N, N) array of spins (+1 or -1).
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

    Returns
    -------
        Updated spins array.
    """
    N = spins.shape[0]
    prob4 = np.exp(-4.0 * J * beta)
    prob8 = np.exp(-8.0 * J * beta)

    for parity in range(2):
        for i in prange(N):
            start_j = (parity + i) % 2
            for j in range(start_j, N, 2):
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


@njit(cache=True, fastmath=True)
def ising_energy_numba(*, spins: np.ndarray, J: float, idx_next: np.ndarray) -> float:
    """
    Calculate the total energy of the Ising lattice.

    Parameters
    ----------
        spins: (N, N) array of spins.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices.

    Returns
    -------
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

    Parameters
    ----------
        spins: (N, N) array of spins (+1 or -1).
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

    Returns
    -------
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


@njit(cache=True, fastmath=True)
def ising_wolff_step_numba(
    *,
    spins: np.ndarray,
    beta: float,
    J: float,
    idx_next: np.ndarray,
    idx_prev: np.ndarray,
    in_cluster: np.ndarray,
    stack: np.ndarray,
    cluster_spins: np.ndarray
) -> tuple:
    """
    Perform one Wolff cluster flip on the Ising lattice.

    A single cluster is grown by probabilistic DFS from a random seed site and
    then flipped in a single collective move.  The bond acceptance probability
    ``P_add = 1 - exp(-2 beta J)`` is derived from the Fortuin-Kasteleyn
    representation and guarantees detailed balance without rejections.

    One call constitutes one *cluster sweep*.  The effective number of site
    updates per call scales as the mean cluster size ``<C>`` and diverges at
    T_c, which is precisely where Metropolis suffers maximum critical slowing
    down.  ``parallel=True`` is silently ignored: cluster growth is
    inherently sequential.

    Parameters
    ----------
        spins: (N, N) array of spins (+1 or -1).
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices (PBC).
        idx_prev: Pre-calculated previous-neighbor indices (PBC).
        in_cluster: Pre-allocated (N, N) boolean array for cluster membership mask.
        stack: Pre-allocated (N*N) int64 array for DFS stack.
        cluster_spins: Pre-allocated (N*N) int64 array to track cluster elements.

    Returns
    -------
        spins: Updated spins array.
        cluster_size: Number of spins flipped in this step.
    """
    N = spins.shape[0]
    p_add = 1.0 - np.exp(-2.0 * J * beta)

    # Choose a random seed site
    si = np.random.randint(0, N)
    sj = np.random.randint(0, N)
    seed_spin = spins[si, sj]

    # Setup DFS from seed
    # Notice: in_cluster is assumed to be fully False upon entry!
    seed_flat = si * N + sj
    in_cluster[si, sj] = True
    stack[0] = seed_flat
    stack_top = 1

    cluster_spins[0] = seed_flat
    cluster_size = 1  # seed site already in cluster

    while stack_top > 0:
        stack_top -= 1
        flat = stack[stack_top]
        ci = flat // N
        cj = flat % N

        inxt = idx_next[ci]
        iprv = idx_prev[ci]
        jnxt = idx_next[cj]
        jprv = idx_prev[cj]

        # North
        if not in_cluster[iprv, cj] and spins[iprv, cj] == seed_spin:
            if np.random.random() < p_add:
                in_cluster[iprv, cj] = True
                new_idx = iprv * N + cj
                stack[stack_top] = new_idx
                stack_top += 1
                cluster_spins[cluster_size] = new_idx
                cluster_size += 1
        # South
        if not in_cluster[inxt, cj] and spins[inxt, cj] == seed_spin:
            if np.random.random() < p_add:
                in_cluster[inxt, cj] = True
                new_idx = inxt * N + cj
                stack[stack_top] = new_idx
                stack_top += 1
                cluster_spins[cluster_size] = new_idx
                cluster_size += 1
        # West
        if not in_cluster[ci, jprv] and spins[ci, jprv] == seed_spin:
            if np.random.random() < p_add:
                in_cluster[ci, jprv] = True
                new_idx = ci * N + jprv
                stack[stack_top] = new_idx
                stack_top += 1
                cluster_spins[cluster_size] = new_idx
                cluster_size += 1
        # East
        if not in_cluster[ci, jnxt] and spins[ci, jnxt] == seed_spin:
            if np.random.random() < p_add:
                in_cluster[ci, jnxt] = True
                new_idx = ci * N + jnxt
                stack[stack_top] = new_idx
                stack_top += 1
                cluster_spins[cluster_size] = new_idx
                cluster_size += 1

    # Flip all cluster spins in-place and reset in_cluster mask to False cleanly
    for i in range(cluster_size):
        flat = cluster_spins[i]
        ci = flat // N
        cj = flat % N
        spins[ci, cj] *= -1
        in_cluster[ci, cj] = False

    return spins, cluster_size


class IsingSimulation(MonteCarloSimulation):
    """
    Simulation of the 2D Ising model on a square lattice.
    """

    _VALID_UPDATES: frozenset = frozenset({'checkerboard', 'random', 'wolff'})

    def __init__(
        self,
        *,
        size: int,
        temp: float,
        J: float = 1.0,
        update: str = 'checkerboard',
        init_state: str = 'random',
        parallel: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize the Ising simulation.

        Parameters
        ----------
            size: Linear dimension L of the L x L lattice.
            temp: Temperature T.
            J: Coupling constant (default 1.0).
            update: Update scheme - ``'checkerboard'`` (default, faster),
                ``'random'`` (random sequential Metropolis, physical
                stochastic dynamics for coarsening studies), or
                ``'wolff'`` (Wolff cluster algorithm, highly efficient near T_c
                due to vanishing critical slowing down).
            init_state: Initial spin configuration: ``'random'`` (default) or ``'ordered'``.
            parallel: Whether to use parallelized Numba kernels (only for checkerboard).
            seed: Optional random seed for reproducibility.

        Raises
        ------
            ValueError: If ``update`` is not one of the recognised schemes.
        """
        super().__init__(size=size, temp=temp, init_state=init_state, seed=seed)
        if update not in self._VALID_UPDATES:
            valid_opts = sorted(self._VALID_UPDATES)
            raise ValueError(f'Unknown update scheme {update!r}. Valid options: {valid_opts}')
        self.J = J
        self.update = update
        self.parallel = parallel

        if self.init_state == 'ordered':
            self.spins = np.ones((size, size), dtype=np.int8)
        else:
            self.spins = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(size, size))

        # Last Wolff cluster size (0 for non-Wolff updates or before any step)
        self.last_cluster_size: int = 0

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
            elif self.update == 'wolff':
                self.spins, self.last_cluster_size = ising_wolff_step_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                    in_cluster=self._wolff_cluster_mask,
                    stack=self._wolff_stack,
                    cluster_spins=self._wolff_cluster_spins,
                )
            elif self.parallel:
                self.spins = ising_step_parallel_numba(
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

    def run_with_cluster_sizes(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the simulation and additionally record the Wolff cluster size at every step.

        For non-Wolff update schemes the cluster-size array is filled with zeros because
        local Metropolis flips exactly one spin at a time (cluster size = 1 by convention
        would also be valid, but zero makes it easy to detect misuse).

        Parameters
        ----------
        n_steps:
            Number of MC steps to perform and record.

        Returns
        -------
        magnetization:
            Array of `|M|` per spin at each step.
        energies:
            Array of energy per spin at each step.
        cluster_sizes:
            Array of cluster sizes (number of spins flipped) at each step.
            Always zero for non-Wolff algorithms.
        """
        magnetization = np.empty(n_steps, dtype=float)
        energies = np.empty(n_steps, dtype=float)
        cluster_sizes = np.zeros(n_steps, dtype=int)
        for i in range(n_steps):
            self.step()
            magnetization[i] = self._get_magnetization()
            energies[i] = self._get_energy()
            cluster_sizes[i] = self.last_cluster_size
        return magnetization, energies, cluster_sizes

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


def main() -> None:
    """CLI entry point for Ising simulation example."""
    import argparse
    import logging

    from utils.system import setup_logging

    parser = argparse.ArgumentParser(description='Ising Model Quick Example')
    parser.add_argument('--size', type=int, default=128, help='Lattice size L')
    parser.add_argument('--temp', type=float, default=2.269, help='Temperature T')
    parser.add_argument('--steps', type=int, default=500, help='MC steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--parallel', action='store_true', help='Use parallel kernels')
    parser.add_argument(
        '--update',
        type=str,
        default='checkerboard',
        choices=['checkerboard', 'random', 'wolff'],
        help='Update scheme (default: checkerboard)',
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level)

    logger.info(
        f'Initializing Ising Model (L={args.size}, T={args.temp}, '
        f'update={args.update}, parallel={args.parallel})...'
    )
    sim = IsingSimulation(
        size=args.size, temp=args.temp, seed=args.seed, parallel=args.parallel, update=args.update
    )

    logger.info(f'Running for {args.steps} steps...')
    mag_history, energy_history = sim.run(n_steps=args.steps)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'2D Ising Model - $L={args.size}, T={args.temp}$', fontsize=14)

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


if __name__ == '__main__':
    main()
