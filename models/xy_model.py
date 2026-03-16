"""
2D XY Model simulation using Metropolis-Hastings and Wolff cluster algorithms.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

from .simulation_base import (
    MonteCarloSimulation,
    calculate_vortex_density_numba,
    calculate_vorticity_numba,
    get_helicity_data_numba,
)


@njit(cache=True, fastmath=True)
def xy_step_numba(
    *, spins: np.ndarray, beta: float, J: float, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the XY lattice.
    Uses a checkerboard update pattern for better Numba optimization.

    Parameters
    ----------
        spins: (N, N, 2) array of unit vectors.
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

    Returns
    -------
        Updated spins array.
    """
    N = spins.shape[0]
    # Batch generate random updates for the whole lattice
    deltas = np.random.uniform(-0.5, 0.5, size=(N, N))
    cos_vals = np.cos(deltas)
    sin_vals = np.sin(deltas)
    random_vals = np.random.random(size=(N, N))

    for parity in range(2):
        for i in range(N):
            # Use striding to avoid 'if' condition in the inner loop
            start_j = (parity + i) % 2
            inxt = idx_next[i]
            iprv = idx_prev[i]
            for j in range(start_j, N, 2):
                jnxt = idx_next[j]
                jprv = idx_prev[j]

                # Neighbor sum vector
                nx = spins[iprv, j, 0] + spins[inxt, j, 0] + spins[i, jprv, 0] + spins[i, jnxt, 0]
                ny = spins[iprv, j, 1] + spins[inxt, j, 1] + spins[i, jprv, 1] + spins[i, jnxt, 1]

                # Current spin
                sx = spins[i, j, 0]
                sy = spins[i, j, 1]

                # Proposed update from pre-calculated values
                c, s = cos_vals[i, j], sin_vals[i, j]

                sx_new = sx * c - sy * s
                sy_new = sx * s + sy * c

                # Re-normalize to prevent drift
                norm = np.sqrt(sx_new**2 + sy_new**2)
                sx_new /= norm
                sy_new /= norm

                # Energy change: -J * (s_new - s_old) . neighbors
                dE = -J * ((sx_new * nx + sy_new * ny) - (sx * nx + ny * sy))

                if dE <= 0 or random_vals[i, j] < np.exp(-dE * beta):
                    spins[i, j, 0] = sx_new
                    spins[i, j, 1] = sy_new
    return spins


@njit(cache=True, fastmath=True, parallel=True)
def xy_step_parallel_numba(
    *, spins: np.ndarray, beta: float, J: float, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
    """
    Parallel version of xy_step_numba.

    Parameters
    ----------
        spins: (N, N, 2) array of unit vectors.
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

    Returns
    -------
        Updated spins array.
    """
    N = spins.shape[0]
    deltas = np.random.uniform(-0.5, 0.5, size=(N, N))
    cos_vals = np.cos(deltas)
    sin_vals = np.sin(deltas)
    random_vals = np.random.random(size=(N, N))

    for parity in range(2):
        for i in prange(N):
            start_j = (parity + i) % 2
            inxt = idx_next[i]
            iprv = idx_prev[i]
            for j in range(start_j, N, 2):
                jnxt = idx_next[j]
                jprv = idx_prev[j]

                nx = spins[iprv, j, 0] + spins[inxt, j, 0] + spins[i, jprv, 0] + spins[i, jnxt, 0]
                ny = spins[iprv, j, 1] + spins[inxt, j, 1] + spins[i, jprv, 1] + spins[i, jnxt, 1]

                sx, sy = spins[i, j, 0], spins[i, j, 1]
                c, s = cos_vals[i, j], sin_vals[i, j]

                sx_new = sx * c - sy * s
                sy_new = sx * s + sy * c
                norm = np.sqrt(sx_new**2 + sy_new**2)
                sx_new /= norm
                sy_new /= norm

                dE = -J * ((sx_new * nx + sy_new * ny) - (sx * nx + sy * ny))

                if dE <= 0 or random_vals[i, j] < np.exp(-dE * beta):
                    spins[i, j, 0] = sx_new
                    spins[i, j, 1] = sy_new
    return spins


@njit(cache=True, fastmath=True)
def xy_step_random_numba(
    *, spins: np.ndarray, beta: float, J: float, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
    """
    Perform one full Monte Carlo sweep of the XY lattice using random sequential updates.

    Parameters
    ----------
        spins: (N, N, 2) array of unit vectors.
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices.
        idx_prev: Pre-calculated previous-neighbor indices.

    Returns
    -------
        Updated spins array.
    """
    N = spins.shape[0]
    num_attempts = N * N
    # Pre-calculate random indices and updates
    indices = np.random.randint(0, num_attempts, size=num_attempts)
    deltas = np.random.uniform(-0.5, 0.5, size=num_attempts)
    cos_vals = np.cos(deltas)
    sin_vals = np.sin(deltas)
    random_vals = np.random.random(size=num_attempts)

    for k in range(num_attempts):
        idx = indices[k]
        i = idx // N
        j = idx % N

        inxt = idx_next[i]
        iprv = idx_prev[i]
        jnxt = idx_next[j]
        jprv = idx_prev[j]

        # Neighbor sum vector
        nx = spins[iprv, j, 0] + spins[inxt, j, 0] + spins[i, jprv, 0] + spins[i, jnxt, 0]
        ny = spins[iprv, j, 1] + spins[inxt, j, 1] + spins[i, jprv, 1] + spins[i, jnxt, 1]

        # Current spin
        sx = spins[i, j, 0]
        sy = spins[i, j, 1]

        # Proposed update
        c, s = cos_vals[k], sin_vals[k]

        sx_new = sx * c - sy * s
        sy_new = sx * s + sy * c

        # Re-normalize
        norm = np.sqrt(sx_new**2 + sy_new**2)
        sx_new /= norm
        sy_new /= norm

        # Energy change
        dE = -J * ((sx_new * nx + sy_new * ny) - (sx * nx + sy * ny))

        if dE <= 0 or random_vals[k] < np.exp(-dE * beta):
            spins[i, j, 0] = sx_new
            spins[i, j, 1] = sy_new
    return spins


@njit(cache=True, fastmath=True)
def xy_energy_numba(*, spins: np.ndarray, J: float, idx_next: np.ndarray) -> float:
    """
    Calculate the total energy of the XY lattice.

    Parameters
    ----------
        spins: (N, N, 2) array of unit vectors.
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
            # Sum unique pairs (right and down)
            dot_right = spins[i, j, 0] * spins[i, jnxt, 0] + spins[i, j, 1] * spins[i, jnxt, 1]
            dot_down = spins[i, j, 0] * spins[inxt, j, 0] + spins[i, j, 1] * spins[inxt, j, 1]
            energy -= J * (dot_right + dot_down)
    return energy / (N * N)


@njit(cache=True, fastmath=True)
def xy_wolff_step_numba(
    *, spins: np.ndarray, beta: float, J: float, idx_next: np.ndarray, idx_prev: np.ndarray
) -> np.ndarray:
    """
    Perform one Wolff cluster flip on the XY lattice (Wolff-Evertz reflection).

    A random mirror-plane axis r\u0302 is sampled uniformly from S^1.  Each spin
    projects onto r\u0302 as sigma_i = s_i \u00b7 r\u0302.  Bonds between neighbouring sites
    with aligned projections (sigma_i * sigma_j > 0) are activated with
    probability ``P_add = 1 - exp(-2 beta J sigma_i sigma_j)``.  All cluster
    spins are then reflected through the plane perpendicular to r\u0302:
    ``s -> s - 2 (s \u00b7 r\u0302) r\u0302``, which preserves unit length exactly and
    satisfies detailed balance for the pure exchange term.

    One call constitutes one cluster sweep.  ``parallel=True`` is silently
    ignored.

    Parameters
    ----------
        spins: (N, N, 2) array of unit vectors.
        beta: Inverse temperature 1/kT.
        J: Coupling constant.
        idx_next: Pre-calculated next-neighbor indices (PBC).
        idx_prev: Pre-calculated previous-neighbor indices (PBC).

    Returns
    -------
        Updated spins array.
    """
    N = spins.shape[0]

    # Random mirror-plane axis r\u0302 = (cos phi, sin phi)
    phi = np.random.uniform(0.0, 2.0 * np.pi)
    rx = np.cos(phi)
    ry = np.sin(phi)

    # Random seed site
    si = np.random.randint(0, N)
    sj = np.random.randint(0, N)

    # Pre-allocate cluster membership mask and DFS stack
    in_cluster = np.zeros((N, N), dtype=np.bool_)
    stack = np.empty(N * N, dtype=np.int64)
    in_cluster[si, sj] = True
    stack[0] = si * N + sj
    stack_top = 1

    while stack_top > 0:
        stack_top -= 1
        flat = stack[stack_top]
        ci = flat // N
        cj = flat % N

        proj_c = spins[ci, cj, 0] * rx + spins[ci, cj, 1] * ry
        inxt = idx_next[ci]
        iprv = idx_prev[ci]
        jnxt = idx_next[cj]
        jprv = idx_prev[cj]

        # North
        if not in_cluster[iprv, cj]:
            proj_n = spins[iprv, cj, 0] * rx + spins[iprv, cj, 1] * ry
            prod = proj_c * proj_n
            if prod > 0.0:
                p_add = 1.0 - np.exp(-2.0 * beta * J * prod)
                if np.random.random() < p_add:
                    in_cluster[iprv, cj] = True
                    stack[stack_top] = iprv * N + cj
                    stack_top += 1
        # South
        if not in_cluster[inxt, cj]:
            proj_n = spins[inxt, cj, 0] * rx + spins[inxt, cj, 1] * ry
            prod = proj_c * proj_n
            if prod > 0.0:
                p_add = 1.0 - np.exp(-2.0 * beta * J * prod)
                if np.random.random() < p_add:
                    in_cluster[inxt, cj] = True
                    stack[stack_top] = inxt * N + cj
                    stack_top += 1
        # West
        if not in_cluster[ci, jprv]:
            proj_n = spins[ci, jprv, 0] * rx + spins[ci, jprv, 1] * ry
            prod = proj_c * proj_n
            if prod > 0.0:
                p_add = 1.0 - np.exp(-2.0 * beta * J * prod)
                if np.random.random() < p_add:
                    in_cluster[ci, jprv] = True
                    stack[stack_top] = ci * N + jprv
                    stack_top += 1
        # East
        if not in_cluster[ci, jnxt]:
            proj_n = spins[ci, jnxt, 0] * rx + spins[ci, jnxt, 1] * ry
            prod = proj_c * proj_n
            if prod > 0.0:
                p_add = 1.0 - np.exp(-2.0 * beta * J * prod)
                if np.random.random() < p_add:
                    in_cluster[ci, jnxt] = True
                    stack[stack_top] = ci * N + jnxt
                    stack_top += 1

    # Reflect all cluster spins through the plane perpendicular to r\u0302
    for i in range(N):
        for j in range(N):
            if in_cluster[i, j]:
                proj = spins[i, j, 0] * rx + spins[i, j, 1] * ry
                spins[i, j, 0] -= 2.0 * proj * rx
                spins[i, j, 1] -= 2.0 * proj * ry

    return spins


class XYSimulation(MonteCarloSimulation):
    """
    Simulation of the 2D XY model on a square lattice.
    """

    _VALID_UPDATES: frozenset = frozenset({'checkerboard', 'random', 'wolff'})

    def __init__(
        self,
        *,
        size: int,
        temp: float,
        J: float = 1.0,
        update: str = 'checkerboard',
        parallel: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize the XY simulation.

        Parameters
        ----------
            size: Linear dimension L of the L x L lattice.
            temp: Temperature T.
            J: Coupling constant (default 1.0).
            update: Update scheme - ``'checkerboard'`` (default, faster),
                ``'random'`` (random sequential Metropolis, physical
                stochastic dynamics for kinetics studies), or
                ``'wolff'`` (Wolff-Evertz cluster algorithm, efficient near
                the BKT transition).
            parallel: Whether to use parallelized Numba kernels (only for checkerboard).
            seed: Optional random seed for reproducibility.

        Raises
        ------
            ValueError: If ``update`` is not one of the recognised schemes.
        """
        super().__init__(size=size, temp=temp, seed=seed)
        if update not in self._VALID_UPDATES:
            valid_opts = sorted(self._VALID_UPDATES)
            raise ValueError(f'Unknown update scheme {update!r}. Valid options: {valid_opts}')
        self.J = J
        self.update = update
        self.parallel = parallel

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
                self.spins = xy_step_random_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
            elif self.update == 'wolff':
                self.spins = xy_wolff_step_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
            elif self.parallel:
                self.spins = xy_step_parallel_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
            else:
                self.spins = xy_step_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
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
            return float(xy_energy_numba(spins=self.spins, J=self.J, idx_next=self.idx_next))
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
            cos_sum, sin_sum = get_helicity_data_numba(spins=self.spins, idx_next=self.idx_next)
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


def main() -> None:
    """CLI entry point for XY simulation example."""
    import argparse
    import logging

    from utils.system_helpers import setup_logging

    parser = argparse.ArgumentParser(description='XY Model Quick Example')
    parser.add_argument('--size', type=int, default=128, help='Lattice size L')
    parser.add_argument('--temp', type=float, default=0.5, help='Temperature T')
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
        f'Initializing XY Model (L={args.size}, T={args.temp}, '
        f'update={args.update}, parallel={args.parallel})...'
    )
    sim = XYSimulation(
        size=args.size, temp=args.temp, seed=args.seed, parallel=args.parallel, update=args.update
    )

    logger.info(f'Running for {args.steps} steps...')
    sim.run(n_steps=args.steps)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    fig.suptitle(f'2D XY Model - $L={args.size}, T={args.temp}$', fontsize=14)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Final Configuration (Phase angle)
    if sim.spins is not None:
        angles = np.arctan2(sim.spins[..., 1], sim.spins[..., 0])
        im1 = ax1.imshow(angles, cmap='hsv', interpolation='none', vmin=-np.pi, vmax=np.pi)
        ax1.set_title('Final Spin Phase')
        ax1.axis('off')
        fig.colorbar(im1, ax=ax1, label='Phase (rad)', shrink=0.8)

    # Vorticity map
    vort = sim._calculate_vorticity()
    im2 = ax2.imshow(vort, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
    ax2.set_title(f'Vorticity (Total: {int(np.sum(np.abs(vort)))})')
    ax2.axis('off')
    fig.colorbar(im2, ax=ax2, ticks=[-1, 0, 1], label='Winding No.', shrink=0.8)

    # Spin-Spin Correlation Function
    r, G_r = sim._calculate_correlation_function()
    ax3.plot(r[1:], G_r[1:], 'o-', markersize=3)
    ax3.set_title('Spin-Spin Correlation G(r)')
    ax3.set_xlabel('Distance r')
    ax3.set_ylabel('G(r)')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')

    # Structure Factor (Radial)
    from utils.physics_helpers import radial_average_sk

    if sim.spins is not None:
        k, sk = radial_average_sk(spins=sim.spins)
        ax4.loglog(k[1:], sk[1:], 'o-', markersize=3, color='tab:green')
        ax4.set_title('Structure Factor S(k)')
        ax4.set_xlabel('|k|')
        ax4.set_ylabel('S(k)')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'xy_example.png')
    plt.savefig(output_file)
    logger.info(f'Simulation finished. Plot saved to {output_file}')


if __name__ == '__main__':
    main()
