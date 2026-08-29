"""
Base classes and shared Numba-accelerated kernels for Monte Carlo simulations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _seed_numba(*, seed: int) -> None:
    """Helper to seed Numba's internal random number generator."""
    np.random.seed(seed)


_MASK64 = (1 << 64) - 1


def _derive_step_seed(*, seed: int, step: int) -> int:
    """Mix a (seed, step) pair into a decorrelated 32-bit per-sweep seed.

    A plain ``seed + step`` derivation makes the per-sweep random stream of
    seed ``s`` at sweep ``t + 1`` identical to that of seed ``s + 1`` at sweep
    ``t``, so replicas seeded with consecutive integers (the common
    ``base_seed + k`` convention) share almost all of their randomness and are
    not statistically independent. The SplitMix64 finalizer scrambles the pair
    so distinct (seed, step) inputs yield uncorrelated seeds.

    Parameters
    ----------
        seed: Simulation base seed (non-negative integer).
        step: Zero-based sweep index.

    Returns
    -------
        A seed in [0, 2**32) suitable for ``np.random.seed`` inside Numba.
    """
    z = (seed * 0xD1B54A32D192ED03 + step * 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    z ^= z >> 31
    return int(z & 0xFFFFFFFF)


@njit(cache=True, fastmath=True)
def _calculate_vorticity_angles_numba(angles: np.ndarray, idx_next: np.ndarray) -> np.ndarray:
    """Fast kernel to calculate vorticity from a 2D array of angles."""
    N = angles.shape[0]
    vorticity = np.zeros((N, N))
    for i in range(N):
        inxt = idx_next[i]
        for j in range(N):
            jnxt = idx_next[j]

            t1 = angles[i, j]
            t2 = angles[i, jnxt]
            t3 = angles[inxt, jnxt]
            t4 = angles[inxt, j]

            d1 = (t2 - t1 + np.pi) % (2 * np.pi) - np.pi
            d2 = (t3 - t2 + np.pi) % (2 * np.pi) - np.pi
            d3 = (t4 - t3 + np.pi) % (2 * np.pi) - np.pi
            d4 = (t1 - t4 + np.pi) % (2 * np.pi) - np.pi

            vorticity[i, j] = np.round((d1 + d2 + d3 + d4) / (2 * np.pi))
    return vorticity


def calculate_vorticity_numba(*, spins: np.ndarray, idx_next: np.ndarray) -> np.ndarray:
    """
    Calculate the vorticity (winding number) of each plaquette for 2D vector spins.
    Optimized by calculating angles first and then using a fast wrapping kernel.

    Parameters
    ----------
        spins: (N, N, 2) array of unit vectors.
        idx_next: Pre-calculated next-neighbor indices.

    Returns
    -------
        vorticity: (N, N) array containing winding numbers (+1, -1, or 0).
    """
    # Vectorized arctan2 is much faster than calling it inside a Numba loop
    angles = np.arctan2(spins[..., 1], spins[..., 0])
    return _calculate_vorticity_angles_numba(angles, idx_next)


@njit(cache=True, fastmath=True)
def _vortex_density_from_vorticity_numba(vorticity: np.ndarray) -> float:
    """Return density of non-zero winding plaquettes."""
    N = vorticity.shape[0]
    count = 0
    for i in range(N):
        for j in range(N):
            if np.abs(vorticity[i, j]) > 0.0:
                count += 1
    return count / float(N * N)


def calculate_vortex_density_numba(*, spins: np.ndarray, idx_next: np.ndarray) -> float:
    """
    Calculate vortex density n_v, i.e. the fraction of plaquettes with non-zero winding.

    Parameters
    ----------
        spins: (N, N, 2) array of unit vectors.
        idx_next: Pre-calculated next-neighbor indices.

    Returns
    -------
        Vortex density n_v in [0, 1].
    """
    vorticity = calculate_vorticity_numba(spins=spins, idx_next=idx_next)
    return float(_vortex_density_from_vorticity_numba(vorticity))


@njit(cache=True, fastmath=True)
def get_helicity_data_numba(*, spins: np.ndarray, idx_next: np.ndarray) -> tuple[float, float]:
    """
    Calculate sum of cos and sin of angle differences in x-direction for helicity modulus.

    Parameters
    ----------
        spins: (N, N, 2) array of unit vectors.
        idx_next: Pre-calculated next-neighbor indices for PBCs.

    Returns
    -------
        cos_sum: Sum of cosine of angle differences.
        sin_sum: Sum of sine of angle differences.
    """
    N = spins.shape[0]
    cos_sum = 0.0
    sin_sum = 0.0
    for i in range(N):
        for j in range(N):
            j_next = idx_next[j]
            # cos(theta_i - theta_j) = s_i . s_j
            cos_sum += spins[i, j, 0] * spins[i, j_next, 0] + spins[i, j, 1] * spins[i, j_next, 1]
            # sin(theta_i - theta_j) = cross product
            sin_sum += spins[i, j, 0] * spins[i, j_next, 1] - spins[i, j, 1] * spins[i, j_next, 0]
    return cos_sum, sin_sum


class MonteCarloSimulation(ABC):
    """
    Abstract base class for 2D lattice Monte Carlo simulations.
    Provides infrastructure for equilibration, measurement runs, and statistical analysis.
    """

    def __init__(
        self, *, size: int, temp: float, init_state: str = 'random', seed: int | None = None
    ):
        """
        Initialize the simulation.

        Parameters
        ----------
            size: Linear dimension L of the N x N lattice.
            temp: Temperature T of the system.
            init_state: Initial spin configuration: ``'random'`` (default) or ``'ordered'``.
            seed: Optional random seed for reproducibility.

        Raises
        ------
            ValueError: If ``size`` is not a positive integer or ``temp`` is not positive.
        """
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError(f'size must be a positive integer, got {size!r}')
        if temp <= 0.0:
            raise ValueError(f'temp must be positive (T > 0), got {temp}')
        if init_state not in {'random', 'ordered'}:
            raise ValueError(f"init_state must be 'random' or 'ordered', got {init_state!r}")
        self.size = size
        self.temp = temp
        self.init_state = init_state
        self.beta = 1.0 / temp
        self.steps = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.spins: np.ndarray | None = None  # To be initialized by subclasses
        self.idx_next = np.roll(np.arange(size), -1)
        self.idx_prev = np.roll(np.arange(size), 1)

        # Pre-allocated buffers for Wolff cluster algorithms
        self._wolff_cluster_mask = np.zeros((size, size), dtype=np.bool_)
        self._wolff_stack = np.empty(size * size, dtype=np.int64)
        self._wolff_cluster_spins = np.empty(size * size, dtype=np.int64)
        # Last Wolff cluster size (0 for non-Wolff updates or before any step)
        self.last_cluster_size: int = 0

        # Pre-calculate radial distance bins for correlation analysis
        center = size // 2
        iy, ix = np.indices((size, size))
        r = np.sqrt((ix - center) ** 2 + (iy - center) ** 2)
        self._r_int_pre = r.astype(int).ravel()
        self._nr_pre = np.bincount(self._r_int_pre)
        self._r_range_pre = np.arange(center)

    def _reseed_numba_for_step(self) -> None:
        """Reseed Numba's RNG deterministically for the upcoming sweep.

        No-op when the simulation is unseeded. Serial kernels become fully
        reproducible; parallel (``prange``) kernels are NOT seed-reproducible
        because only the calling thread's Numba RNG state is seeded.
        """
        if self.seed is not None:
            _seed_numba(seed=_derive_step_seed(seed=self.seed, step=self.steps))

    @abstractmethod
    def step(self) -> None:
        """Perform one full Monte Carlo sweep of the lattice."""

    @abstractmethod
    def _get_magnetization(self) -> float:
        """Return the current absolute magnetization per site."""

    @abstractmethod
    def _get_energy(self) -> float:
        """Return the current energy per site."""

    @abstractmethod
    def _get_structure_factor_squared_unshifted(self) -> np.ndarray:
        """Return the unshifted squared magnitude of the Fourier transform of spins."""

    def _calculate_structure_factor(self) -> np.ndarray:
        """Calculate the 2D structure factor S(k)."""
        Sk_sq = self._get_structure_factor_squared_unshifted()
        return np.fft.fftshift(Sk_sq) / (self.size**2)

    def _calculate_correlation_function(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the real-space spin-spin correlation function G(r)
        by taking the inverse Fourier transform of the structure factor S(k).

        Returns
        -------
            r: Radial distances.
            G_r: Radially averaged correlation values.
        """
        Sk_sq = self._get_structure_factor_squared_unshifted()

        # G(r) is the inverse Fourier transform of S(k) (Wiener-Khinchin theorem)
        G_r_map = np.real(np.fft.ifft2(Sk_sq))

        # Shift the (0,0) correlation to the center of the map for radial averaging
        G_r_map = np.fft.fftshift(G_r_map)

        # Radially average G(r) using pre-calculated masks
        tbin = np.bincount(self._r_int_pre, G_r_map.ravel())

        # Avoid division by zero
        radial_profile = np.divide(
            tbin, self._nr_pre, out=np.zeros_like(tbin, dtype=float), where=self._nr_pre != 0
        )

        # Normalize by G(0) so that G(0) = 1
        if radial_profile[0] != 0:
            radial_profile /= radial_profile[0]

        center = self.size // 2
        return self._r_range_pre, radial_profile[:center]

    def equilibrate(self, *, n_steps: int) -> None:
        """
        Perform equilibration steps without recording measurements.

        Parameters
        ----------
            n_steps: Number of MC steps to perform.
        """
        for _ in range(n_steps):
            self.step()

    def run(self, *, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the simulation and record magnetization and energy at each step.

        Parameters
        ----------
            n_steps: Number of MC steps to perform and record.

        Returns
        -------
            magnetization: Array of recorded magnetization values.
            energies: Array of recorded energy values.
        """
        magnetization: np.ndarray = np.empty(n_steps)
        energies: np.ndarray = np.empty(n_steps)
        for i in range(n_steps):
            self.step()
            magnetization[i] = self._get_magnetization()
            energies[i] = self._get_energy()
        return magnetization, energies
