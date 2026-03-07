"""
Base classes and shared Numba-accelerated kernels for Monte Carlo simulations.
"""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit


@njit(cache=True)
def _seed_numba(seed: int) -> None:
    """Helper to seed Numba's internal random number generator."""
    np.random.seed(seed)


@njit(cache=True)
def calculate_vorticity_numba(spins: np.ndarray) -> np.ndarray:
    """
    Calculate the vorticity (winding number) of each plaquette for 2D vector spins.

    Args:
        spins: (N, N, 2) array of unit vectors.

    Returns:
        vorticity: (N, N) array containing winding numbers (+1, -1, or 0).
    """
    N = spins.shape[0]
    vorticity = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            i_next = 0 if i == N - 1 else i + 1
            j_next = 0 if j == N - 1 else j + 1

            t1 = np.arctan2(spins[i, j, 1], spins[i, j, 0])
            t2 = np.arctan2(spins[i, j_next, 1], spins[i, j_next, 0])
            t3 = np.arctan2(spins[i_next, j_next, 1], spins[i_next, j_next, 0])
            t4 = np.arctan2(spins[i_next, j, 1], spins[i_next, j, 0])

            d1 = (t2 - t1 + np.pi) % (2 * np.pi) - np.pi
            d2 = (t3 - t2 + np.pi) % (2 * np.pi) - np.pi
            d3 = (t4 - t3 + np.pi) % (2 * np.pi) - np.pi
            d4 = (t1 - t4 + np.pi) % (2 * np.pi) - np.pi

            vorticity[i, j] = np.round((d1 + d2 + d3 + d4) / (2 * np.pi))
    return vorticity

@njit(cache=True)
def get_helicity_data_numba(spins: np.ndarray) -> tuple[float, float]:
    """
    Calculate sum of cos and sin of angle differences in x-direction for helicity modulus.

    Args:
        spins: (N, N, 2) array of unit vectors.

    Returns:
        cos_sum: Sum of cosine of angle differences.
        sin_sum: Sum of sine of angle differences.
    """
    N = spins.shape[0]
    cos_sum = 0.0
    sin_sum = 0.0
    for i in range(N):
        for j in range(N):
            j_next = 0 if j == N - 1 else j + 1
            # cos(theta_i - theta_j) = s_i . s_j
            cos_sum += spins[i, j, 0]*spins[i, j_next, 0] + spins[i, j, 1]*spins[i, j_next, 1]
            # sin(theta_i - theta_j) = cross product
            sin_sum += spins[i, j, 0]*spins[i, j_next, 1] - spins[i, j, 1]*spins[i, j_next, 0]
    return cos_sum, sin_sum

class MonteCarloSimulation(ABC):
    """
    Abstract base class for 2D lattice Monte Carlo simulations.
    Provides infrastructure for equilibration, measurement runs, and statistical analysis.
    """

    def __init__(self, size: int, temp: float, seed: int | None = None):
        """
        Initialize the simulation.

        Args:
            size: Linear dimension L of the N x N lattice.
            temp: Temperature T of the system.
            seed: Optional random seed for reproducibility.

        Raises:
            ValueError: If ``size`` is not a positive integer or ``temp`` is not positive.
        """
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError(f"size must be a positive integer, got {size!r}")
        if temp <= 0.0:
            raise ValueError(f"temp must be positive (T > 0), got {temp}")
        self.size = size
        self.temp = temp
        self.beta = 1.0 / temp
        self.steps = 0
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.spins: np.ndarray | None = None  # To be initialized by subclasses

        # Pre-calculate neighbor indices for Periodic Boundary Conditions (PBC)
        self.idx_next = np.roll(np.arange(size), -1)
        self.idx_prev = np.roll(np.arange(size), 1)


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

        Returns:
            r: Radial distances.
            G_r: Radially averaged correlation values.
        """
        Sk_sq = self._get_structure_factor_squared_unshifted()

        # G(r) is the inverse Fourier transform of S(k) (Wiener-Khinchin theorem)
        G_r_map = np.real(np.fft.ifft2(Sk_sq))

        # Shift the (0,0) correlation to the center of the map for radial averaging
        G_r_map = np.fft.fftshift(G_r_map)

        # Radially average G(r)
        N = self.size
        center = N // 2
        y, x = np.indices((N, N))
        r = np.sqrt((x - center)**2 + (y - center)**2)
        r_int = r.astype(int)

        tbin = np.bincount(r_int.ravel(), G_r_map.ravel())
        nr = np.bincount(r_int.ravel())

        # Avoid division by zero
        radial_profile = np.divide(tbin, nr, out=np.zeros_like(tbin, dtype=float), where=nr!=0)

        # Normalize by G(0) so that G(0) = 1
        if radial_profile[0] != 0:
            radial_profile /= radial_profile[0]

        return np.arange(center), radial_profile[:center]

    def equilibrate(self, n_steps: int) -> None:
        """
        Perform equilibration steps without recording measurements.

        Args:
            n_steps: Number of MC steps to perform.
        """
        for _ in range(n_steps):
            self.step()

    def run(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Run the simulation and record magnetization and energy at each step.

        Args:
            n_steps: Number of MC steps to perform and record.

        Returns:
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
