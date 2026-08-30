"""
2D q-state Clock Model simulation using Metropolis-Hastings and Wolff cluster algorithms.

NOTE: ``ClockSimulation`` (continuous XY-plus-anisotropy form) and
``DiscreteClockSimulation`` (integer state indices with cosine lookup tables) are
collocated here for convenience.  When either class grows beyond roughly 400 lines
of unique logic, split them into ``clock_model_continuous.py`` and
``clock_model_discrete.py`` and keep this file as a re-export shim for backward
compatibility.
"""
from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

from .simulation_base import (
    MonteCarloSimulation,
    VectorSpinObservablesMixin,
    calculate_vortex_density_numba,
    calculate_vorticity_numba,
    get_helicity_data_numba,
    o2_wolff_step_numba,
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

    Parameters
    ----------
    spins : np.ndarray
        (N, N, 2) array of unit vectors.
    beta : float
        Inverse temperature 1/kT.
    J : float
        Coupling constant.
    A : float
        Anisotropy strength.
    q : int
        Number of clock states.
    idx_next : np.ndarray
        Pre-calculated next-neighbor indices.
    idx_prev : np.ndarray
        Pre-calculated previous-neighbor indices.

    Returns
    -------
    np.ndarray
        Updated spins array.
    """
    N = spins.shape[0]
    # Batch generate random updates
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

                # Neighbor sum
                nx = spins[iprv, j, 0] + spins[inxt, j, 0] + spins[i, jprv, 0] + spins[i, jnxt, 0]
                ny = spins[iprv, j, 1] + spins[inxt, j, 1] + spins[i, jprv, 1] + spins[i, jnxt, 1]

                sx, sy = spins[i, j, 0], spins[i, j, 1]

                # Proposed update from pre-calculated values
                delta = deltas[i, j]
                c, s = cos_vals[i, j], sin_vals[i, j]

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

                if dE <= 0 or random_vals[i, j] < np.exp(-dE * beta):
                    spins[i, j, 0] = sx_new
                    spins[i, j, 1] = sy_new
    return spins


@njit(cache=True, fastmath=True, parallel=True)
def clock_step_parallel_numba(
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
    Parallel version of clock_step_numba.

    Parameters
    ----------
    spins : np.ndarray
        (N, N, 2) array of unit vectors.
    beta : float
        Inverse temperature 1/kT.
    J : float
        Coupling constant.
    A : float
        Anisotropy strength.
    q : int
        Number of clock states.
    idx_next : np.ndarray
        Pre-calculated next-neighbor indices.
    idx_prev : np.ndarray
        Pre-calculated previous-neighbor indices.

    Returns
    -------
    np.ndarray
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
                delta = deltas[i, j]
                c, s = cos_vals[i, j], sin_vals[i, j]

                sx_new = sx * c - sy * s
                sy_new = sx * s + sy * c
                norm = np.sqrt(sx_new**2 + sy_new**2)
                sx_new /= norm
                sy_new /= norm

                dE_inter = -J * ((sx_new * nx + sy_new * ny) - (sx * nx + sy * ny))
                phi_old = np.arctan2(sy, sx)
                phi_new = phi_old + delta
                dE_aniso = -A * (np.cos(q * phi_new) - np.cos(q * phi_old))

                dE = dE_inter + dE_aniso

                if dE <= 0 or random_vals[i, j] < np.exp(-dE * beta):
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

    Parameters
    ----------
    spins : np.ndarray
        (N, N, 2) array of unit vectors.
    beta : float
        Inverse temperature 1/kT.
    J : float
        Coupling constant.
    A : float
        Anisotropy strength.
    q : int
        Number of clock states.
    idx_next : np.ndarray
        Pre-calculated next-neighbor indices.
    idx_prev : np.ndarray
        Pre-calculated previous-neighbor indices.

    Returns
    -------
    np.ndarray
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

        # Neighbor sum
        nx = spins[iprv, j, 0] + spins[inxt, j, 0] + spins[i, jprv, 0] + spins[i, jnxt, 0]
        ny = spins[iprv, j, 1] + spins[inxt, j, 1] + spins[i, jprv, 1] + spins[i, jnxt, 1]

        sx, sy = spins[i, j, 0], spins[i, j, 1]

        # Proposed update
        delta = deltas[k]
        c, s = cos_vals[k], sin_vals[k]

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

        if dE <= 0 or random_vals[k] < np.exp(-dE * beta):
            spins[i, j, 0] = sx_new
            spins[i, j, 1] = sy_new
    return spins


@njit(cache=True, fastmath=True)
def clock_energy_numba(
    *, spins: np.ndarray, J: float, A: float, q: int, idx_next: np.ndarray
) -> float:
    """
    Calculate the total energy of the Clock Model lattice.

    Parameters
    ----------
    spins : np.ndarray
        (N, N, 2) array of unit vectors.
    J : float
        Coupling constant.
    A : float
        Anisotropy strength.
    q : int
        Number of clock states.
    idx_next : np.ndarray
        Pre-calculated next-neighbor indices.

    Returns
    -------
    energy : float
        Total energy per site.
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


class ClockSimulation(VectorSpinObservablesMixin, MonteCarloSimulation):
    """
    Simulation of the 2D q-state clock model on a square lattice.
    """

    _VALID_UPDATES: frozenset = frozenset({'checkerboard', 'random', 'wolff'})

    def __init__(
        self,
        *,
        size: int,
        temp: float,
        J: float = 1.0,
        A: float = 1.0,
        q: int = 6,
        update: str = 'checkerboard',
        init_state: str = 'random',
        parallel: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize the Clock Model simulation.

        Parameters
        ----------
        size : int
            Linear dimension L of the L x L lattice.
        temp : float
            Temperature T.
        J : float
            Coupling constant (default 1.0).
        A : float
            Anisotropy strength (default 1.0).
        q : int
            Number of clock states (default 6). Must be \u2265 2.
        update : str
            Update scheme - ``'checkerboard'`` (default, faster),
            ``'random'`` (random sequential Metropolis, physical
            stochastic dynamics for kinetics studies), or
            ``'wolff'`` (Wolff-Evertz cluster algorithm using the exchange
            term J; detailed balance holds exactly only when A=0).
        init_state : str
            Initial spin configuration: ``'random'`` (default) or ``'ordered'``.
        parallel : bool
            Whether to use parallelized Numba kernels (only for
            checkerboard). Parallel kernels are NOT seed-reproducible:
            only the calling thread's Numba RNG is seeded, so two runs
            with the same seed may differ.
        seed : int | None
            Optional random seed for reproducibility.

        Raises
        ------
        ValueError
            If ``q`` is less than 2 or update scheme is unknown.
        """
        super().__init__(size=size, temp=temp, init_state=init_state, seed=seed)
        if q < 2:
            raise ValueError(f'q must be >= 2 (number of clock states), got {q}')
        if update not in self._VALID_UPDATES:
            valid_opts = sorted(self._VALID_UPDATES)
            raise ValueError(f'Unknown update scheme {update!r}. Valid options: {valid_opts}')
        if update == 'wolff' and A != 0.0:
            # The Wolff-Evertz reflection ignores the anisotropy term, so the
            # sampled distribution is exactly Boltzmann only for A = 0.
            warnings.warn(
                f'ClockSimulation with update=\'wolff\' ignores the anisotropy '
                f'term (A={A}); detailed balance holds exactly only for A=0. '
                f'Set A=0.0 for exchange-only studies or use a local update '
                f'scheme for anisotropic sampling.',
                UserWarning,
                stacklevel=2,
            )
        self.J = J
        self.A = A
        self.q = q
        self.update = update
        self.parallel = parallel

        if self.init_state == 'ordered':
            # Ordered state: all spins at angle 0 -> (1, 0)
            self.spins = np.zeros((size, size, 2), dtype=float)
            self.spins[..., 0] = 1.0
        else:
            # Initialize random spins as 2D unit vectors
            angles = self.rng.uniform(0, 2 * np.pi, size=(size, size))
            self.spins = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

    def step(self) -> None:
        """Perform one Monte Carlo step using Numba."""
        if self.spins is not None:
            self._reseed_numba_for_step()

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
            elif self.update == 'wolff':
                self.spins, self.last_cluster_size = o2_wolff_step_numba(
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
                self.spins = clock_step_parallel_numba(
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


# ---------------------------------------------------------------------------
# Discrete q-state clock model (integer spin representation)
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def discrete_clock_step_numba(
    *,
    spins: np.ndarray,
    beta: float,
    J: float,
    q: int,
    cos_table: np.ndarray,
    idx_next: np.ndarray,
    idx_prev: np.ndarray,
) -> np.ndarray:
    """
    One Metropolis sweep of the discrete clock model (checkerboard update).

    Parameters
    ----------
    spins : np.ndarray
        (N, N) array of integer spin states in {0, ..., q-1}.
    beta : float
        Inverse temperature 1/kT.
    J : float
        Coupling constant.
    q : int
        Number of clock states.
    cos_table : np.ndarray
        Pre-computed cos(2*pi*d/q) for d in {0, ..., q-1}.
    idx_next : np.ndarray
        Pre-calculated next-neighbor indices.
    idx_prev : np.ndarray
        Pre-calculated previous-neighbor indices.

    Returns
    -------
    np.ndarray
        Updated spins array.
    """
    N = spins.shape[0]
    proposals = np.random.randint(0, q, size=(N, N))
    random_vals = np.random.random(size=(N, N))

    for parity in range(2):
        for i in range(N):
            start_j = (parity + i) % 2
            inxt = idx_next[i]
            iprv = idx_prev[i]
            for j in range(start_j, N, 2):
                jnxt = idx_next[j]
                jprv = idx_prev[j]

                s_old = spins[i, j]
                s_new = proposals[i, j]
                if s_new == s_old:
                    continue

                # Neighbor states
                n0 = spins[iprv, j]
                n1 = spins[inxt, j]
                n2 = spins[i, jprv]
                n3 = spins[i, jnxt]

                # dE = -J * sum_nn [cos(theta_new - theta_nn) - cos(theta_old - theta_nn)]
                dE = -J * (
                    cos_table[(s_new - n0) % q]
                    + cos_table[(s_new - n1) % q]
                    + cos_table[(s_new - n2) % q]
                    + cos_table[(s_new - n3) % q]
                    - cos_table[(s_old - n0) % q]
                    - cos_table[(s_old - n1) % q]
                    - cos_table[(s_old - n2) % q]
                    - cos_table[(s_old - n3) % q]
                )

                if dE <= 0 or random_vals[i, j] < np.exp(-dE * beta):
                    spins[i, j] = s_new
    return spins


@njit(cache=True, fastmath=True, parallel=True)
def discrete_clock_step_parallel_numba(
    *,
    spins: np.ndarray,
    beta: float,
    J: float,
    q: int,
    cos_table: np.ndarray,
    idx_next: np.ndarray,
    idx_prev: np.ndarray,
) -> np.ndarray:
    """
    Parallel version of discrete_clock_step_numba.

    Parameters
    ----------
    spins : np.ndarray
        (N, N) array of integer spin states in {0, ..., q-1}.
    beta : float
        Inverse temperature 1/kT.
    J : float
        Coupling constant.
    q : int
        Number of clock states.
    cos_table : np.ndarray
        Pre-computed cos(2*pi*d/q) for d in {0, ..., q-1}.
    idx_next : np.ndarray
        Pre-calculated next-neighbor indices.
    idx_prev : np.ndarray
        Pre-calculated previous-neighbor indices.

    Returns
    -------
    np.ndarray
        Updated spins array.
    """
    N = spins.shape[0]
    proposals = np.random.randint(0, q, size=(N, N))
    random_vals = np.random.random(size=(N, N))

    for parity in range(2):
        for i in prange(N):
            start_j = (parity + i) % 2
            inxt = idx_next[i]
            iprv = idx_prev[i]
            for j in range(start_j, N, 2):
                jnxt = idx_next[j]
                jprv = idx_prev[j]

                s_old = spins[i, j]
                s_new = proposals[i, j]
                if s_new == s_old:
                    continue

                n0, n1 = spins[iprv, j], spins[inxt, j]
                n2, n3 = spins[i, jprv], spins[i, jnxt]

                dE = -J * (
                    cos_table[(s_new - n0) % q]
                    + cos_table[(s_new - n1) % q]
                    + cos_table[(s_new - n2) % q]
                    + cos_table[(s_new - n3) % q]
                    - cos_table[(s_old - n0) % q]
                    - cos_table[(s_old - n1) % q]
                    - cos_table[(s_old - n2) % q]
                    - cos_table[(s_old - n3) % q]
                )

                if dE <= 0 or random_vals[i, j] < np.exp(-dE * beta):
                    spins[i, j] = s_new
    return spins


@njit(cache=True, fastmath=True)
def discrete_clock_step_random_numba(
    *,
    spins: np.ndarray,
    beta: float,
    J: float,
    q: int,
    cos_table: np.ndarray,
    idx_next: np.ndarray,
    idx_prev: np.ndarray,
) -> np.ndarray:
    """
    One Metropolis sweep of the discrete clock model (random sequential update).

    N^2 single-spin flip attempts at uniformly random sites, giving physical
    stochastic dynamics suitable for kinetics studies.

    Parameters
    ----------
    spins : np.ndarray
        (N, N) array of integer spin states in {0, ..., q-1}.
    beta : float
        Inverse temperature 1/kT.
    J : float
        Coupling constant.
    q : int
        Number of clock states.
    cos_table : np.ndarray
        Pre-computed cos(2*pi*d/q) for d in {0, ..., q-1}.
    idx_next : np.ndarray
        Pre-calculated next-neighbor indices.
    idx_prev : np.ndarray
        Pre-calculated previous-neighbor indices.

    Returns
    -------
    np.ndarray
        Updated spins array.
    """
    N = spins.shape[0]
    num_attempts = N * N
    indices = np.random.randint(0, num_attempts, size=num_attempts)
    proposals = np.random.randint(0, q, size=num_attempts)
    random_vals = np.random.random(size=num_attempts)

    for k in range(num_attempts):
        idx = indices[k]
        i = idx // N
        j = idx % N

        s_old = spins[i, j]
        s_new = proposals[k]
        if s_new == s_old:
            continue

        inxt = idx_next[i]
        iprv = idx_prev[i]
        jnxt = idx_next[j]
        jprv = idx_prev[j]

        n0 = spins[iprv, j]
        n1 = spins[inxt, j]
        n2 = spins[i, jprv]
        n3 = spins[i, jnxt]

        dE = -J * (
            cos_table[(s_new - n0) % q]
            + cos_table[(s_new - n1) % q]
            + cos_table[(s_new - n2) % q]
            + cos_table[(s_new - n3) % q]
            - cos_table[(s_old - n0) % q]
            - cos_table[(s_old - n1) % q]
            - cos_table[(s_old - n2) % q]
            - cos_table[(s_old - n3) % q]
        )

        if dE <= 0 or random_vals[k] < np.exp(-dE * beta):
            spins[i, j] = s_new
    return spins


@njit(cache=True, fastmath=True)
def discrete_clock_energy_numba(
    *,
    spins: np.ndarray,
    J: float,
    q: int,
    cos_table: np.ndarray,
    idx_next: np.ndarray,
) -> float:
    """
    Total energy per site of the discrete clock model.

    Sums -J * cos(theta_i - theta_j) over unique right/down neighbor pairs.

    Parameters
    ----------
    spins : np.ndarray
        (N, N) array of integer spin states in {0, ..., q-1}.
    J : float
        Coupling constant.
    q : int
        Number of clock states.
    cos_table : np.ndarray
        Pre-computed cos(2*pi*d/q) for d in {0, ..., q-1}.
    idx_next : np.ndarray
        Pre-calculated next-neighbor indices.

    Returns
    -------
    float
        Energy per site.
    """
    N = spins.shape[0]
    energy = 0.0
    for i in range(N):
        inxt = idx_next[i]
        for j in range(N):
            jnxt = idx_next[j]
            s = spins[i, j]
            energy -= J * (
                cos_table[(s - spins[i, jnxt]) % q] + cos_table[(s - spins[inxt, j]) % q]
            )
    return energy / (N * N)


class DiscreteClockSimulation(VectorSpinObservablesMixin, MonteCarloSimulation):
    """
    Simulation of the 2D q-state clock model with discrete integer spins.

    Each spin takes one of q evenly spaced orientations theta_k = 2*pi*k/q.
    The Hamiltonian is E = -J * sum_<i,j> cos(theta_i - theta_j).

    Unlike ``ClockSimulation``, which uses continuous unit-vector spins plus
    an anisotropy term, this class represents spins as integers in {0, ..., q-1}
    and enforces the discrete symmetry exactly.
    """

    _VALID_UPDATES: frozenset = frozenset({'checkerboard', 'random'})

    def __init__(
        self,
        *,
        size: int,
        temp: float,
        J: float = 1.0,
        q: int = 6,
        update: str = 'checkerboard',
        init_state: str = 'random',
        parallel: bool = False,
        seed: int | None = None,
    ):
        """
        Initialize the discrete clock model simulation.

        Parameters
        ----------
        size : int
            Linear dimension L of the L x L lattice.
        temp : float
            Temperature T.
        J : float
            Coupling constant (default 1.0).
        q : int
            Number of clock states (default 6). Must be >= 2.
        update : str
            Update scheme - ``'checkerboard'`` (default, faster) or
            ``'random'`` (random sequential Metropolis, more physical
            stochastic dynamics for kinetics studies).
        init_state : str
            Initial spin configuration: ``'random'`` (default) or ``'ordered'``.
        parallel : bool
            Whether to use parallelized Numba kernels (only for
            checkerboard). Parallel kernels are NOT seed-reproducible:
            only the calling thread's Numba RNG is seeded, so two runs
            with the same seed may differ.
        seed : int | None
            Optional random seed for reproducibility.

        Raises
        ------
        ValueError
            If ``q`` is less than 2 or update scheme is unknown.
        """
        super().__init__(size=size, temp=temp, init_state=init_state, seed=seed)
        if q < 2:
            raise ValueError(f'q must be >= 2 (number of clock states), got {q}')
        if update not in self._VALID_UPDATES:
            valid_opts = sorted(self._VALID_UPDATES)
            raise ValueError(f'Unknown update scheme {update!r}. Valid options: {valid_opts}')
        self.J = J
        self.q = q
        self.update = update
        self.parallel = parallel

        # Pre-compute lookup tables (no trig inside kernels)
        angles = 2.0 * np.pi * np.arange(q) / q
        self.cos_table = np.cos(angles)  # cos(2*pi*d/q)  for dE calculation
        self._cos_angles = np.cos(angles)  # per-state cos for observables
        self._sin_angles = np.sin(angles)  # per-state sin for observables

        if self.init_state == 'ordered':
            # Initialize ordered discrete spins (state 0)
            self.spins = np.zeros((size, size), dtype=np.int32)
        else:
            # Initialize random discrete spins
            self.spins = self.rng.integers(0, q, size=(size, size), dtype=np.int32)

    def step(self) -> None:
        """Perform one Monte Carlo sweep using Numba."""
        if self.spins is not None:
            self._reseed_numba_for_step()

            if self.update == 'random':
                self.spins = discrete_clock_step_random_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    q=self.q,
                    cos_table=self.cos_table,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
            elif self.parallel:
                self.spins = discrete_clock_step_parallel_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    q=self.q,
                    cos_table=self.cos_table,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
            else:
                self.spins = discrete_clock_step_numba(
                    spins=self.spins,
                    beta=self.beta,
                    J=self.J,
                    q=self.q,
                    cos_table=self.cos_table,
                    idx_next=self.idx_next,
                    idx_prev=self.idx_prev,
                )
        self.steps += 1

    def _get_magnetization(self) -> float:
        """Calculate magnetization magnitude per spin."""
        if self.spins is not None:
            mx = np.sum(self._cos_angles[self.spins])
            my = np.sum(self._sin_angles[self.spins])
            return float(np.sqrt(mx**2 + my**2) / (self.size**2))
        return 0.0

    def _get_energy(self) -> float:
        """Calculate energy per spin."""
        if self.spins is not None:
            return float(
                discrete_clock_energy_numba(
                    spins=self.spins,
                    J=self.J,
                    q=self.q,
                    cos_table=self.cos_table,
                    idx_next=self.idx_next,
                )
            )
        return 0.0

    def _spins_as_vectors(self) -> np.ndarray:
        """Convert integer spin states to (N, N, 2) unit-vector representation."""
        sx = self._cos_angles[self.spins]
        sy = self._sin_angles[self.spins]
        return np.stack([sx, sy], axis=-1)

    def _calculate_vorticity(self) -> np.ndarray:
        """Calculate the vorticity (winding number) of each plaquette."""
        if self.spins is not None:
            vectors = self._spins_as_vectors()
            return np.asarray(calculate_vorticity_numba(spins=vectors, idx_next=self.idx_next))
        return np.array([])

    def _get_vortex_density(self) -> float:
        """Calculate vortex density n_v, the fraction of plaquettes with non-zero winding."""
        if self.spins is not None:
            vectors = self._spins_as_vectors()
            return float(calculate_vortex_density_numba(spins=vectors, idx_next=self.idx_next))
        return 0.0

    def _get_helicity_data(self) -> tuple[float, float]:
        """Calculate sum of cos and sin of angle differences in x-direction."""
        if self.spins is not None:
            vectors = self._spins_as_vectors()
            cos_sum, sin_sum = get_helicity_data_numba(spins=vectors, idx_next=self.idx_next)
            return float(cos_sum), float(sin_sum)
        return 0.0, 0.0

    def _get_structure_factor_squared_unshifted(self) -> np.ndarray:
        """Calculate the unshifted squared magnitude of the Fourier transform."""
        if self.spins is not None:
            vectors = self._spins_as_vectors()
            sx = vectors[..., 0]
            sy = vectors[..., 1]
            Sk_x = np.fft.fft2(sx)
            Sk_y = np.fft.fft2(sy)
            return np.asarray(np.abs(Sk_x) ** 2 + np.abs(Sk_y) ** 2)
        return np.array([])


def main() -> None:
    """CLI entry point for Clock simulation example."""
    import argparse
    import logging

    from utils.system import setup_logging

    parser = argparse.ArgumentParser(description='Clock Model Quick Example')
    parser.add_argument('--size', type=int, default=128, help='Lattice size L')
    parser.add_argument('--temp', type=float, default=0.2, help='Temperature T')
    parser.add_argument('--q', type=int, default=6, help='Clock states q')
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
        f'Initializing {args.q}-state Clock Model (L={args.size}, T={args.temp}, '
        f'update={args.update}, parallel={args.parallel})...'
    )
    sim = ClockSimulation(
        size=args.size, temp=args.temp, q=args.q, seed=args.seed, parallel=args.parallel,
        update=args.update
    )

    logger.info(f'Running for {args.steps} steps...')
    mag_history, energy_history = sim.run(n_steps=args.steps)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'2D {args.q}-state Clock Model - $L={args.size}, T={args.temp}$', fontsize=14)

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
    vort = sim.calculate_vorticity()
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


if __name__ == '__main__':
    main()
