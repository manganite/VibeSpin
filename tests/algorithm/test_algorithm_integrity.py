"""
Integrity tests for the Metropolis-Hastings implementation.
Probes detailed balance, ergodicity, and proposal symmetry.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.clock_model import ClockSimulation, DiscreteClockSimulation
from models.ising_model import IsingSimulation, ising_energy_numba
from models.xy_model import XYSimulation


def test_ising_detailed_balance():
    """
    Empirical stationarity check for the random-site Metropolis kernel.

    On a 2x2 Ising lattice all 16 microstates and their Boltzmann weights are
    exactly enumerable (using the same double-bond-counting energy kernel the
    dynamics uses, so the comparison is self-consistent). A long trajectory
    from the actual kernel must reproduce this distribution; any violation of
    detailed balance or ergodicity in the compiled kernel shifts the empirical
    frequencies and fails the tolerance.
    """
    import itertools

    size = 2
    temp = 2.5
    beta = 1.0 / temp
    J = 1.0
    n_sweeps = 40_000
    sim = IsingSimulation(size=size, temp=temp, J=J, update='random', seed=3)

    counts: dict[tuple, int] = {}
    for _ in range(n_sweeps):
        sim.step()
        key = tuple(sim.spins.flatten())
        counts[key] = counts.get(key, 0) + 1

    # Exact Boltzmann weights for all 2^4 states of the 2x2 lattice.
    states = list(itertools.product((-1, 1), repeat=4))
    weights = {}
    for st in states:
        spins = np.array(st, dtype=np.int8).reshape(size, size)
        e_total = ising_energy_numba(spins=spins, J=J, idx_next=sim.idx_next) * size * size
        weights[st] = np.exp(-beta * e_total)
    Z = sum(weights.values())

    # Statistical precision: with 40k correlated sweeps, per-state frequencies
    # are accurate to well under 0.02 in absolute probability.
    for st in states:
        p_exact = weights[st] / Z
        p_empirical = counts.get(st, 0) / n_sweeps
        assert p_empirical == pytest.approx(p_exact, abs=0.02), (
            f'State {st}: empirical {p_empirical:.4f} vs Boltzmann {p_exact:.4f}'
        )


def test_ising_ergodicity():
    """
    Verify ergodicity for a tiny Ising system.
    Ensure all 2^N configurations are visited.
    """
    size = 2 # 2x2 = 4 spins -> 16 states
    sim = IsingSimulation(size=size, temp=10.0, update='random', seed=42)

    visited_states = set()
    n_steps = 1000

    for _ in range(n_steps):
        # Convert 2D array to a tuple/string key
        state_key = tuple(sim.spins.flatten())
        visited_states.add(state_key)
        sim.step()

    # For a 2x2 lattice, there are 2^4 = 16 possible states.
    # At T=10, we should definitely hit all 16 in 1000 steps.
    assert len(visited_states) == 16


def test_xy_proposal_symmetry():
    """
    Empirically verify that the XY checkerboard kernel uses symmetric proposals.

    With J=0 every move has dE=0 and is accepted, so the observed per-site
    angle changes ARE the raw proposal distribution. The checkerboard sweep
    visits each site exactly once, so per-sweep angle differences must stay
    inside the documented [-0.5, 0.5] proposal window, be centered at zero
    (the Metropolis symmetric-proposal prerequisite), and split evenly
    between positive and negative rotations.
    """
    n_sweeps = 20
    sim = XYSimulation(size=32, temp=1.0, J=0.0, update='checkerboard', seed=5)

    deltas = []
    for _ in range(n_sweeps):
        before = np.arctan2(sim.spins[..., 1], sim.spins[..., 0])
        sim.step()
        after = np.arctan2(sim.spins[..., 1], sim.spins[..., 0])
        wrapped = (after - before + np.pi) % (2.0 * np.pi) - np.pi
        deltas.append(wrapped.ravel())
    all_deltas = np.concatenate(deltas)

    assert np.max(np.abs(all_deltas)) <= 0.5 + 1e-9
    # Mean of ~20k uniform(-0.5, 0.5) draws has sigma ~ 0.002.
    assert abs(np.mean(all_deltas)) < 0.01
    frac_positive = np.mean(all_deltas > 0)
    assert frac_positive == pytest.approx(0.5, abs=0.02)

def test_discrete_clock_ergodicity():
    """
    Verify ergodicity for a tiny discrete clock system.
    Ensure all q^N configurations are reachable.
    """
    size = 1 # 1x1 = 1 spin
    q = 3
    sim = DiscreteClockSimulation(size=size, temp=10.0, q=q, update='random', seed=42)

    visited_states = set()
    n_steps = 100

    for _ in range(n_steps):
        state_key = sim.spins[0, 0]
        visited_states.add(state_key)
        sim.step()

    # With q=3 and L=1, there are 3 possible states.
    assert len(visited_states) == q


def test_markov_property_reproducibility():
    """
    Verify that the next state depends ONLY on the current state and RNG.
    If we force two simulations into the same state, their next step must be identical
    (assuming same RNG state).
    """
    size = 10
    temp = 2.0
    sim1 = IsingSimulation(size=size, temp=temp, seed=42)
    sim2 = IsingSimulation(size=size, temp=temp, seed=42)

    # Ensure they start identical
    assert np.array_equal(sim1.spins, sim2.spins)

    # Run one step
    sim1.step()
    sim2.step()
    assert np.array_equal(sim1.spins, sim2.spins)

    # Now manually deviate sim2
    sim2.spins[0, 0] *= -1

    # Resync seeds to ensure the same 'random' choices are made
    # (assuming the next step consumes the same amount of entropy)
    from models.simulation_base import _seed_numba
    _seed_numba(seed=100)
    sim1.step()

    _seed_numba(seed=100)
    sim2.step()

    # They should still be different because their starting states were different
    assert not np.array_equal(sim1.spins, sim2.spins)


def test_ising_wolff_detailed_balance():
    """
    Empirical detailed balance for the Ising Wolff algorithm.

    On a 4x4 lattice with J=1 and T=5, consider the transition between
    state A (all spins +1) and state B (site (0,0) flipped to -1).

    The only Wolff path A->B is: seed lands on (0,0) with all 4 like-sign
    bonds rejected.  The only Wolff path B->A is: seed lands on (0,0)
    with no same-sign neighbors available.

        T(A->B) = (1/N^2) * exp(-8 beta J)
        T(B->A) = (1/N^2)
        ratio   = exp(-8 beta J) = exp(-beta * DeltaE)

    This is the Fortuin-Kasteleyn detailed balance condition made explicit for
    a specific, analytically tractable transition.
    """
    L, T, J = 4, 5.0, 1.0
    beta = 1.0 / T
    p_add = 1.0 - np.exp(-2.0 * beta * J)

    state_A = np.ones((L, L), dtype=np.int8)
    state_B = state_A.copy()
    state_B[0, 0] = -1

    # ΔE = 8J: four bonds around (0,0) each flip from -J to +J
    expected_ratio = np.exp(-beta * 8.0 * J)
    # Sanity-check analytical result against the bond-rejection probability.
    assert expected_ratio == pytest.approx((1.0 - p_add) ** 4, rel=1e-9)

    n_trials = 25_000

    # Forward: A -> B
    sim_fwd = IsingSimulation(size=L, temp=T, J=J, update='wolff', seed=42)
    count_A_to_B = 0
    for _ in range(n_trials):
        sim_fwd.spins = state_A.copy()
        sim_fwd.step()
        if np.array_equal(sim_fwd.spins, state_B):
            count_A_to_B += 1

    # Reverse: B -> A
    sim_rev = IsingSimulation(size=L, temp=T, J=J, update='wolff', seed=99)
    count_B_to_A = 0
    for _ in range(n_trials):
        sim_rev.spins = state_B.copy()
        sim_rev.step()
        if np.array_equal(sim_rev.spins, state_A):
            count_B_to_A += 1

    # Statistical precision: ~315 expected A->B counts → ±18 → rtol of ~0.2 is safe
    empirical_ratio = count_A_to_B / count_B_to_A
    np.testing.assert_allclose(empirical_ratio, expected_ratio, rtol=0.2)


def test_ising_wolff_cluster_spin_consistency():
    """
    Structural integrity check: every spin that changes in a Wolff step
    must have had the same value as all other changed spins before the step.

    The cluster growth algorithm only adds same-sign neighbours, so after
    reflection the set of flipped sites must be a monochromatic island.
    """
    sim = IsingSimulation(size=12, temp=2.269, update='wolff', seed=7)
    sim.equilibrate(n_steps=50)

    for _ in range(100):
        spins_before = sim.spins.copy()
        sim.step()
        diff_mask = spins_before != sim.spins
        if not np.any(diff_mask):
            continue  # cluster happened to be empty (pathological edge case)
        flipped_values = spins_before[diff_mask]
        unique_vals = np.unique(flipped_values)
        assert unique_vals.size == 1, (
            f'Wolff cluster contained mixed spin signs: {unique_vals}'
        )


def test_ising_wolff_ergodicity():
    """
    Verify ergodicity of the Wolff cluster algorithm for a tiny Ising system.
    All 2^4 = 16 configurations of a 2x2 lattice must be reachable at high T.
    """
    size = 2  # 2x2 = 4 spins -> 16 states
    sim = IsingSimulation(size=size, temp=10.0, update='wolff', seed=42)

    visited_states = set()
    n_steps = 2000

    for _ in range(n_steps):
        state_key = tuple(sim.spins.flatten())
        visited_states.add(state_key)
        sim.step()

    # At T=10 (beta=0.1), p_add ≈ 0.18 so clusters are small and all configs
    # are reachable; all 16 states must appear within 2000 sweeps.
    assert len(visited_states) == 16


def test_xy_wolff_unit_norm_preservation():
    """
    Verify that the XY Wolff reflection preserves spin unit length exactly.
    After every cluster flip, all spins must remain normalised to 1.
    """
    sim = XYSimulation(size=8, temp=0.8, update='wolff', seed=7)
    for _ in range(20):
        sim.step()
        norms = np.linalg.norm(sim.spins, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_clock_wolff_unit_norm_preservation():
    """
    Verify that the Clock Wolff reflection preserves spin unit length exactly.
    """
    sim = ClockSimulation(size=8, temp=0.5, q=6, A=0.0, update='wolff', seed=13)
    for _ in range(20):
        sim.step()
        norms = np.linalg.norm(sim.spins, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def _discrete_clock_wolff_transition_matrix(*, q: int, temp: float):
    """Build the exact one-step Wolff transition matrix on a 2x2 discrete clock lattice.

    Wolff cluster growth is equivalent to activating every satisfied bond
    independently with probability ``1 - exp(-2 beta J sigma_i sigma_j)`` and
    reflecting the connected component of the seed. On a 2x2 torus each
    neighbour pair is joined by two bonds (direct and wrap-around), exactly as
    in the energy kernel, so all 8 bond slots are enumerated explicitly.
    """
    import itertools

    L, J = 2, 1.0
    beta = 1.0 / temp
    sim = DiscreteClockSimulation(size=L, temp=temp, q=q, update='wolff')
    sigma = sim._wolff_sigma_table
    n_sites = L * L
    bonds = []
    for i in range(L):
        for j in range(L):
            bonds.append((i * L + j, i * L + (j + 1) % L))
            bonds.append((i * L + j, ((i + 1) % L) * L + j))
    states = list(itertools.product(range(q), repeat=n_sites))
    index = {s: k for k, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for k, s in enumerate(states):
        for m in range(q):
            sig = [sigma[(2 * x - m) % (2 * q)] for x in s]
            p_bond = []
            for a, b in bonds:
                prod = sig[a] * sig[b]
                p_bond.append(1.0 - np.exp(-2.0 * beta * J * prod) if prod > 0 else 0.0)
            for seed in range(n_sites):
                for active in itertools.product((0, 1), repeat=len(bonds)):
                    w = 1.0
                    for on, p in zip(active, p_bond, strict=True):
                        w *= p if on else 1.0 - p
                    if w == 0.0:
                        continue
                    comp = {seed}
                    grown = True
                    while grown:
                        grown = False
                        for on, (a, b) in zip(active, bonds, strict=True):
                            if on and ((a in comp) != (b in comp)):
                                comp |= {a, b}
                                grown = True
                    s_new = tuple((m - x) % q if t in comp else x for t, x in enumerate(s))
                    P[k, index[s_new]] += w / (q * n_sites)
    theta = 2.0 * np.pi * np.array(states) / q
    energy = np.array([
        -J * sum(np.cos(th[a] - th[b]) for a, b in bonds) for th in theta
    ])
    pi = np.exp(-beta * (energy - energy.min()))
    pi /= pi.sum()
    return P, pi, index


@pytest.mark.parametrize('q, temp', [(3, 0.9), (4, 1.1)])
def test_discrete_clock_wolff_exact_detailed_balance(q, temp):
    """
    Exact detailed-balance check for the discrete clock Wolff reflection.

    The full transition matrix of the reflection about the q mirror axes is
    enumerated on a 2x2 lattice using the simulation's own projection table.
    It must be row-stochastic and satisfy pi_s P_ss' = pi_s' P_s's for the
    Boltzmann weights of the discrete clock Hamiltonian.
    """
    P, pi, _ = _discrete_clock_wolff_transition_matrix(q=q, temp=temp)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)
    flux = pi[:, None] * P
    np.testing.assert_allclose(flux, flux.T, atol=1e-14)
    # Ergodicity: P is irreducible because single-site clusters can reach
    # every state; a positive power of P must have no zero entries.
    reach = np.linalg.matrix_power((P > 0).astype(float), 4)
    assert np.all(reach > 0)


def test_discrete_clock_wolff_kernel_matches_exact_row():
    """
    The compiled discrete clock Wolff kernel must reproduce the exact
    one-step transition probabilities from a fixed 2x2 configuration.
    """
    from scipy.stats import chisquare

    q, temp = 3, 0.9
    P, _, index = _discrete_clock_wolff_transition_matrix(q=q, temp=temp)
    start = (0, 1, 2, 0)
    s0 = np.array(start, dtype=np.int32).reshape(2, 2)
    sim = DiscreteClockSimulation(size=2, temp=temp, q=q, update='wolff', seed=11)
    n_trials = 30_000
    counts = np.zeros(len(index))
    for _ in range(n_trials):
        sim.spins = s0.copy()
        sim.step()
        counts[index[tuple(int(x) for x in sim.spins.ravel())]] += 1
    expected = P[index[start]]
    support = expected > 0
    assert counts[~support].sum() == 0
    assert chisquare(counts[support], n_trials * expected[support]).pvalue > 1e-3


def test_discrete_clock_wolff_states_and_cluster_size():
    """Wolff reflections keep every spin in {0, ..., q-1} and report the cluster size."""
    sim = DiscreteClockSimulation(size=8, temp=0.8, q=6, update='wolff', seed=21)
    for _ in range(50):
        sim.step()
        assert sim.spins.dtype == np.int32
        assert sim.spins.min() >= 0 and sim.spins.max() < 6
        assert 1 <= sim.last_cluster_size <= 64
