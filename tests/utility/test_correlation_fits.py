"""
Unit tests for the correlation decay fits in ``utils.observables``.

Each fit is checked against data whose answer is known exactly, then against
the degenerate inputs a real measurement can produce, since the scripts write
whatever these return straight into their NPZ files.
"""
from __future__ import annotations

import numpy as np

from utils.observables import fit_correlation_exponent, fit_correlation_length


def test_correlation_length_recovers_an_exact_exponential() -> None:
    """A pure exponential must return its own decay length."""
    r = np.arange(0, 33, dtype=float)
    xi_true = 4.5
    G = 0.8 * np.exp(-r / xi_true)
    assert np.isclose(fit_correlation_length(r=r, G=G), xi_true, rtol=1e-8)


def test_correlation_exponent_recovers_an_exact_power_law() -> None:
    """A pure power law must return its own exponent."""
    r = np.arange(0, 65, dtype=float)
    eta_true = 0.25
    with np.errstate(divide='ignore'):
        G = np.where(r > 0, r ** (-eta_true), 1.0)
    assert np.isclose(fit_correlation_exponent(r=r, G=G), eta_true, rtol=1e-8)


def test_fits_ignore_distances_beyond_a_quarter_of_the_lattice() -> None:
    """Points past L/4, where periodic images flatten G(r), must not be fitted."""
    r = np.arange(0, 33, dtype=float)
    xi_true = 4.0
    G = np.exp(-r / xi_true)
    # r runs to L/2, so the default window ends at r = 16. Corrupting only the
    # points beyond it must leave the answer untouched.
    G_corrupted = G.copy()
    G_corrupted[17:] = 0.5
    assert np.isclose(
        fit_correlation_length(r=r, G=G_corrupted),
        fit_correlation_length(r=r, G=G),
        rtol=1e-12,
    )


def test_explicit_window_overrides_the_default() -> None:
    """An explicit r_max must widen or narrow the window as asked."""
    r = np.arange(0, 33, dtype=float)
    G = np.exp(-r / 4.0)
    G[17:] = 0.5
    assert not np.isclose(
        fit_correlation_length(r=r, G=G, r_max=32), 4.0, rtol=1e-3,
    )


def test_growing_data_yields_no_correlation_length() -> None:
    """A non-negative slope is not a decay, so no length can be reported."""
    r = np.arange(0, 33, dtype=float)
    assert np.isnan(fit_correlation_length(r=r, G=np.exp(r / 4.0)))


def test_fits_return_nan_on_degenerate_input() -> None:
    """Too few usable points, all-zero, and empty input must return NaN."""
    empty = np.array([])
    assert np.isnan(fit_correlation_length(r=empty, G=empty))
    assert np.isnan(fit_correlation_exponent(r=empty, G=empty))

    r = np.arange(0, 33, dtype=float)
    zeros = np.zeros_like(r)
    assert np.isnan(fit_correlation_length(r=r, G=zeros))
    assert np.isnan(fit_correlation_exponent(r=r, G=zeros))

    # A single point inside the window cannot define a line.
    single = np.zeros_like(r)
    single[3] = 0.5
    assert np.isnan(fit_correlation_length(r=r, G=single))
    assert np.isnan(fit_correlation_exponent(r=r, G=single))


def test_fits_tolerate_nan_values_in_the_series() -> None:
    """Non-finite samples are dropped rather than poisoning the whole fit."""
    r = np.arange(0, 33, dtype=float)
    G = np.exp(-r / 4.0)
    G_with_nan = G.copy()
    G_with_nan[5] = np.nan
    result = fit_correlation_length(r=r, G=G_with_nan)
    assert np.isfinite(result)
    assert np.isclose(result, 4.0, rtol=1e-8)
