"""
Domain growth analysis for the 2D Ising model.

Starting from a fully disordered (infinite-temperature) initial state, the
system is quenched to a temperature T < T_c.  The characteristic domain size
R(t) is extracted from the structure factor S(k) at each measurement time
using the first-moment estimator:

    R(t) = Σ_k S(k) / Σ_k |k| S(k)

On a log-log plot the Allen-Cahn (Model A) growth law predicts:

    R(t) ~ t^(1/2)
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.ising_model import IsingSimulation
from utils.system_helpers import _BAR_FORMAT, ensure_results_dir, save_plot, setup_logging

# ---------------------------------------------------------------------------
# Physical analysis
# ---------------------------------------------------------------------------

def compute_domain_size(sim: IsingSimulation) -> float:
    """Extract characteristic domain size R from the structure factor S(k).

    The structure factor is centred (DC at (N//2, N//2)) via ``np.fft.fftshift``
    and the first-moment estimator is applied:

        R = 2π × Σ_k S(k) / Σ_k |k| S(k)

    where |k| is in units of 2π/L (reciprocal lattice units) and the sum
    excludes the DC mode.  R grows as coarsening reduces |k_peak|.

    Args:
        sim: An IsingSimulation instance with a current spin configuration.

    Returns:
        Characteristic domain size R in lattice units (> 0), or 0.0 if the
        structure factor is identically zero.
    """
    N = sim.size
    Sk_sq_unshifted = sim._get_structure_factor_squared_unshifted()

    # Centre the DC component and normalise
    Sk = np.fft.fftshift(Sk_sq_unshifted) / (N * N)

    # Wavevector magnitudes in reciprocal lattice units (cycles per site ×2π)
    kvals = np.fft.fftshift(np.fft.fftfreq(N)) * 2.0 * np.pi
    KX, KY = np.meshgrid(kvals, kvals, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)

    # Exclude the DC mode (centre pixel)
    cx = N // 2
    mask = np.ones((N, N), dtype=bool)
    mask[cx, cx] = False

    S_k = Sk[mask]
    K_k = K[mask]

    denominator = float(np.sum(K_k * S_k))
    if denominator == 0.0:
        return 0.0
    # R = 2π / <|k|>_S  —  grows when the peak shifts to smaller |k|
    return 2.0 * np.pi * float(np.sum(S_k) / denominator)


def compute_mean_intercept_length(sim: IsingSimulation) -> float:
    """Estimate domain size using the stereological mean intercept length (MIL).

    Test lines are cast along every row (x-direction) and every column
    (y-direction) of the lattice.  A **domain wall** is detected wherever
    adjacent spins differ in sign.  With periodic boundary conditions the
    wrap-around pair is also counted.

    The mean intercept length (Saltykov / Delesse estimator) is then:

        λ = L / <N_walls>

    where L = N is the line length in lattice units and <N_walls> is the mean
    number of domain-wall crossings per test line, averaged over all 2N lines
    (N rows + N columns).

    For a perfectly ordered lattice (one domain) <N_walls> → 0 and λ → L.
    For a fully disordered lattice <N_walls> → N/2 and λ → 2 (one lattice
    spacing, half a repeat).

    Args:
        sim: An IsingSimulation instance with a current spin configuration.

    Returns:
        Mean intercept length λ in lattice units (≥ 1), or float(N) if the
        lattice is fully ordered (no domain walls).
    """
    if sim.spins is None:
        return 0.0
    spins = sim.spins.astype(np.int8)   # ensure integer type for sign product
    N = sim.size

    # --- interior sign changes (both directions) ----------------------------
    # rows: compare spins[:,j] with spins[:,j+1]  →  shape (N, N-1)
    row_walls = np.sum(spins[:, :-1] * spins[:, 1:] < 0)  # scalar
    # columns: compare spins[i,:] with spins[i+1,:]  →  shape (N-1, N)
    col_walls = np.sum(spins[:-1, :] * spins[1:, :] < 0)  # scalar

    # --- periodic wrap-around -----------------------------------------------
    row_wrap = int(np.sum(spins[:, -1] * spins[:, 0] < 0))   # N values
    col_wrap = int(np.sum(spins[-1, :] * spins[0, :] < 0))   # N values

    # total walls across all 2N test lines
    total_walls = int(row_walls) + int(col_walls) + row_wrap + col_wrap

    # mean walls per line (2N lines total)
    mean_walls = total_walls / (2 * N)

    if mean_walls == 0.0:
        return float(N)   # fully ordered — single domain spans the lattice
    return float(N) / mean_walls


def compute_correlation_length(sim: IsingSimulation) -> float:
    """Estimate domain size as the correlation length ξ from G(r) along x.

    Computes the row-averaged real-space spin-spin pair correlation

        G(r) = <s(x, y) s(x+r, y)>_{x,y}

    via the Wiener-Khinchin theorem (row-wise FFT autocorrelation), then
    finds the first r where G(r) drops below 1/e by linear interpolation.
    This r is the correlation length ξ, which grows as domains coarsen.

    Args:
        sim: An IsingSimulation instance with a current spin configuration.

    Returns:
        Correlation length ξ in lattice units, or 0.0 if G never drops below
        1/e within the available range.
    """
    if sim.spins is None:
        return 0.0
    N = sim.size
    s = sim.spins.astype(float)

    # Row-wise circular autocorrelation via Wiener-Khinchin theorem
    F = np.fft.rfft(s, axis=1)                              # (N, N//2+1)
    G_full = np.mean(np.fft.irfft(np.abs(F) ** 2, n=N, axis=1), axis=0) / N

    # Normalise so G(0) = 1; only use r = 1 … N//2
    G0 = float(G_full[0])
    if G0 == 0.0:
        return 0.0
    r_half = N // 2 + 1
    r_vals = np.arange(1, r_half, dtype=float)
    G = G_full[1:r_half] / G0

    inv_e = 1.0 / np.e
    below = np.where(G < inv_e)[0]
    if len(below) == 0:
        return float(r_vals[-1])   # G stays above 1/e across the whole range
    idx = below[0]
    if idx == 0:
        return float(r_vals[0])
    # Linear interpolation between idx-1 and idx
    r0, r1 = r_vals[idx - 1], r_vals[idx]
    g0, g1 = float(G[idx - 1]), float(G[idx])
    return r0 + (inv_e - g0) * (r1 - r0) / (g1 - g0)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the domain growth simulation and save a log-log plot."""
    parser = argparse.ArgumentParser(description='2D Ising Model Domain Growth Analysis')
    parser.add_argument('--size', type=int, default=512, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.1, help='Quench temperature T')
    parser.add_argument('--max-steps', type=int, default=1000, help='Total MC steps')
    parser.add_argument('--samples', type=int, default=10, help='Number of measurement points')
    parser.add_argument('--fit-min', type=int, default=20, help='Min step for power-law fit')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    T = args.temp
    MAX_STEPS = args.max_steps
    N_SAMPLES = args.samples
    FIT_MIN_STEP = args.fit_min
    T_CRIT: float = 2.269   # Critical temperature for reference

    logger.info(f"Ising domain growth analysis (L={L}, T={T:.3f} < T_c={T_CRIT})")
    logger.info(f"Measuring R(t) at {N_SAMPLES} log-spaced steps up to t={MAX_STEPS} ...")

    # Logarithmically-spaced integer step targets (unique, sorted)
    step_targets: np.ndarray = np.unique(
        np.logspace(0, np.log10(MAX_STEPS), num=N_SAMPLES).astype(int)
    )
    logger.debug(f"Step targets: {step_targets}")

    # Start from a fully disordered state and quench to T < T_c
    sim = IsingSimulation(size=L, temp=T, update='random')

    N_steps = len(step_targets)
    t = np.zeros(N_steps, dtype=float)
    R_sk = np.zeros(N_steps, dtype=float)
    R_mil = np.zeros(N_steps, dtype=float)
    R_xi = np.zeros(N_steps, dtype=float)

    current_step: int = 0

    for i, target in enumerate(tqdm(step_targets, bar_format=_BAR_FORMAT, desc='Sweeping')):
        steps_to_run = int(target) - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = int(target)

        R_sk[i]  = compute_domain_size(sim)
        R_mil[i] = compute_mean_intercept_length(sim)
        R_xi[i]  = compute_correlation_length(sim)
        t[i]     = float(current_step)
        logger.debug(f"Step {current_step}: R_sk={R_sk[i]:.2f}, R_mil={R_mil[i]:.2f}, R_xi={R_xi[i]:.2f}")

    # Power-law fits in log-log space (skip early transient)
    fit_mask = t >= FIT_MIN_STEP

    def power_fit(t_arr: np.ndarray, R_arr: np.ndarray,
                  mask: np.ndarray) -> tuple[float, float] | tuple[None, None]:
        """Return (exponent, prefactor) from a log-log linear fit, or (None, None)."""
        valid = mask & (R_arr > 0)
        if valid.sum() < 3:
            return None, None
        coeffs = np.polyfit(np.log(t_arr[valid]), np.log(R_arr[valid]), 1)
        return float(coeffs[0]), float(np.exp(coeffs[1]))

    exp_sk,  pre_sk  = power_fit(t, R_sk,  fit_mask)
    exp_mil, pre_mil = power_fit(t, R_mil, fit_mask)
    exp_xi,  pre_xi  = power_fit(t, R_xi,  fit_mask)

    if exp_sk  is not None:
        logger.info(f"S(k) first-moment exponent : {exp_sk:.3f}  (Allen–Cahn: 0.500)")
    if exp_mil is not None:
        logger.info(f"MIL exponent               : {exp_mil:.3f}  (Allen–Cahn: 0.500)")
    if exp_xi  is not None:
        logger.info(f"G(r) ξ exponent            : {exp_xi:.3f}  (Allen–Cahn: 0.500)")

    # -----------------------------------------------------------------------
    # Plot — left: linear scale, right: log-log scale
    # -----------------------------------------------------------------------
    title = (f'2D Ising Domain Growth — $T = {T}$ ($< T_c \\approx {T_CRIT}$), $L = {L}$')
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=13)

    for ax in (ax_lin, ax_log):
        # Structure-factor estimator
        ax.plot(t, R_sk, 'o', markersize=4, color='tab:blue',
                label='$R_{S(k)}$ — structure-factor')
        if exp_sk is not None:
            ax.plot(t[fit_mask], pre_sk * t[fit_mask] ** exp_sk,
                    '--', color='tab:blue', linewidth=1.3, alpha=0.7,
                    label=rf'Fit $S(k)$: $t^{{{exp_sk:.2f}}}$')

        # MIL estimator
        ax.plot(t, R_mil, 's', markersize=4, color='tab:orange',
                label='$R_\\mathrm{MIL}$ — mean intercept length')
        if exp_mil is not None:
            ax.plot(t[fit_mask], pre_mil * t[fit_mask] ** exp_mil,
                    '--', color='tab:orange', linewidth=1.3, alpha=0.7,
                    label=rf'Fit MIL: $t^{{{exp_mil:.2f}}}$')

        # Correlation-length estimator
        ax.plot(t, R_xi, '^', markersize=4, color='tab:green',
                label=r'$\xi$ — G(r) correlation length')
        if exp_xi is not None:
            ax.plot(t[fit_mask], pre_xi * t[fit_mask] ** exp_xi,
                    '--', color='tab:green', linewidth=1.3, alpha=0.7,
                    label=rf'Fit $\xi$: $t^{{{exp_xi:.2f}}}$')

        ax.set_xlabel('Monte Carlo Steps $t$', fontsize=12)
        ax.set_ylabel('Domain Size $R(t)$ (lattice units)', fontsize=12)
        ax.legend(fontsize=9)

    ax_lin.set_title('Linear scale', fontsize=11)
    ax_lin.grid(True, alpha=0.25)

    ax_log.set_xscale('log')
    ax_log.set_yscale('log')
    ax_log.set_title('Log-log scale', fontsize=11)
    ax_log.grid(True, which='both', alpha=0.25)

    output_dir = ensure_results_dir(args.output_dir)
    save_plot('domain_growth.png', directory=output_dir)


if __name__ == '__main__':
    main()
