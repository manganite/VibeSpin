"""
Domain snapshot visualisation for the 2D Ising model.

Quenches from a disordered state to T < T_c and records the spin configuration
at every 10th Monte Carlo step, plotting them as a grid of images so the
coarsening of magnetic domains is clearly visible over time.

Row 1 – Spin configurations.
Row 2 – Circularly-averaged structure factor S(|k|) on log-log axes.
Row 3 – Real-space pair correlation G(r) along the x-direction, averaged over
         all rows and normalised so G(0) = 1.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from models.ising_model import IsingSimulation
from utils.system_helpers import ensure_results_dir, save_plot


def radial_average_sk(spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the circularly averaged structure factor S(|k|).

    Bins S(k) by integer pixel radius from the DC centre of the shifted FFT,
    then averages within each annular bin.

    Args:
        spins: (N, N) integer spin array.

    Returns:
        k_vals: Wavevector magnitudes in units of 2π/N (reciprocal lattice).
        S_radial: Mean S(k) value for each annular bin.
    """
    N = spins.shape[0]
    Sk_raw = np.abs(np.fft.fft2(spins.astype(float))) ** 2
    Sk = np.fft.fftshift(Sk_raw) / (N * N)

    cx = N // 2
    iy, ix = np.indices((N, N))
    r_int = np.sqrt((ix - cx) ** 2 + (iy - cx) ** 2).astype(int)

    # Average within each annular bin up to the Nyquist radius
    r_max = cx
    mask = r_int <= r_max
    tbin = np.bincount(r_int[mask].ravel(), Sk[mask].ravel())
    nbin = np.bincount(r_int[mask].ravel())
    S_radial = np.where(nbin > 0, tbin / nbin, 0.0)

    # Convert bin index → |k| in reciprocal lattice units (2π/N per bin)
    k_vals = np.arange(len(S_radial)) * (2.0 * np.pi / N)

    return k_vals, S_radial


def pair_correlation_x(spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the real-space spin-spin pair correlation G(r) along x.

    Uses the Wiener-Khinchin theorem: the autocorrelation of each row is the
    inverse FFT of the row's power spectrum.  Results are averaged over all
    rows (y positions) and normalised so G(0) = 1.

    With periodic boundary conditions the result is a circular autocorrelation,
    so only the first half (r = 0 … N//2) is meaningful.

    Args:
        spins: (N, N) integer spin array.

    Returns:
        r_vals: Lag distances r = 0 … N//2 in lattice units.
        G: Normalised pair correlation G(r) / G(0).
    """
    N = spins.shape[0]
    s = spins.astype(float)

    # Row-wise power spectrum and inverse FFT → circular autocorrelation per row
    F = np.fft.rfft(s, axis=1)               # shape (N, N//2+1)
    autocorr = np.fft.irfft(np.abs(F) ** 2, n=N, axis=1)  # shape (N, N)

    # Average over rows and normalise by N (circular sum → mean)
    G_full = np.mean(autocorr, axis=0) / N   # G(r), length N

    r_half = N // 2 + 1
    r_vals = np.arange(r_half)
    G = G_full[:r_half]

    # Normalise so G(0) = 1
    if G[0] != 0.0:
        G = G / G[0]

    return r_vals, G


def main() -> None:
    """Run the snapshot simulation and save a 2-row (spins + S(|k|)) figure."""
    parser = argparse.ArgumentParser(description='2D Ising Model Domain Snapshot Visualisation')
    parser.add_argument('--size', type=int, default=512, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=2.0, help='Quench temperature T')
    parser.add_argument('--targets', type=int, nargs='+', default=[1, 10, 100, 1000],
                        help='MC steps at which to take snapshots')
    parser.add_argument('--output-dir', type=str, default='results/ising', help='Output directory')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    L = args.size
    T = args.temp
    STEP_TARGETS = sorted(args.targets)
    T_CRIT: float = 2.269

    print(f"Ising domain snapshots (L={L}, T={T})")
    print(f"Recording snapshots at steps {STEP_TARGETS} ...")

    sim = IsingSimulation(size=L, temp=T, update='random')
    n_targets: int = len(STEP_TARGETS)
    snapshots_spins: np.ndarray = np.zeros((n_targets, L, L), dtype=np.int8)
    snapshots_t: np.ndarray = np.zeros(n_targets, dtype=int)
    current_step: int = 0

    for i, target in enumerate(STEP_TARGETS):
        steps_to_run = target - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = target
        if sim.spins is not None:
            snapshots_spins[i] = sim.spins.copy()
            snapshots_t[i] = target

    print(f"Collected {n_targets} snapshots. Saving figure ...")

    # --- 3 × N layout: row 0 = spins, row 1 = S(|k|), row 2 = G(r) ---------
    n_cols = n_targets
    fig, axes = plt.subplots(3, n_cols,
                             figsize=(n_cols * 3.5, 10.8),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.30})
    fig.suptitle(
        f'2D Ising Domain Evolution — T = {T} (< T_c ≈ {T_CRIT}), L = {L}',
        fontsize=13, y=1.01,
    )

    for col in range(n_targets):
        t = int(snapshots_t[col])
        spins = snapshots_spins[col]
        # --- top row: spin configuration ------------------------------------
        ax_spin = axes[0, col]
        ax_spin.imshow(spins, cmap='binary', interpolation='none',
                       vmin=-1, vmax=1)
        ax_spin.set_title(f't = {t} sweep{"s" if t > 1 else ""}', fontsize=11)
        ax_spin.axis('off')

        # --- bottom row: circularly averaged S(|k|) -------------------------
        ax_sk = axes[1, col]
        k_vals, S_radial = radial_average_sk(spins)

        # Skip k = 0 (DC mode) for the plot
        ax_sk.plot(k_vals[1:], S_radial[1:], linewidth=1.2)
        ax_sk.set_xscale('log')
        ax_sk.set_yscale('log')
        ax_sk.set_xlabel('$|k|$ (rad / lattice site)', fontsize=9)
        if col == 0:
            ax_sk.set_ylabel('$S(|k|)$', fontsize=9)
        ax_sk.grid(True, which='both', alpha=0.25)

        # --- third row: real-space pair correlation G(r) along x -------------
        ax_gr = axes[2, col]
        r_vals, G = pair_correlation_x(spins)

        # Skip r=0 so the log x-axis is well defined
        ax_gr.plot(r_vals[1:], G[1:], linewidth=1.2)
        ax_gr.axhline(0, color='tab:gray', linewidth=0.7, linestyle='--')
        ax_gr.axhline(1.0 / np.e, color='tab:red', linewidth=0.8,
                      linestyle=':', alpha=0.7, label='$1/e$')
        ax_gr.set_xscale('log')
        ax_gr.set_xlabel('$r$ (lattice sites)', fontsize=9)
        if col == 0:
            ax_gr.set_ylabel('$G(r)$ / $G(0)$', fontsize=9)
        ax_gr.grid(True, which='both', alpha=0.25)

        # Find ξ where G(r) first drops below 1/e via linear interpolation
        inv_e = 1.0 / np.e
        r_plot = r_vals[1:]
        G_plot = G[1:]
        below = np.where(G_plot < inv_e)[0]
        if len(below) > 0:
            idx = below[0]
            if idx > 0:
                # Linear interpolation between idx-1 and idx
                r0, r1 = float(r_plot[idx - 1]), float(r_plot[idx])
                g0, g1 = float(G_plot[idx - 1]), float(G_plot[idx])
                xi = r0 + (inv_e - g0) * (r1 - r0) / (g1 - g0)
            else:
                xi = float(r_plot[idx])
            ax_gr.axvline(xi, color='tab:red', linewidth=1.0, linestyle='--', alpha=0.8)
            ax_gr.text(xi * 1.15, inv_e + 0.04,
                       f'$\\xi = {xi:.1f}$', fontsize=8, color='tab:red')


    plt.tight_layout()
    output_dir = ensure_results_dir(args.output_dir)
    save_plot('domain_snapshots.png', directory=output_dir)



if __name__ == '__main__':
    main()
