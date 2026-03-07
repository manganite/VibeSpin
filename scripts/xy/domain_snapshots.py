"""
Domain snapshot visualisation for the 2D XY model.

Quenches from a disordered state to T < T_BKT and records the spin configuration
at multiple time steps, plotting phase configurations, vorticity maps, and
radially averaged correlation functions G(r).
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.xy_model import XYSimulation
from utils.system_helpers import ensure_results_dir, save_plot, setup_logging


def main() -> None:
    """Run the snapshot simulation and generate a multi-row evolution figure."""
    parser = argparse.ArgumentParser(description='2D XY Model Domain Snapshot Visualisation')
    parser.add_argument('--size', type=int, default=128, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.5, help='Quench temperature T')
    parser.add_argument('--targets', type=int, nargs='+', default=[1, 10, 100, 1000],
                        help='MC steps at which to take snapshots')
    parser.add_argument('--output-dir', type=str, default='results/xy', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    T = args.temp
    STEP_TARGETS = sorted(args.targets)
    T_BKT: float = 0.893

    logger.info(f"XY domain snapshots (L={L}, T={T})")
    logger.info(f"Recording snapshots at steps {STEP_TARGETS} ...")

    sim = XYSimulation(size=L, temp=T)
    n_targets: int = len(STEP_TARGETS)
    
    # Storage for snapshots
    snapshots_angles: list[np.ndarray] = []
    snapshots_vort: list[np.ndarray] = []
    snapshots_gr: list[tuple[np.ndarray, np.ndarray]] = []
    
    current_step: int = 0

    for i, target in enumerate(STEP_TARGETS):
        steps_to_run = target - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = target
        
        if sim.spins is not None:
            # 1. Phase angles
            angles = np.arctan2(sim.spins[..., 1], sim.spins[..., 0])
            snapshots_angles.append(angles)
            
            # 2. Vorticity
            snapshots_vort.append(sim._calculate_vorticity())
            
            # 3. Correlation function
            snapshots_gr.append(sim._calculate_correlation_function())
            
            logger.debug(f"Captured snapshot at step {target}")

    logger.info(f"Collected {n_targets} snapshots. Saving figure ...")

    # --- 3 × N layout: row 0 = phase, row 1 = vorticity, row 2 = G(r) ---------
    n_cols = n_targets
    fig, axes = plt.subplots(3, n_cols,
                             figsize=(n_cols * 4, 12.5),
                             gridspec_kw={'hspace': 0.45, 'wspace': 0.25})
    fig.suptitle(
        f'2D XY Model Domain Evolution — T = {T} (< T_BKT ≈ {T_BKT}), L = {L}',
        fontsize=14, y=0.98,
    )

    for col in range(n_targets):
        t = STEP_TARGETS[col]
        
        # --- Row 0: Phase Configuration ---
        ax_phase = axes[0, col]
        im_p = ax_phase.imshow(snapshots_angles[col], cmap='hsv', interpolation='none',
                               vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title(f't = {t} sweep{"s" if t > 1 else ""}', fontsize=12)
        ax_phase.axis('off')
        if col == n_targets - 1:
            fig.colorbar(im_p, ax=ax_phase, label='Phase (rad)', shrink=0.8)

        # --- Row 1: Vorticity Map ---
        ax_vort = axes[1, col]
        vort = snapshots_vort[col]
        im_v = ax_vort.imshow(vort, cmap='bwr', interpolation='none', vmin=-1, vmax=1)
        ax_vort.axis('off')
        # Count non-zero vorticity
        v_count = int(np.sum(np.abs(vort)))
        ax_vort.set_title(f'Vortices: {v_count}', fontsize=10)
        if col == n_targets - 1:
            fig.colorbar(im_v, ax=ax_vort, ticks=[-1, 0, 1], label='Winding No.', shrink=0.8)

        # --- Row 2: Correlation G(r) ---
        ax_gr = axes[2, col]
        r, G = snapshots_gr[col]
        
        # Plot using linear y-scale and log x-scale (Ising style)
        ax_gr.plot(r[1:], G[1:], linewidth=1.5)
        ax_gr.axhline(0, color='tab:gray', linewidth=0.7, linestyle='--')
        
        inv_e = 1.0 / np.e
        ax_gr.axhline(inv_e, color='tab:red', linewidth=0.8,
                      linestyle=':', alpha=0.7, label='$1/e$')
        
        ax_gr.set_xscale('log')
        ax_gr.set_xlabel('Distance r', fontsize=10)
        if col == 0:
            ax_gr.set_ylabel('$G(r)$ / $G(0)$', fontsize=10)
        ax_gr.grid(True, which='both', alpha=0.3)
        ax_gr.set_ylim(-0.1, 1.1)

        # Find xi where G(r) first drops below 1/e via linear interpolation
        r_plot = r[1:]
        G_plot = G[1:]
        below = np.where(G_plot < inv_e)[0]
        if len(below) > 0:
            idx = below[0]
            if idx > 0:
                r0, r1 = float(r_plot[idx - 1]), float(r_plot[idx])
                g0, g1 = float(G_plot[idx - 1]), float(G_plot[idx])
                xi = r0 + (inv_e - g0) * (r1 - r0) / (g1 - g0)
            else:
                xi = float(r_plot[idx])
            ax_gr.axvline(xi, color='tab:red', linewidth=1.0, linestyle='--', alpha=0.8)
            ax_gr.text(xi * 1.15, inv_e + 0.04,
                       f'$\\xi = {xi:.1f}$', fontsize=9, color='tab:red')

    output_dir = ensure_results_dir(args.output_dir)
    save_plot('domain_snapshots.png', directory=output_dir)


if __name__ == '__main__':
    main()
