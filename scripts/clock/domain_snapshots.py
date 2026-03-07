"""
Domain snapshot visualisation for the 2D q-state Clock model.

Quenches from a disordered state to T < T_c and records the spin configuration
at multiple time steps, plotting phase configurations, vorticity maps, and
radially averaged correlation functions G(r).
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from models.clock_model import ClockSimulation
from utils.system_helpers import ensure_results_dir, save_plot, setup_logging


def main() -> None:
    """Run the snapshot simulation and generate a multi-row evolution figure."""
    parser = argparse.ArgumentParser(description='2D Clock Model Domain Snapshot Visualisation')
    parser.add_argument('--size', type=int, default=128, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.2, help='Quench temperature T')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    parser.add_argument('--aniso', type=float, default=0.5, help='Anisotropy strength A')
    parser.add_argument('--targets', type=int, nargs='+', default=[1, 10, 100, 1000],
                        help='MC steps at which to take snapshots')
    parser.add_argument('--output-dir', type=str, default='results/clock', help='Output directory')
    parser.add_argument('--log-file', type=str, default=None, help='Optional log file path')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_arguments() if hasattr(parser, 'parse_arguments') else parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(level=log_level, log_file=args.log_file)

    L = args.size
    T = args.temp
    Q = args.q
    A = args.aniso
    STEP_TARGETS = sorted(args.targets)

    logger.info(f"Clock domain snapshots (L={L}, T={T}, q={Q}, A={A})")
    logger.info(f"Recording snapshots at steps {STEP_TARGETS} ...")

    sim = ClockSimulation(size=L, temp=T, q=Q, A=A)
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
                             figsize=(n_cols * 4, 12),
                             gridspec_kw={'hspace': 0.35, 'wspace': 0.25})
    fig.suptitle(
        f'2D {Q}-state Clock Model Evolution — T = {T}, L = {L}, A = {A}',
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
        ax_gr.plot(r[1:], G[1:], linewidth=1.5)
        ax_gr.set_xscale('log')
        ax_gr.set_yscale('log')
        ax_gr.set_xlabel('r', fontsize=10)
        if col == 0:
            ax_gr.set_ylabel('G(r)', fontsize=10)
        ax_gr.grid(True, which='both', alpha=0.3)
        ax_gr.set_ylim(1e-2, 1.1)

    output_dir = ensure_results_dir(args.output_dir)
    save_plot('domain_snapshots.png', directory=output_dir)


if __name__ == '__main__':
    main()
