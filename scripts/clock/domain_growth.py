"""
Domain growth analysis for the 2D q-state Clock model.

Quenches from a disordered state to T < T_c and records:
1. Characteristic domain size R(t) from the structure factor first moment.
2. Correlation length xi(t) from the G(r) decay.
3. Vortex density n_v(t) = (Number of vortices) / Area.

Predicts R(t) ~ t^(1/2) and n_v(t) ~ t^(-1).
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.clock_model import ClockSimulation
from utils.physics_helpers import radial_average_sk, pair_correlation_x
from utils.system_helpers import _BAR_FORMAT, ensure_results_dir, save_plot, setup_logging

# ---------------------------------------------------------------------------
# Physical analysis
# ---------------------------------------------------------------------------

def compute_domain_size(sim: ClockSimulation) -> float:
    """Extract characteristic domain size R from the structure factor S(k)."""
    N = sim.size
    kvals, S_radial = radial_average_sk(sim.spins)
    
    # R = 2π * Σ S(k) / Σ |k| S(k)
    # Skip DC mode (k=0)
    S_k = S_radial[1:]
    K_k = kvals[1:]
    
    denominator = float(np.sum(K_k * S_k))
    if denominator == 0.0:
        return 0.0
    return 2.0 * np.pi * float(np.sum(S_k) / denominator)


def compute_vortex_density(sim: ClockSimulation) -> float:
    """Calculate the number of vortices per unit area."""
    vorticity = sim._calculate_vorticity()
    total_vortices = np.sum(np.abs(vorticity))
    return float(total_vortices / (sim.size**2))


def compute_correlation_length(sim: ClockSimulation) -> float:
    """Estimate domain size as the correlation length xi from G(r)."""
    r_vals, G = pair_correlation_x(sim.spins)
    
    inv_e = 1.0 / np.e
    below = np.where(G < inv_e)[0]
    if len(below) == 0:
        return float(r_vals[-1])
    idx = below[0]
    if idx == 0:
        return float(r_vals[0])
    
    r0, r1 = float(r_vals[idx - 1]), float(r_vals[idx])
    g0, g1 = float(G[idx - 1]), float(G[idx])
    return r0 + (inv_e - g0) * (r1 - r0) / (g1 - g0)


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the domain growth simulation and save analysis plots."""
    parser = argparse.ArgumentParser(description='2D Clock Model Domain Growth Analysis')
    parser.add_argument('--size', type=int, default=256, help='Linear lattice size L')
    parser.add_argument('--temp', type=float, default=0.1, help='Quench temperature T')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    parser.add_argument('--aniso', type=float, default=0.5, help='Anisotropy strength A')
    parser.add_argument('--max-steps', type=int, default=1000, help='Total MC steps')
    parser.add_argument('--samples', type=int, default=15, help='Number of measurement points')
    parser.add_argument('--fit-min', type=int, default=20, help='Min step for power-law fit')
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
    MAX_STEPS = args.max_steps
    N_SAMPLES = args.samples
    FIT_MIN_STEP = args.fit_min

    logger.info(f"Clock domain growth analysis (L={L}, T={T:.3f}, q={Q}, A={A})")
    logger.info(f"Measuring R(t) and n_v(t) at {N_SAMPLES} log-spaced steps up to t={MAX_STEPS} ...")

    # Logarithmically-spaced step targets
    step_targets = np.unique(np.logspace(0, np.log10(MAX_STEPS), num=N_SAMPLES).astype(int))

    sim = ClockSimulation(size=L, temp=T, q=Q, A=A)

    N_data = len(step_targets)
    t = np.zeros(N_data)
    R_sk = np.zeros(N_data)
    R_xi = np.zeros(N_data)
    v_dens = np.zeros(N_data)

    current_step = 0
    for i, target in enumerate(tqdm(step_targets, bar_format=_BAR_FORMAT, desc='Simulating')):
        steps_to_run = int(target) - current_step
        for _ in range(steps_to_run):
            sim.step()
        current_step = int(target)

        t[i] = float(current_step)
        R_sk[i] = compute_domain_size(sim)
        R_xi[i] = compute_correlation_length(sim)
        v_dens[i] = compute_vortex_density(sim)
        
        logger.debug(f"t={current_step}: R_sk={R_sk[i]:.2f}, xi={R_xi[i]:.2f}, n_v={v_dens[i]:.4f}")

    # Power law fits
    fit_mask = t >= FIT_MIN_STEP
    
    def power_fit(t_arr, y_arr, mask):
        valid = mask & (y_arr > 0)
        if valid.sum() < 3: return None, None
        coeffs = np.polyfit(np.log(t_arr[valid]), np.log(y_arr[valid]), 1)
        return float(coeffs[0]), float(np.exp(coeffs[1]))

    exp_sk, pre_sk = power_fit(t, R_sk, fit_mask)
    exp_v, pre_v = power_fit(t, v_dens, fit_mask)

    if exp_sk: logger.info(f"Domain Size R(t) exponent: {exp_sk:.3f} (Allen-Cahn: 0.5)")
    if exp_v: logger.info(f"Vortex Density n_v(t) exponent: {exp_v:.3f} (Theory: -1.0)")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Domain Growth R(t)
    ax1.loglog(t, R_sk, 'o', label='$R_{S(k)}$ (Structure Factor)')
    ax1.loglog(t, R_xi, 's', label='$\\xi$ (Correlation length)')
    if exp_sk:
        ax1.loglog(t[fit_mask], pre_sk * t[fit_mask]**exp_sk, 'r--', label=f'Fit: $t^{{{exp_sk:.2f}}}$')
    ax1.set_xlabel('Time t (sweeps)')
    ax1.set_ylabel('Domain Size R(t)')
    ax1.set_title('Domain Coarsening')
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()

    # 2. Vortex Proliferation n_v(t)
    ax2.loglog(t, v_dens, 'D', color='tab:purple', label='$n_v(t)$ (Vortex Density)')
    if exp_v:
        ax2.loglog(t[fit_mask], pre_v * t[fit_mask]**exp_v, 'k--', label=f'Fit: $t^{{{exp_v:.2f}}}$')
    ax2.set_xlabel('Time t (sweeps)')
    ax2.set_ylabel('Vortex Density $n_v$')
    ax2.set_title('Vortex Decay')
    ax2.grid(True, which='both', alpha=0.3)
    ax2.legend()

    output_dir = ensure_results_dir(args.output_dir)
    save_plot('domain_growth.png', directory=output_dir)


if __name__ == '__main__':
    main()
