"""
Wolff cluster algorithm efficiency demonstration for the 2D clock model.

The clock counterpart of the Ising and XY comparisons.  The temperature window
brackets both crossovers of the q=6 model, the lower one out of the ordered
phase and the upper one out of the quasi-ordered phase into the disordered
one.

The comparison runs on the discrete clock model. Its Wolff update reflects
clusters about the q mirror axes of the clock, which maps allowed states onto
allowed states and satisfies detailed balance for the full discrete
Hamiltonian, so both algorithms sample the same model.  The continuous
``ClockSimulation`` cannot serve here: its cluster update ignores the
anisotropy and is only valid at A = 0, where the model is plain XY.

Results are saved to ``results/clock/wolff_efficiency.npz`` for notebook re-use
and ``results/clock/wolff_efficiency.png`` as a four-panel figure.
"""
from __future__ import annotations

import argparse

from models.clock_model import DiscreteClockSimulation
from utils.efficiency_runner import add_wolff_efficiency_arguments, run_wolff_efficiency
from utils.system import parse_args_compat

# Approximate crossover temperatures of the q=6 clock model, matching the
# phase boundaries used by scripts/clock/correlation_comparison.py.
_T1_CLOCK6_APPROX = 0.68
_T2_CLOCK6_APPROX = 0.92


def main() -> None:
    """Execute the clock efficiency comparison and save the figure and data."""
    parser = argparse.ArgumentParser(
        description='Wolff vs. Metropolis efficiency demo for the 2D clock model.',
    )
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    add_wolff_efficiency_arguments(
        parser=parser, size=64, t_min=0.3, t_max=1.4, output_dir='results/clock',
    )
    args = parse_args_compat(parser=parser)

    # The approximate crossover temperatures are specific to q=6.
    transitions = (
        {'$T_1$ (approx)': _T1_CLOCK6_APPROX, '$T_2$ (approx)': _T2_CLOCK6_APPROX}
        if args.q == 6
        else None
    )

    run_wolff_efficiency(
        args=args,
        model_cls=DiscreteClockSimulation,
        model_kwargs={'q': args.q},
        model_label=f'{args.q}-state Clock Model',
        transitions=transitions,
    )


if __name__ == '__main__':
    main()
