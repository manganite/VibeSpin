"""
Standardized temperature sweep for the 2D Clock model.
Calculates magnetization, energy, susceptibility, and specific heat, and
produces a thermodynamics figure plus a companion diagnostics figure.
"""
from __future__ import annotations

import argparse

from models.clock_model import ClockSimulation
from utils.sweep_runner import add_temperature_sweep_arguments, run_temperature_sweep
from utils.system import parse_args_compat

# Approximate crossover temperatures of the q=6 clock model, matching the
# phase boundaries used by scripts/clock/correlation_comparison.py.
_T1_CLOCK6_APPROX = 0.68
_T2_CLOCK6_APPROX = 0.92


def main() -> None:
    """
    Execute the temperature sweep and generate the thermodynamics and
    companion diagnostics figures.
    """
    parser = argparse.ArgumentParser(description='2D Clock Model Temperature Sweep')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    parser.add_argument('--aniso', type=float, default=0.0, help='Anisotropy parameter A')
    parser.add_argument('--discrete', action='store_true', help='Use discrete angle representation')
    add_temperature_sweep_arguments(
        parser=parser,
        size=48,
        t_min=0.1,
        t_max=2.0,
        t_points=60,
        # As for XY, the clock model needs a high cap for the convergence
        # criterion rather than the cap to end equilibration.
        eq_max_steps=200_000,
        meas_steps=20_000,
        output_dir='results/clock',
        transition_help=(
            'Transition overlay preset for plotting; auto (default) shows the '
            'approximate q=6 crossover temperatures (only for q=6), none '
            'disables the overlay'
        ),
    )
    args = parse_args_compat(parser=parser)

    variant = 'discrete' if args.discrete else f'continuous (A={args.aniso})'

    # Crossover overlay: the approximate T1/T2 values are specific to q=6, so
    # the overlay is skipped for other q even under 'auto'/'theory'.
    transitions = (
        {'T1 (approx)': _T1_CLOCK6_APPROX, 'T2 (approx)': _T2_CLOCK6_APPROX}
        if args.transition_preset in ('auto', 'theory') and args.q == 6
        else None
    )

    run_temperature_sweep(
        args=args,
        model_cls=ClockSimulation,
        model_kwargs={'q': args.q, 'A': args.aniso},
        model_label=f'{args.q}-state Clock',
        plot_title=f'{args.q}-state Clock Temperature Sweep ($L={args.size}$)',
        metadata_note=(
            f'L={args.size}, q={args.q}, {variant}, target_n_seeds={args.n_seeds}'
        ),
        transition_temperatures=transitions,
        variant_note=f'{variant}, ',
        # As for XY, no ordered-start preference: the intermediate phase of the
        # clock model is quasi-ordered rather than ordered.
    )


if __name__ == '__main__':
    main()
