"""
Standardized temperature sweep for the 2D Clock model.
Calculates magnetization, energy, susceptibility, and specific heat, and
produces a thermodynamics figure plus a companion diagnostics figure.

The sweep runs the discrete clock model by default; ``--continuous`` selects
planar spins with the anisotropy ``-A cos(q theta)`` instead.
"""
from __future__ import annotations

import argparse
import math

from scripts.clock._model_choice import add_clock_model_arguments, resolve_clock_model
from utils.sweep_runner import add_temperature_sweep_arguments, run_temperature_sweep
from utils.system import parse_args_compat

# Approximate crossover temperatures of the q=6 clock model, matching the
# phase boundaries used by scripts/clock/correlation_comparison.py.
_T1_CLOCK6_APPROX = 0.68
_T2_CLOCK6_APPROX = 0.92

# Anisotropy of the continuous model when --continuous is given without
# --aniso; it matches the ClockSimulation constructor default.
_DEFAULT_CONTINUOUS_ANISO = 1.0

# Temperature below which the discrete q-state clock model has true long-range
# order. For q = 2, 3 and 4 these are exact: q = 2 is the Ising model, q = 3
# maps onto the 3-state Potts model with coupling 3J/2, and q = 4 decouples
# into two Ising models with coupling J/2. For q = 6 the value is the
# approximate lower BKT temperature T1. Other q have no entry, so the sweep
# runs without the ordered-start fallback for them.
_ORDERED_BELOW_DISCRETE: dict[int, float] = {
    2: 2.0 / math.log(1.0 + math.sqrt(2.0)),
    3: 1.5 / math.log(1.0 + math.sqrt(3.0)),
    4: 1.0 / math.log(1.0 + math.sqrt(2.0)),
    6: _T1_CLOCK6_APPROX,
}


def main() -> None:
    """
    Execute the temperature sweep and generate the thermodynamics and
    companion diagnostics figures.
    """
    parser = argparse.ArgumentParser(description='2D Clock Model Temperature Sweep')
    parser.add_argument('--q', type=int, default=6, help='Number of clock states')
    add_clock_model_arguments(parser=parser, default_aniso=_DEFAULT_CONTINUOUS_ANISO)
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

    choice = resolve_clock_model(
        parser=parser, args=args, default_aniso=_DEFAULT_CONTINUOUS_ANISO,
    )
    variant = choice.variant

    # Crossover overlay: the approximate T1/T2 values belong to the discrete
    # q=6 model, so the overlay is skipped for other q and for the continuous
    # model, whose transition temperatures depend on A.
    transitions = (
        {'T1 (approx)': _T1_CLOCK6_APPROX, 'T2 (approx)': _T2_CLOCK6_APPROX}
        if args.transition_preset in ('auto', 'theory') and args.q == 6
        and choice.is_discrete
        else None
    )

    run_temperature_sweep(
        args=args,
        model_cls=choice.model_cls,
        model_kwargs=choice.model_kwargs,
        model_label=f'{args.q}-state Clock',
        plot_title=f'{args.q}-state Clock Temperature Sweep ($L={args.size}$)',
        metadata_note=(
            f'L={args.size}, q={args.q}, {variant}, target_n_seeds={args.n_seeds}'
        ),
        transition_temperatures=transitions,
        variant_note=f'{variant}, ',
        # The discrete model orders like Ising below its lowest transition, and
        # random starts strand in domain-wall states there, so the ordered start
        # replaces a stuck random start. The continuous model gets no such
        # preference: its transition temperatures depend on A, and between T1
        # and T2 the clock model is only quasi-ordered.
        ordered_start_below=(
            _ORDERED_BELOW_DISCRETE.get(args.q) if choice.is_discrete else None
        ),
    )


if __name__ == '__main__':
    main()
