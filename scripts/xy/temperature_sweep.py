"""
Standardized temperature sweep for the 2D XY model.
Calculates magnetization, energy, susceptibility, and specific heat, and
produces a thermodynamics figure plus a companion diagnostics figure.
"""
from __future__ import annotations

import argparse

from models.xy_model import XYSimulation
from utils.sweep_runner import add_temperature_sweep_arguments, run_temperature_sweep
from utils.system import parse_args_compat

_TBKT_XY_THEORY = 0.893


def main() -> None:
    """
    Execute the temperature sweep and generate the thermodynamics and
    companion diagnostics figures.
    """
    parser = argparse.ArgumentParser(description='2D XY Model Temperature Sweep')
    add_temperature_sweep_arguments(
        parser=parser,
        size=64,
        t_min=0.1,
        t_max=2.0,
        t_points=60,
        # The XY model relaxes far more slowly than Ising, so the cap has to be
        # an order of magnitude higher for the convergence criterion to decide
        # rather than the cap.
        eq_max_steps=200_000,
        meas_steps=20_000,
        output_dir='results/xy',
        transition_help=(
            'Transition overlay preset for plotting; auto (default) shows the '
            'known theoretical transition, none disables the overlay'
        ),
    )
    args = parse_args_compat(parser=parser)

    # Transition overlay: 'auto' and 'theory' enable it, 'none' disables it.
    transitions = (
        {'Theory (BKT)': _TBKT_XY_THEORY}
        if args.transition_preset in ('auto', 'theory')
        else None
    )

    run_temperature_sweep(
        args=args,
        model_cls=XYSimulation,
        model_kwargs={},
        model_label='XY',
        plot_title=f'XY Model Temperature Sweep ($L={args.size}$)',
        metadata_note=(
            f'L={args.size}, target_n_seeds={args.n_seeds}, meas_steps={args.meas_steps}'
        ),
        transition_temperatures=transitions,
        # No ordered-start preference: below T_BKT the XY model has only
        # quasi-long-range order, so an ordered start is not the state the
        # sweep is trying to reach.
    )


if __name__ == '__main__':
    main()
