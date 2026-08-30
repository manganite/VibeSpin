"""
Standardized temperature sweep for the 2D Ising model.
Calculates magnetization, energy, susceptibility, and specific heat, and
produces a thermodynamics figure plus a companion diagnostics figure.
"""
from __future__ import annotations

import argparse

from models.ising_model import IsingSimulation
from utils.sweep_runner import add_temperature_sweep_arguments, run_temperature_sweep
from utils.system import parse_args_compat

_TC_ISING_THEORY = 2.26918531421


def main() -> None:
    """
    Execute the temperature sweep and generate the thermodynamics and
    companion diagnostics figures.
    """
    parser = argparse.ArgumentParser(description='2D Ising Model Temperature Sweep')
    add_temperature_sweep_arguments(
        parser=parser,
        size=64,
        t_min=0.1,
        t_max=4.0,
        t_points=60,
        eq_max_steps=20_000,
        meas_steps=20_000,
        output_dir='results/ising',
        transition_help=(
            'Transition overlay preset for plotting; auto (default) shows the '
            'known theoretical transition, none disables the overlay'
        ),
    )
    args = parse_args_compat(parser=parser)

    # Transition overlay: 'auto' and 'theory' enable it, 'none' disables it.
    transitions = (
        {'Theory': _TC_ISING_THEORY}
        if args.transition_preset in ('auto', 'theory')
        else None
    )

    run_temperature_sweep(
        args=args,
        model_cls=IsingSimulation,
        model_kwargs={},
        model_label='Ising',
        plot_title=f'Ising Model Temperature Sweep ($L={args.size}$)',
        metadata_note=(
            f'L={args.size}, target_n_seeds={args.n_seeds}, '
            f'meas_steps={args.meas_steps}, confidence={args.confidence_level}'
        ),
        transition_temperatures=transitions,
        # Below T_c the Ising model has true long-range order, so a random
        # start that the stuck detector strands in a domain-wall state is
        # replaced by the ordered start rather than discarded.
        ordered_start_below=_TC_ISING_THEORY,
    )


if __name__ == '__main__':
    main()
