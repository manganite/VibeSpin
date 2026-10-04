"""
Shared command-line handling for choosing the clock-model representation.

The clock scripts default to ``DiscreteClockSimulation``, which restricts every
spin to the q allowed orientations and is the model the literature values of
the two BKT temperatures refer to. ``--continuous`` selects ``ClockSimulation``,
planar spins with the crystal-field term ``-A cos(q theta)``, whose transition
temperatures depend on ``A`` and approach the discrete values only as A grows.
"""
from __future__ import annotations

import argparse
from typing import Any, NamedTuple

from models.clock_model import ClockSimulation, DiscreteClockSimulation


class ClockModelChoice(NamedTuple):
    """Resolved clock model class, constructor arguments, and a short label.

    Attributes
    ----------
    model_cls : type
        ``DiscreteClockSimulation`` or ``ClockSimulation``.
    model_kwargs : dict[str, Any]
        Constructor keyword arguments beyond size, temperature, and seed.
    variant : str
        Human-readable description used in log lines, titles, and metadata.
    is_discrete : bool
        Whether the discrete representation was selected.
    """

    model_cls: type
    model_kwargs: dict[str, Any]
    variant: str
    is_discrete: bool


def add_clock_model_arguments(
    *, parser: argparse.ArgumentParser, default_aniso: float
) -> None:
    """Register ``--discrete``, ``--continuous``, and ``--aniso`` on a parser.

    The script itself must already define ``--q``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to extend.
    default_aniso : float
        Anisotropy used by the continuous model when ``--aniso`` is omitted.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--discrete', action='store_true',
        help='Discrete clock model with integer states (default)',
    )
    group.add_argument(
        '--continuous', action='store_true',
        help='Continuous planar spins with anisotropy -A cos(q theta) instead of '
             'the discrete model',
    )
    parser.add_argument(
        '--aniso', type=float, default=None,
        help=f'Anisotropy A of the continuous model (default {default_aniso}); '
             'requires --continuous',
    )


def resolve_clock_model(
    *, parser: argparse.ArgumentParser, args: argparse.Namespace, default_aniso: float
) -> ClockModelChoice:
    """Translate parsed clock-model flags into a model class and its arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser used to report invalid flag combinations.
    args : argparse.Namespace
        Parsed arguments containing ``q``, ``continuous``, and ``aniso``.
    default_aniso : float
        Anisotropy used by the continuous model when ``--aniso`` is omitted.

    Returns
    -------
    ClockModelChoice
        The resolved model selection.
    """
    if args.continuous:
        aniso = default_aniso if args.aniso is None else float(args.aniso)
        return ClockModelChoice(
            model_cls=ClockSimulation,
            model_kwargs={'q': args.q, 'A': aniso},
            variant=f'continuous (A={aniso})',
            is_discrete=False,
        )
    if args.aniso is not None:
        # parser.error exits; the discrete model has no anisotropy parameter.
        parser.error('--aniso applies only to the continuous model; add --continuous')
    return ClockModelChoice(
        model_cls=DiscreteClockSimulation,
        model_kwargs={'q': args.q},
        variant='discrete',
        is_discrete=True,
    )
