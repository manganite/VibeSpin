"""
Wolff cluster algorithm efficiency demonstration for the 2D XY model.

The XY counterpart of the Ising comparison, using the Wolff-Evertz reflection
rather than a Z2 cluster flip.  The temperature window brackets the BKT
transition, where the correlation length diverges exponentially rather than as
a power law, so critical slowing down of the local update sets in more gently
than it does at a second-order critical point.

Results are saved to ``results/xy/wolff_efficiency.npz`` for notebook re-use
and ``results/xy/wolff_efficiency.png`` as a four-panel figure.
"""
from __future__ import annotations

import argparse

from models.xy_model import XYSimulation
from utils.efficiency_runner import add_wolff_efficiency_arguments, run_wolff_efficiency
from utils.system import parse_args_compat

#: Accepted BKT transition temperature of the 2D XY model.
TBKT_XY: float = 0.893


def main() -> None:
    """Execute the XY efficiency comparison and save the figure and data."""
    parser = argparse.ArgumentParser(
        description='Wolff vs. Metropolis efficiency demo for the 2D XY model.',
    )
    add_wolff_efficiency_arguments(
        parser=parser, size=64, t_min=0.4, t_max=1.5, output_dir='results/xy',
    )
    args = parse_args_compat(parser=parser)

    run_wolff_efficiency(
        args=args,
        model_cls=XYSimulation,
        model_kwargs={},
        model_label='2D XY Model',
        transitions={r'$T_{\mathrm{BKT}}$': TBKT_XY},
    )


if __name__ == '__main__':
    main()
