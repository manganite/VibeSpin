"""
Wolff cluster algorithm efficiency demonstration for the 2D Ising model.

Compares integrated autocorrelation time, independent samples per second,
mean cluster size fraction, and susceptibility between the Metropolis
checkerboard and Wolff cluster algorithms across a temperature range centred
on the critical point.

Results are saved to ``results/ising/wolff_efficiency.npz`` for notebook
re-use and ``results/ising/wolff_efficiency.png`` as a four-panel figure.
"""
from __future__ import annotations

import argparse

import numpy as np

from models.ising_model import IsingSimulation
from utils.efficiency_runner import add_wolff_efficiency_arguments, run_wolff_efficiency
from utils.system import parse_args_compat

#: Exact Onsager critical temperature for the 2D nearest-neighbour Ising model.
TC_ISING: float = 2.0 / np.log(1.0 + np.sqrt(2.0))


def main() -> None:
    """Execute the Ising efficiency comparison and save the figure and data."""
    parser = argparse.ArgumentParser(
        description='Wolff vs. Metropolis efficiency demo for the 2D Ising model.',
    )
    add_wolff_efficiency_arguments(
        parser=parser, size=64, t_min=1.8, t_max=3.2, output_dir='results/ising',
    )
    args = parse_args_compat(parser=parser)

    run_wolff_efficiency(
        args=args,
        model_cls=IsingSimulation,
        model_kwargs={},
        model_label='2D Ising Model',
        transitions={r'$T_c$': TC_ISING},
    )


if __name__ == '__main__':
    main()
