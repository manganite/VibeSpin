# VibeSpin

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://manganite.github.io/VibeSpin/)
[![Tests](https://github.com/manganite/VibeSpin/actions/workflows/tests.yml/badge.svg)](https://github.com/manganite/VibeSpin/actions/workflows/tests.yml)

VibeSpin is a Python framework for high-performance simulation and analysis of two-dimensional lattice spin models. The codebase focuses on three foundational systems: the **Ising model**, the **XY model**, and the **q-state Clock model** — provided as `ClockSimulation` (continuous XY-plus-anisotropy form) and `DiscreteClockSimulation` (integer state indices with cosine lookup tables). It combines Numba-accelerated Monte Carlo dynamics with a robust analysis suite for equilibrium observables, coarsening kinetics, and topological defect tracking.

The implementation is optimized for speed, scalability, and physical repeatability. Core kernels utilize **Numba JIT compilation** with optional **multi-core parallelization**, periodic boundaries are handled via precomputed index arrays, and all stochastic trajectories are fully deterministic when seeded.

## Project Policies & Scope Discipline

VibeSpin enforces strict development policies for physical correctness, code quality, reproducibility, and documentation. All edits must follow the **Scope Discipline**: never change, rewrite, or delete code/text unrelated to the current task. See [AGENTS.md](./agents.md) for the full list of mandatory development policies.

## Bibliography

[1] L. Onsager, "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition," *Physical Review*, vol. 65, no. 3-4, pp. 117–149, 1944. [APS Open Access](https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117)

[2] J. M. Kosterlitz and D. J. Thouless, "Ordering, metastability and phase transitions in two-dimensional systems," *Journal of Physics C: Solid State Physics*, vol. 6, no. 7, pp. 1181–1203, 1973. [IOP Open Access](https://iopscience.iop.org/article/10.1088/0022-3719/6/7/010)

[3] J. Villain, "Theory of one- and two-dimensional magnets with an easy magnetization plane. II. The planar, classical, two-dimensional magnet," *J. Phys. France* 36, 581-590 (1975). [Open Access](https://doi.org/10.1051/jphys:01975003606058100)

[4] W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications," *Biometrika*, vol. 57, no. 1, pp. 97–109, 1970. [Oxford Academic](https://academic.oup.com/biomet/article/57/1/97/252073)

[5] U. Wolff, "Collective Monte Carlo Updating for Spin Systems," *Physical Review Letters*, vol. 62, no. 4, pp. 361–364, 1989. [APS Open Access](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.62.361)

