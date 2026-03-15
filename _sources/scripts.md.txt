# Scripts Catalog: VibeSpin

VibeSpin provides a suite of entry-point scripts for conducting physics experiments. These are organized by model family in the `scripts/` directory.

## 1. Ising Model (`scripts/ising/`)

- **`temperature_sweep.py`**: Conducts a full thermodynamic sweep across a range of temperatures. Plots $|M|$, $E$, $\chi$, and $C_v$.
- **`ordering_kinetics.py`**: Quenches the system to $T < T_c$ and tracks the growth of domain size $R(t)$ over time.
- **`ordering_evolution.py`**: Generates visual snapshots of the lattice configuration, structure factor, and correlation functions during a quench.
- **`correlation_divergence.py`**: Extracts the critical exponent $\nu$ by fitting the correlation length divergence near $T_c$.
- **`correlation_comparison.py`**: Compares the functional form of $G(r)$ in the ferromagnetic, critical, and paramagnetic phases.

## 2. XY Model (`scripts/xy/`)

- **`temperature_sweep.py`**: Standard thermodynamic sweep for continuous vector spins.
- **`ordering_kinetics.py`**: Quenches to $T < T_{BKT}$ and tracks the decay of vortex density and the growth of the correlation length.
- **`ordering_evolution.py`**: Visual snapshots including phase maps and vorticity configurations during ordering.
- **`bkt_transition.py`**: Specifically focuses on the BKT transition by measuring average vortex density vs. temperature.
- **`helicity_modulus.py`**: Calculates the superfluid stiffness to identify the universal jump at $T_{BKT}$.
- **`compare_correlations.py`**: Contrasts power-law decay (topological order) with exponential decay (disordered phase).

## 3. Clock Model (`scripts/clock/`)

- **`temperature_sweep.py`**: Thermodynamic sweep for the q-state clock model.
- **`ordering_kinetics.py`**: Analyzes the ordering dynamics after a quench.
- **`ordering_evolution.py`**: Visualizes the evolution of discrete phase domains.
- **`compare_discrete_vs_continuous.py`**: Provides a side-by-side performance and physical comparison between the continuous (XY + anisotropy) and discrete implementations.

## Usage Guidelines

### Update Schemes
As mandated in `AGENTS.md`:
- Use **`--update random`** (or default in kinetics scripts) for any time-dependent study.
- Use **`--update checkerboard`** (default in sweep scripts) for equilibrium measurements.

### Results
All scripts save their output (plots and data files) to the `results/` directory, sub-divided by model and experiment type.
