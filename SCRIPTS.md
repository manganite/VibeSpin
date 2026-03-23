# Scripts Catalog: VibeSpin

VibeSpin provides a suite of entry-point scripts for conducting physics experiments. These are organized by model family in the `scripts/` directory.

## 1. Ising Model (`scripts/ising/`)

- **`temperature_sweep.py`**: Conducts a full thermodynamic sweep across a range of temperatures. Reports $|M|$, $E$, $\chi$, $C_v$, entropy $S(T)$ from $C_v/T$ integration, and integrated autocorrelation time $\tau_{\mathrm{int}}$. Supports `--n-seeds` for seeded ensembles and uncertainty controls including `--confidence-level`, `--derived-uncertainty-method`, `--derived-bootstrap-resamples`, `--entropy-uncertainty-method`, and `--entropy-bootstrap-resamples`. Convergence diagnostics can be tuned via `--eq-qs-sigma-threshold` and `--eq-qs-min-steps` to detect and stop early on low-temperature stuck states. Strict quality control is available via `--strict-uncertainty` with threshold controls for undefined and unstable points. For `n_seeds=1`, primary and derived observables use single-trajectory blocking, while entropy uncertainty falls back to propagation from the specific-heat uncertainty. For `n_seeds>1`, uncertainty combines between-seed spread and within-seed blocking error. Saves a standardized NPZ file `results/ising/temperature_sweep_data.npz` containing both legacy scalar arrays and per-observable uncertainty schema keys (`<obs>_value`, `<obs>_err`, `<obs>_ci_low`, `<obs>_ci_high`, `<obs>_tau_int`, `<obs>_n_eff`, `<obs>_samples`) plus additive extras (`entropy_*`, `tau_int_ci_*`, quality flags). The plotting output is split into a thermodynamics summary figure and a companion diagnostics figure, where entropy and $\tau_{\mathrm{int}}$ intervals are shown as ribbons and unavailable uncertainty is marked explicitly.
- **`ordering_kinetics.py`**: Quenches the system to $T < T_c$ and tracks the growth of domain size $R(t)$ over time.
- **`ordering_evolution.py`**: Generates visual snapshots of the lattice configuration, structure factor, and correlation functions during a quench.
- **`correlation_divergence.py`**: Extracts the critical exponent $\nu$ by fitting the correlation length divergence near $T_c$.
- **`correlation_comparison.py`**: Compares the functional form of $G(r)$ in the ferromagnetic, critical, and paramagnetic phases.
- **`wolff_efficiency.py`**: Compares the Metropolis checkerboard and Wolff cluster algorithms across the critical regime. Reports integrated autocorrelation time $\tau_{\mathrm{int}}$, independent samples per second (ISS), mean cluster size fraction $\langle C \rangle/N^2$, and susceptibility $\chi(T)$ for both algorithms. Saves results to `results/ising/wolff_efficiency.npz` for re-use by `notebooks/Wolff_Efficiency.ipynb` and a 4-panel summary figure to `results/ising/wolff_efficiency.png`.
- **`measure_z.py`**: Specifically measures the dynamical critical exponent $z$ at the critical temperature $T_c$. Sweeps lattice sizes $L$ to extract the scaling law $\tau_{\mathrm{int}} \propto L^z$ for both Metropolis and Wolff algorithms. Runs `--n-seeds` independent replicas per (algorithm, $L$) point and saves per-seed sample arrays alongside median and 16–84% percentile summaries to `results/ising/dynamic_exponent_z.npz` for use in `notebooks/Dynamic_Critical_Exponents.ipynb`. The NPZ also includes standardized uncertainty keys (`tau_metro_value`, `tau_metro_err`, `tau_metro_ci_low`, `tau_metro_ci_high`, and Wolff equivalents).

## 2. XY Model (`scripts/xy/`)

- **`temperature_sweep.py`**: Standard thermodynamic sweep for continuous vector spins, including $|M|$, $E$, $\chi$, $C_v$, entropy $S(T)$, and integrated autocorrelation time $\tau_{\mathrm{int}}$. Supports `--n-seeds` with the same single-seed blocking, entropy fallback propagation, multi-seed hierarchical uncertainty behavior, and uncertainty-control CLI options as the Ising sweep, plus per-temperature quality flags. Saves `results/xy/temperature_sweep_data.npz` with the same standardized uncertainty schema and additive extras as the Ising sweep, along with separate thermodynamics and diagnostics figures for clearer uncertainty visualization.
- **`ordering_kinetics.py`**: Quenches to $T < T_{BKT}$ and tracks the decay of vortex density and the growth of the correlation length.
- **`ordering_evolution.py`**: Visual snapshots including phase maps and vorticity configurations during ordering.
- **`bkt_transition.py`**: Specifically focuses on the BKT transition by measuring average vortex density vs. temperature.
- **`helicity_modulus.py`**: Calculates the superfluid stiffness to identify the universal jump at $T_{BKT}$.
- **`compare_correlations.py`**: Contrasts power-law decay (topological order) with exponential decay (disordered phase).

## 3. Clock Model (`scripts/clock/`)

- **`temperature_sweep.py`**: Thermodynamic sweep for the q-state clock model, including $|M|$, $E$, $\chi$, $C_v$, entropy $S(T)$ (with optional $S_{\mathrm{ref}}=\ln q$ normalization), and integrated autocorrelation time $\tau_{\mathrm{int}}$. Supports `--n-seeds` with the same single-seed blocking, entropy fallback propagation, multi-seed hierarchical uncertainty behavior, and uncertainty-control CLI options as the Ising sweep, plus per-temperature quality flags. Saves `results/clock/temperature_sweep_data.npz` with the same standardized uncertainty schema and additive extras as the Ising sweep, along with separate thermodynamics and diagnostics figures for clearer uncertainty visualization.
- **`ordering_kinetics.py`**: Analyzes the ordering dynamics after a quench.
- **`ordering_evolution.py`**: Visualizes the evolution of discrete phase domains.
- **`compare_discrete_vs_continuous.py`**: Provides a side-by-side performance and physical comparison between the continuous (XY + anisotropy) and discrete implementations.

## 4. Cross-Model Benchmarking (`scripts/benchmarks/`)

- **`throughput.py`**: Cross-model throughput and scaling benchmark. Measures sweeps/s, ns/site, and per-call analysis costs (thermodynamic, $G(r)$, vorticity, helicity) across all eight model variants and a range of lattice sizes. Saves a 6-panel summary figure to `results/benchmarks/scaling_benchmark.png` and all metrics to `results/benchmarks/scaling_benchmark.npz`. The NPZ file is loaded by `notebooks/Performance_Benchmarks.ipynb` to avoid re-running the benchmark on every notebook execution.

## Usage Guidelines

### Update Schemes
As mandated in `AGENTS.md`:
- Use **`--update random`** (or default in kinetics scripts) for any time-dependent study.
- Use **`--update checkerboard`** (default in sweep scripts) for equilibrium measurements.
- Use **`--update wolff`** when sweeping temperatures within roughly 20% of $T_c$. The Wolff cluster algorithm reduces $\tau_{\mathrm{int}}$ by an order of magnitude near criticality, improving the ISS rate proportionally. See `notebooks/Wolff_Efficiency.ipynb` for a quantitative demonstration.

### Results
All scripts save their output (plots and data files) to the `results/` directory, sub-divided by model and experiment type.
