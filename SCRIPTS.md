# Scripts Catalog: VibeSpin

VibeSpin provides a suite of entry-point scripts for conducting physics experiments. These are organized by model family in the `scripts/` directory.

## 1. Ising Model (`scripts/ising/`)

- **`temperature_sweep.py`**: Conducts a full thermodynamic sweep across a range of temperatures. Reports $|M|$, $E$, $\chi$, $C_v$, entropy $S(T)$ from $C_v/T$ integration, and integrated autocorrelation time $\tau_{\mathrm{int}}$. Supports `--n-seeds` for seeded ensembles and uncertainty controls including `--confidence-level`, `--derived-uncertainty-method`, `--derived-bootstrap-resamples`, `--entropy-uncertainty-method`, and `--entropy-bootstrap-resamples`. Convergence diagnostics can be tuned via `--eq-qs-sigma-threshold` and `--eq-qs-min-steps` to detect and stop early on low-temperature stuck states. Strict quality control is available via `--strict-uncertainty` with threshold controls for undefined and unstable points. For `n_seeds=1`, primary and derived observables use single-trajectory blocking, while entropy uncertainty falls back to propagation from the specific-heat uncertainty. For `n_seeds>1`, uncertainty combines between-seed spread and within-seed blocking error. Saves a standardized NPZ file `results/ising/temperature_sweep_data.npz` containing both legacy scalar arrays and per-observable uncertainty schema keys (`<obs>_value`, `<obs>_err`, `<obs>_ci_low`, `<obs>_ci_high`, `<obs>_tau_int`, `<obs>_n_eff`, `<obs>_samples`) plus additive extras (`entropy_*` and per-temperature quality flags). The plotting output is split into a thermodynamics summary figure and a companion diagnostics figure, where entropy and $\tau_{\mathrm{int}}$ intervals are shown as ribbons and unavailable uncertainty is marked explicitly.
- **`diag_eq_traces.py`**: Generates a 4x6 diagnostic visualization of two-start equilibration traces (random vs. ordered start) covering 6 representative seeds across 4 temperatures. Pre-screens seeds at low temperature to identify and highlight trajectories that become stuck in metastable states (marked with a star). The script also produces a detailed 2x3 panel for $T=2.00$ showing mutual cross-bands and automatic convergence markers for each seed. Results are saved as PNG figures in the output directory.
- **`ordering_kinetics.py`**: Quenches the system to $T < T_c$ and tracks the growth of domain size $R(t)$ over time. Supports `--seeds N` and `--base-seed` for multi-seed ensemble averaging with IQR-based error bars. Saves per-seed arrays and median trajectory to `results/ising/ordering_kinetics.npz`.
- **`ordering_evolution.py`**: Generates visual snapshots of the lattice configuration, structure factor, and correlation functions during a quench.
- **`correlation_divergence.py`**: Extracts the critical exponent $\nu$ by fitting the correlation length divergence near $T_c$.
- **`correlation_comparison.py`**: Compares the functional form of $G(r)$ in the ferromagnetic, critical, and paramagnetic phases. Uses two-start convergence equilibration (`--eq-probe`, `--eq-max`) with a `--seed` flag for reproducible runs, matching the XY and Clock correlation scripts. Saves `results/ising/correlation_comparison.npz`.
- **`coarsening_analysis.py`**: Precomputes supplementary coarsening analyses for the `Correlation_and_Coarsening.ipynb` notebook: quench-depth sensitivity (3 temperatures, multi-seed), equilibrium-to-coarsening crossover ($\xi_{\mathrm{eq}}$ measurement plus coarsening traces), and a stochastic ensemble (8 seeds) visualizing run-to-run variability. Saves all data to `results/ising/coarsening_analysis.npz`.
- **`wolff_efficiency.py`**: Compares the Metropolis checkerboard and Wolff cluster algorithms across the critical regime. Reports integrated autocorrelation time $\tau_{\mathrm{int}}$, independent samples per second (ISS), mean cluster size fraction $\langle C \rangle/N^2$, and susceptibility $\chi(T)$ for both algorithms. Saves results to `results/ising/wolff_efficiency.npz` for re-use by `notebooks/Wolff_Efficiency.ipynb` and a 4-panel summary figure to `results/ising/wolff_efficiency.png`.
- **`measure_z.py`**: Specifically measures the dynamical critical exponent $z$ at the critical temperature $T_c$. Sweeps lattice sizes $L$ to extract the scaling law $\tau_{\mathrm{int}} \propto L^z$ for both Metropolis and Wolff algorithms. Runs `--n-seeds` independent replicas per (algorithm, $L$) point and saves per-seed sample arrays alongside median and 16–84% percentile summaries to `results/ising/dynamic_exponent_z.npz` for use in `notebooks/Dynamic_Critical_Exponents.ipynb`. The NPZ also includes standardized uncertainty keys (`tau_metro_value`, `tau_metro_err`, `tau_metro_ci_low`, `tau_metro_ci_high`, and Wolff equivalents).

## 2. XY Model (`scripts/xy/`)

- **`temperature_sweep.py`**: Standard thermodynamic sweep for continuous vector spins, including $|M|$, $E$, $\chi$, $C_v$, entropy $S(T)$, and integrated autocorrelation time $\tau_{\mathrm{int}}$. Supports `--n-seeds` with the same single-seed blocking, entropy fallback propagation, multi-seed hierarchical uncertainty behavior, uncertainty-control CLI options, and `--eq-qs-*` stuck-detection flags as the Ising sweep, plus per-temperature quality flags. Saves `results/xy/temperature_sweep_data.npz` with the same standardized uncertainty schema and additive extras as the Ising sweep, along with separate thermodynamics and diagnostics figures for clearer uncertainty visualization.
- **`ordering_kinetics.py`**: Quenches to $T < T_{BKT}$ and tracks the decay of vortex density and the growth of the correlation length. Supports `--seeds N` and `--base-seed` for multi-seed ensemble averaging with IQR-based error bars.
- **`ordering_evolution.py`**: Visual snapshots including phase maps and vorticity configurations during ordering.
- **`bkt_transition.py`**: Specifically focuses on the BKT transition by measuring average vortex density vs. temperature.
- **`helicity_modulus.py`**: Calculates the superfluid stiffness to identify the universal jump at $T_{BKT}$.
- **`correlation_comparison.py`**: Contrasts power-law decay (topological order) with exponential decay (disordered phase). Renamed from `compare_correlations.py` to match its Ising and Clock siblings.

## 3. Clock Model (`scripts/clock/`)

- **`temperature_sweep.py`**: Thermodynamic sweep for the q-state clock model, including $|M|$, $E$, $\chi$, $C_v$, entropy $S(T)$, a `--transition-preset` overlay marking the approximate $q=6$ crossover temperatures, and integrated autocorrelation time $\tau_{\mathrm{int}}$. Supports `--n-seeds` with the same single-seed blocking, entropy fallback propagation, multi-seed hierarchical uncertainty behavior, uncertainty-control CLI options, and `--eq-qs-*` stuck-detection flags as the Ising sweep, plus per-temperature quality flags. Saves `results/clock/temperature_sweep_data.npz` with the same standardized uncertainty schema and additive extras as the Ising sweep, along with separate thermodynamics and diagnostics figures for clearer uncertainty visualization.
- **`ordering_kinetics.py`**: Analyzes the ordering dynamics after a quench. Supports `--seeds N` and `--base-seed` for multi-seed ensemble averaging with IQR-based error bars.
- **`ordering_evolution.py`**: Visualizes the evolution of discrete phase domains.
- **`correlation_comparison.py`**: Compares the functional form of $G(r)$ in the ordered ($T < T_1$), quasi-ordered ($T_1 < T < T_2$), and disordered ($T > T_2$) phases of the $q = 6$ clock model. Uses convergence equilibration. Saves `results/clock/correlation_comparison.npz`.
- **`compare_discrete_vs_continuous.py`**: Provides a side-by-side performance and physical comparison between the continuous (XY + anisotropy, strength via `--aniso`) and discrete implementations. Supports `--seed` for reproducible sweeps and `--log-file` like the other clock scripts.

## 4. Cross-Model Benchmarking (`scripts/benchmarks/`)

- **`throughput.py`**: Cross-model throughput and scaling benchmark. Measures sweeps/s, ns/site, and per-call analysis costs (thermodynamic, $G(r)$, vorticity, helicity) across all eight model variants and a range of lattice sizes. Saves a 6-panel summary figure to `results/benchmarks/scaling_benchmark.png` and all metrics to `results/benchmarks/scaling_benchmark.npz`. The NPZ file is loaded by `notebooks/Performance_Benchmarks.ipynb` to avoid re-running the benchmark on every notebook execution.

## 5. Regenerating All Data (`scripts/generate_all.py`)

- **`generate_all.py`**: Runs every script that a notebook reads data from, in one command,
  and reports what it produced and how long each step took. Without arguments each script
  runs at its own production defaults, which is the data the published figures are built
  from and takes roughly an hour on four cores. `--quick` runs the same sixteen scripts at
  sharply reduced lattices and step counts, which exercises the whole pipeline in about a
  minute but produces output that is not physics. `--only` and `--skip` select by substring
  of the `model/script` key, `--list` prints the table, `--dry-run` prints the commands,
  `--skip-existing` leaves present files alone, and `--fail-fast` stops at the first failure
  instead of reporting all of them at the end. A failing run exits non-zero, so the command
  works as a build step. The set of datasets it covers is held to the pipeline table below
  by `tests/integration/test_generate_all.py`, so a new data-producing script cannot be
  added to one without the other.

## Usage Guidelines

### Update Schemes
As mandated in `AGENTS.md`, each script hardcodes the physically appropriate update scheme:
- Kinetics scripts (`ordering_kinetics.py`, `ordering_evolution.py`, coarsening analyses) use **random site selection**, the only scheme valid for time-dependent studies.
- Sweep scripts (`temperature_sweep.py` and related equilibrium tools) use **checkerboard updates** for maximum equilibrium throughput.
- `wolff_efficiency.py` and `measure_z.py` additionally run the **Wolff cluster algorithm**, which reduces $\tau_{\mathrm{int}}$ by an order of magnitude near $T_c$ and improves the ISS rate proportionally. See `notebooks/Wolff_Efficiency.ipynb` for a quantitative demonstration.
- Only the model `main()` entry points (`python -m models.ising_model` etc.) expose an `--update` CLI flag for direct experimentation.

### Results
All scripts save their output (plots and data files) to the `results/` directory, sub-divided by model and experiment type.

## Production Defaults and Notebook Fallbacks

Scripts and notebooks serve different goals, so their parameters differ deliberately.
Script defaults favour data quality: the three temperature sweeps now run 60 temperature
points with 20 000 measurement steps and 5 seed replicas (Ising and XY at L=64, Clock at
L=48), which is what makes the hierarchical between-seed uncertainty meaningful; with the
previous single-seed default no between-seed component existed at all. A full sweep takes
roughly 3 minutes (Ising), 11 minutes (XY) and 16 minutes (Clock) on four cores.

Notebook fallbacks favour responsiveness: they run only when the cached NPZ is absent and
are sized so a reader never waits on a seemingly frozen cell. Each sweep fallback completes
in well under a minute and still resolves the transition; the notebooks print the lattice
size and step count they used, so a fallback figure is never mistaken for production data.

## Data Pipeline

`python -m scripts.generate_all` regenerates every entry in this table at once; the sections above describe running an individual script.

Scripts serve a dual role: they generate cached NPZ data files for notebooks and produce quick-check PNG figures for immediate visual verification. Notebooks are the curated presentation layer and load precomputed NPZ files to avoid re-running expensive simulations. When NPZ data is absent, notebooks provide lightweight fallback computations for demonstration purposes.

The table below maps each script to its data output and the notebook(s) that consume it. Scripts marked "figure only" produce PNG visualizations but no cached data; their analyses are reproduced inline within notebooks at small system sizes when needed.

| Script | NPZ Output | Consuming Notebook(s) |
|--------|-----------|----------------------|
| **Ising** | | |
| `scripts/ising/temperature_sweep.py` | `results/ising/temperature_sweep_data.npz` | `Ising_Temperature_Sweep.ipynb`, `Ising_Relaxation_and_Autocorrelation_Analysis.ipynb` |
| `scripts/ising/measure_z.py` | `results/ising/dynamic_exponent_z.npz` | `Dynamic_Critical_Exponents.ipynb` |
| `scripts/ising/wolff_efficiency.py` | `results/ising/wolff_efficiency.npz` | `Wolff_Efficiency.ipynb` |
| `scripts/ising/correlation_comparison.py` | `results/ising/correlation_comparison.npz` | `Correlation_and_Coarsening.ipynb` |
| `scripts/ising/correlation_divergence.py` | `results/ising/correlation_divergence.npz` | `Correlation_and_Coarsening.ipynb` |
| `scripts/ising/ordering_kinetics.py` | `results/ising/ordering_kinetics.npz` | `Correlation_and_Coarsening.ipynb` |
| `scripts/ising/coarsening_analysis.py` | `results/ising/coarsening_analysis.npz` | `Correlation_and_Coarsening.ipynb` |
| `scripts/ising/diag_eq_traces.py` | *(figure only)* | — |
| `scripts/ising/ordering_evolution.py` | *(figure only)* | — |
| **XY** | | |
| `scripts/xy/temperature_sweep.py` | `results/xy/temperature_sweep_data.npz` | `XY_Temperature_Sweep.ipynb` |
| `scripts/xy/bkt_transition.py` | `results/xy/bkt_transition.npz` | `BKT_Transition.ipynb` |
| `scripts/xy/helicity_modulus.py` | `results/xy/helicity_modulus.npz` | `BKT_Transition.ipynb` |
| `scripts/xy/correlation_comparison.py` | `results/xy/correlation_comparison.npz` | `BKT_Transition.ipynb`, `Correlation_and_Coarsening.ipynb` |
| `scripts/xy/ordering_kinetics.py` | `results/xy/ordering_kinetics.npz` | `Correlation_and_Coarsening.ipynb` |
| `scripts/xy/ordering_evolution.py` | *(figure only)* | — |
| **Clock** | | |
| `scripts/clock/temperature_sweep.py` | `results/clock/temperature_sweep_data.npz` | `Clock_Temperature_Sweep.ipynb` |
| `scripts/clock/ordering_kinetics.py` | `results/clock/ordering_kinetics.npz` | `Correlation_and_Coarsening.ipynb` |
| `scripts/clock/correlation_comparison.py` | `results/clock/correlation_comparison.npz` | `Correlation_and_Coarsening.ipynb` |
| `scripts/clock/ordering_evolution.py` | *(figure only)* | — |
| `scripts/clock/compare_discrete_vs_continuous.py` | *(figure only)* | — |
| **Benchmarks** | | |
| `scripts/benchmarks/throughput.py` | `results/benchmarks/scaling_benchmark.npz` | `Performance_Benchmarks.ipynb` |
