# Physics and Algorithm Guide: VibeSpin

This document details the theoretical foundations, physical models, and algorithmic requirements of the VibeSpin framework.

## 1. Physical Models

VibeSpin implements three foundational 2D lattice spin models. Each is defined by its Hamiltonian (energy function) and state space.

### Ising Model
The simplest model of ferromagnetism.
- **State Space**: Discrete spins $s_i \in \{+1, -1\}$.
- **Hamiltonian**: $E = -J \sum_{\langle i,j \rangle} s_i s_j$, where $\langle i,j \rangle$ denotes nearest-neighbor pairs.
- **Physics**: Exhibits a second-order phase transition at $T_c \approx 2.269J/k_B$ in 2D.

### XY Model
A continuous model for planar spins.
- **State Space**: 2D unit vectors $\mathbf{s}_i = (\cos\theta_i, \sin\theta_i)$.
- **Hamiltonian**: $E = -J \sum_{\langle i,j \rangle} \mathbf{s}_i \cdot \mathbf{s}_j = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j)$.
- **Physics**: Exhibits the Berezinskii–Kosterlitz–Thouless (BKT) transition driven by vortex-antivortex pairing.

### q-state Clock Model
Interpolates between Ising ($q=2$) and XY ($q \to \infty$).
- **State Space**: Spins constrained to $q$ discrete angles $\theta_k = 2\pi k/q$.
- **Implementation**: VibeSpin provides both **Continuous** (XY with anisotropy) and **Discrete** (integer lookup) versions.
- **Hamiltonian**: Same as XY, but with discrete allowed states.

## 2. The Metropolis-Hastings Algorithm

All simulations in VibeSpin utilize the Metropolis-Hastings algorithm to sample the Boltzmann distribution $P(s) \propto \exp(-\beta E(s))$.

### Detailed Balance
To ensure the simulation converges to the correct equilibrium distribution, the transition probability $W(A \to B)$ must satisfy:
$P(A) W(A \to B) = P(B) W(B \to A)$
This is achieved using the Metropolis acceptance rule:
$P_{acc} = \min(1, \exp(-\beta \Delta E))$

### Ergodicity
The proposal distribution must allow the system to reach any state from any other state in a finite number of steps. 
- **Ising**: Single-spin flips ensure ergodicity.
- **Discrete Clock**: Global site-state proposals (`randint(0, q)`) provide high ergodicity.
- **XY**: Uniform phase proposals in $[-\delta, \delta]$ ensure any configuration is reachable over multiple steps.

### Update Schemes
- **Checkerboard Update**: Used for **Equilibrium/Thermodynamics**. Divides the lattice into two independent sublattices (like a chessboard). Each sublattice is updated in parallel, maximizing cache efficiency and SIMD vectorization.
- **Random Site Selection**: Mandatory for **Kinetics/Dynamics**. Randomly selects $N^2$ sites per sweep. This preserves the exact stochastic trajectory required for studying coarsening and aging.

## 3. Physical Observables

### Thermodynamic Averages
- **Magnetization**: $M = \frac{1}{N} \sum_i s_i$. Measures order.
- **Energy**: $E = \langle H \rangle / N$.
- **Susceptibility**: $\chi = \frac{N}{T} (\langle M^2 \rangle - \langle M \rangle^2)$.
- **Specific Heat**: $C_v = \frac{N}{T^2} (\langle E^2 \rangle - \langle E \rangle^2)$.

### Spatial Diagnostics
- **Correlation Function $G(r)$**: $\langle \mathbf{s}(0) \cdot \mathbf{s}(r) \rangle$. Measures the decay of order over distance.
- **Structure Factor $S(k)$**: The Fourier transform of the spin field. Identifies dominant fluctuation modes.

### Topological Diagnostics
- **Vorticity**: Calculated by summing directed phase differences around each plaquette (square of 4 sites).
- **Vortex Density**: The fraction of plaquettes with non-zero winding number.
- **Helicity Modulus**: Measures the system's stiffness against a phase-twist, used to identify the BKT transition.
