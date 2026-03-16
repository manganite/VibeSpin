# Physics and Algorithm Guide: VibeSpin

This document details the theoretical foundations, physical models, and algorithmic requirements of the VibeSpin framework.

## 1. Physical Models

VibeSpin implements three foundational 2D lattice spin models, each defined by its Hamiltonian and state space. They span the full spectrum from discrete to continuous on-site symmetry, and together they cover the major universality classes accessible in two dimensions.

### Ising Model

The Ising model assigns a scalar spin $s_i \in \{+1, -1\}$ to every site of a square lattice. Nearest-neighbor pairs interact through the Hamiltonian

$$E = -J \sum_{\langle i,j \rangle} s_i s_j,$$

where $\langle i,j \rangle$ runs over all distinct neighbor bonds. The competition between the ferromagnetic coupling $J > 0$, which favors alignment, and thermal fluctuations drives a second-order phase transition at the Onsager critical point $T_c \approx 2.269\,J/k_B$. Below $T_c$ the system spontaneously magnetizes; above it, entropy dominates and the net magnetization vanishes.

### XY Model

In the XY model each spin is a 2D unit vector $\mathbf{s}_i = (\cos\theta_i,\,\sin\theta_i)$ free to point in any planar direction. The Hamiltonian takes the form

$$E = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j).$$

Because continuous symmetry cannot break spontaneously in two dimensions (Mermin–Wagner theorem), the XY model does not develop true long-range order at any finite temperature. Instead it undergoes the Berezinskii–Kosterlitz–Thouless (BKT) transition: at low temperature, bound vortex–antivortex pairs maintain quasi-long-range order with algebraically decaying correlations, while above $T_{\mathrm{BKT}}$ the pairs unbind and correlations decay exponentially. This topological mechanism makes the 2D XY model qualitatively distinct from conventional order–disorder transitions.

### q-state Clock Model

The clock model interpolates between the Ising limit ($q = 2$) and the XY limit ($q \to \infty$) by restricting spins to $q$ equally spaced angles $\theta_k = 2\pi k / q$. VibeSpin provides two representations. The **continuous** form retains the XY interaction and adds an anisotropy potential that pins spins toward the discrete directions:

$$E = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j) \;-\; A \sum_i \cos(q\,\theta_i).$$

The **discrete** form evaluates the same interaction directly on integer state indices $k_i \in \{0,\dots,q-1\}$ using precomputed cosine lookup tables, eliminating per-site trigonometric calls. For large $q$ the model exhibits two successive BKT-type crossovers — one for the onset of quasi-long-range order and a lower one for the discrete locking transition — while for small $q$ the behavior collapses to an Ising-like single transition.

## 2. The Metropolis-Hastings Algorithm

All simulations in VibeSpin sample the Boltzmann distribution $P(s) \propto \exp(-\beta E(s))$ via the Metropolis-Hastings algorithm.

### Detailed Balance

Convergence to the target distribution requires that every pair of configurations $A$ and $B$ satisfies $P(A)\,W(A \to B) = P(B)\,W(B \to A)$. The Metropolis acceptance rule

$$P_{\mathrm{acc}} = \min\!\bigl(1,\;\exp(-\beta\,\Delta E)\bigr)$$

fulfills this condition exactly, accepting all energy-lowering moves and accepting energy-raising moves with an exponentially suppressed probability.

### Ergodicity

The proposal distribution must connect every configuration to every other in a finite number of steps. In the Ising model, single-spin flips are sufficient because any configuration can be reached one flip at a time. The discrete clock model draws each proposal uniformly from the full set of $q$ states (`randint(0, q)`), so every site can reach any allowed orientation in a single move. For the XY model, uniform phase perturbations drawn from $[-\delta,\,\delta]$ ensure that successive proposals can accumulate to traverse the entire $[0, 2\pi)$ circle.

### Update Schemes

VibeSpin enforces a strict separation between two update strategies, each valid only in its own physical regime. **Checkerboard updates** are used exclusively for equilibrium and thermodynamic measurements: the lattice is divided into two independent sublattices (analogous to the black and white squares of a chessboard), and each sublattice is swept in parallel. Because no two simultaneously updated sites share a neighbor, this scheme is both correct and highly vectorizable, maximizing SIMD and multi-core throughput. **Random site selection** is mandatory for kinetics and non-equilibrium dynamics: $N^2$ sites are chosen uniformly at random per sweep, preserving the exact stochastic trajectory needed to study coarsening, domain growth, and aging phenomena. Mixing the two schemes across regimes would invalidate either the parallelism guarantee or the physical time evolution.

## 3. Physical Observables

### Thermodynamic Averages

The equilibrium state of each model is characterized by four primary thermodynamic quantities. The **magnetization** $M = N^{-1}\sum_i s_i$ tracks the degree of spontaneous symmetry breaking — it saturates to unity in the ground state and vanishes in the paramagnetic phase. The **energy per site** $E = \langle H \rangle / N$ reflects the average bond alignment. Fluctuations of these quantities yield two response functions: the **magnetic susceptibility** $\chi = (N/T)\bigl(\langle M^2 \rangle - \langle M \rangle^2\bigr)$, which diverges at a continuous phase transition, and the **specific heat** $C_v = (N/T^2)\bigl(\langle E^2 \rangle - \langle E \rangle^2\bigr)$, whose peak marks the critical region.

Temperature-sweep simulations also compute the **entropy** by integrating the specific-heat curve downward from a high-temperature reference:

$$S(T) = S_{\mathrm{ref}} - \int_T^{T_{\mathrm{ref}}} \frac{C_v(T')}{T'}\,dT'.$$

The highest simulated temperature serves as the reference point. For clock models the absolute high-temperature limit is $S_{\mathrm{ref}} = \ln q$ per site (in units of $k_B$), corresponding to equipartition over all $q$ orientations.

Finally, the **integrated autocorrelation time** $\tau_{\mathrm{int}}$, extracted from the magnetization time series, quantifies how many sweeps separate statistically independent samples. Near a critical point $\tau_{\mathrm{int}}$ diverges — the hallmark of critical slowing down — and its magnitude directly governs the statistical efficiency of the Monte Carlo run. VibeSpin uses $\tau_{\mathrm{int}}$ not only as a reported observable but as an active control signal: the adaptive equilibration routine in `utils/system_helpers.py` extends the burn-in until the probe window spans at least `factor` $\times$ $\tau_{\mathrm{int}}$ sweeps, ensuring that equilibrium measurements are taken from a decorrelated state regardless of how strongly $\tau_{\mathrm{int}}$ varies with temperature.

### Spatial Diagnostics

Two complementary probes measure the spatial structure of equilibrium configurations. The **pair correlation function** $G(r) = \langle \mathbf{s}(0) \cdot \mathbf{s}(r) \rangle$ reveals how spin–spin alignment decays with distance: exponentially in the disordered phase (with a correlation length $\xi$) and algebraically in the quasi-long-range-ordered regime of the XY model. Its Fourier counterpart, the **structure factor** $S(k)$, highlights the dominant fluctuation wavevectors — a sharp peak at $k = 0$ signals ferromagnetic order, while a broad ring indicates short-range correlations at a characteristic length scale.

### Topological Diagnostics

In models with continuous or near-continuous symmetry (XY and large-$q$ clock), point-like topological defects govern the phase behavior. The **vorticity** of a plaquette (elementary square of four sites) is obtained by summing the directed nearest-neighbor phase differences around its perimeter; a non-zero winding number $\pm 1$ identifies a vortex or antivortex core. The **vortex density** — the fraction of plaquettes carrying a defect — rises sharply above the BKT temperature as thermally excited pairs unbind. The **helicity modulus** $\Upsilon$ measures the free-energy cost of imposing an infinitesimal phase twist across the system. In the low-temperature phase $\Upsilon$ remains finite, reflecting superfluid-like stiffness; it drops discontinuously to zero at $T_{\mathrm{BKT}}$ with the universal Nelson–Kosterlitz jump $\Upsilon(T_{\mathrm{BKT}}^-) = 2T_{\mathrm{BKT}}/\pi$, providing the cleanest numerical signature of the BKT transition in two dimensions.

## 4. The Wolff Cluster Algorithm

### Motivation: Critical Slowing Down

The Metropolis single-spin-flip algorithm becomes increasingly inefficient as a continuous phase transition is approached. Near the critical point the correlation length $\xi$ diverges, and the spin configurations develop large coherent domains whose characteristic size $\xi$ sets the natural unit of any proposed single-site change. Because a single flip disturbs only one spin at a time, the algorithm must perform $O(\xi^z)$ sweeps to decorrelate the system from one independent sample to the next, where the dynamic critical exponent $z \approx 2.17$ for the 2D Ising model. This quadratic growth of autocorrelation time with lattice size — the critical slowing down — makes precise equilibrium measurements near $T_c$ computationally expensive with Metropolis alone.

The Wolff cluster algorithm eliminates critical slowing down by operating at the scale of the correlated domain rather than the individual spin. Rather than proposing a single flip, it grows an entire correlated cluster and flips it as a single collective move. The dynamic exponent drops to $z \approx 0.25$ for the 2D Ising model — an order-of-magnitude reduction in autocorrelation time at criticality.

### Ising Wolff: Fortuin-Kasteleyn Construction

The theoretical basis for the Ising Wolff algorithm is the Fortuin-Kasteleyn (FK) random-cluster representation, which maps the Ising partition function onto a bond-percolation problem. Starting from a randomly chosen seed site with spin $\sigma_0 = \pm 1$, the algorithm explores all connected same-spin neighbors. Each bond $(i,j)$ between two aligned spins is activated independently with the probability

$$P_{\mathrm{add}} = 1 - e^{-2\beta J}.$$

After the cluster $\mathcal{C}$ is fully grown, all spins in $\mathcal{C}$ are flipped simultaneously: $\sigma_i \to -\sigma_i$ for all $i \in \mathcal{C}$. The bond probability is derived precisely so that this collective move satisfies detailed balance without any rejection step — the cluster flip is always accepted. This zero-rejection property, combined with the divergence of the mean cluster size $\langle|\mathcal{C}|\rangle \sim \xi^{d_f}$ at $T_c$ (where $d_f$ is the fractal dimension of the FK cluster), is the mechanism behind the dramatic acceleration near criticality.

### XY and Clock Wolff: The Reflection Trick

The single-component Ising flip does not generalise directly to models with continuous $O(2)$ symmetry such as the XY and clock models — there is no natural binary decomposition. The Wolff-Evertz generalisation proceeds via a random mirror-plane construction. A unit vector $\hat{r} = (\cos\phi,\,\sin\phi)$ is drawn uniformly from the unit circle. Each spin is projected onto $\hat{r}$: $\sigma_i = \mathbf{s}_i \cdot \hat{r}$. This projection maps the $O(2)$ spins onto a real-valued "Ising-like" field, and bonds between neighbouring sites with aligned projections ($\sigma_i \sigma_j > 0$) are activated with probability

$$P_{\mathrm{add}} = 1 - e^{-2\beta J\,\sigma_i \sigma_j}.$$

Once the cluster is formed, every cluster spin is reflected through the hyperplane perpendicular to $\hat{r}$:

$$\mathbf{s}_i \to \mathbf{s}_i - 2(\mathbf{s}_i \cdot \hat{r})\,\hat{r}.$$

This reflection preserves the Euclidean norm $|\mathbf{s}_i| = 1$ exactly, requires no renormalisation, and satisfies detailed balance for the pure Heisenberg exchange term $-J \mathbf{s}_i \cdot \mathbf{s}_j$. For the **continuous clock model**, which includes an additional crystal-field anisotropy $-A\cos(q\theta_i)$, the reflection symmetry required by the FK bond construction holds only for the exchange part of the Hamiltonian; the anisotropy term breaks this symmetry and falls outside the standard Wolff-Evertz derivation. The Wolff kernel therefore satisfies detailed balance exactly only when $A = 0$, and converges to an approximation of the correct equilibrium distribution for $A > 0$. Its use in the clock model is most appropriate when $A \ll J$, or as a diagnostic in the XY-like regime.

### Practical Semantics

One call to a Wolff `step()` constitutes one cluster sweep, meaning a single cluster is grown and flipped. This differs from the Metropolis convention, where one `step()` comprises $N^2$ single-spin-flip attempts. The two schemes are therefore not directly comparable on a per-step basis near $T_c$; the relevant comparison is per unit of computational time or per independent sample. Because the mean cluster size scales with the correlation length, the cost per cluster sweep also scales with $\xi$, but the autocorrelation time in units of sweeps falls far faster than it rises in cost, resulting in a substantial net gain precisely where it is most needed.

The `parallel=True` flag is silently ignored when `update='wolff'`. Cluster growth is a sequential depth-first search whose frontier depends on each newly added site; it cannot be decomposed into independent sublattices and is therefore incompatible with the checkerboard parallelisation strategy. The Wolff algorithm is inherently a single-threaded traversal, and its performance advantage over Metropolis is algorithmic rather than hardware-parallel.
