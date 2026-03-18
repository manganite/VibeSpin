# Physics and Algorithm Guide: VibeSpin

This document details the theoretical foundations, physical models, and algorithmic requirements of the VibeSpin framework.

## 1. Physical Models

VibeSpin implements three foundational 2D lattice spin models, each defined by its Hamiltonian and state space. They span the full spectrum from discrete to continuous on-site symmetry, and together they cover the major universality classes accessible in two dimensions.



### Ising Model

For a comprehensive introduction to the Ising model and its exact solution in two dimensions, see Onsager [[1]](#Bibliography). See also the [Ising model article on Wikipedia](https://en.wikipedia.org/wiki/Ising_model) for a modern summary.

The Ising model assigns a scalar spin $s_i \in \{+1, -1\}$ to every site of a square lattice. Nearest-neighbor pairs interact through the Hamiltonian

$$E = -J \sum_{\langle i,j \rangle} s_i s_j,$$

where $\langle i,j \rangle$ runs over all distinct neighbor bonds. The competition between the ferromagnetic coupling $J > 0$, which favors alignment, and thermal fluctuations drives a second-order phase transition at the Onsager critical point $T_c \approx 2.269\,J/k_B$. Below $T_c$ the system spontaneously magnetizes; above it, entropy dominates and the net magnetization vanishes.

### XY Model



In the XY model each spin is a 2D unit vector $\mathbf{s}_i = (\cos\theta_i,\,\sin\theta_i)$ free to point in any planar direction. The Hamiltonian takes the form

$$E = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j).$$



Because continuous symmetry cannot break spontaneously in two dimensions (Mermin–Wagner theorem), the XY model does not develop true long-range order at any finite temperature. Instead it undergoes the Berezinskii–Kosterlitz–Thouless (BKT) transition: at low temperature, bound vortex–antivortex pairs maintain quasi-long-range order with algebraically decaying correlations, while above $T_{\mathrm{BKT}}$ the pairs unbind and correlations decay exponentially. This topological mechanism makes the 2D XY model qualitatively distinct from conventional order–disorder transitions.
For background, see Kosterlitz and Thouless [[2]](#Bibliography), Mermin and Wagner [[3]](#Bibliography), and Villain [[11]](#Bibliography).
See also the [XY model article on Wikipedia](https://en.wikipedia.org/wiki/XY_model).

### q-state Clock Model



The clock model interpolates between the Ising limit ($q = 2$) and the XY limit ($q \to \infty$) by restricting spins to $q$ equally spaced angles $\theta_k = 2\pi k / q$. VibeSpin provides two representations. The **continuous** form retains the XY interaction and adds an anisotropy potential that pins spins toward the discrete directions. For a review, see Lapilli et al. [[4]](#Bibliography).
See also the [Clock model (Vector Potts model) article on Wikipedia](https://en.wikipedia.org/wiki/Potts_model#Vector_Potts_model).

$$E = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j) \;-\; A \sum_i \cos(q\,\theta_i).$$

The **discrete** form evaluates the same interaction directly on integer state indices $k_i \in \{0,\dots,q-1\}$ using precomputed cosine lookup tables, eliminating per-site trigonometric calls. For large $q$ the model exhibits two successive BKT-type crossovers — one for the onset of quasi-long-range order and a lower one for the discrete locking transition — while for small $q$ the behavior collapses to an Ising-like single transition.

## 2. The Metropolis-Hastings Algorithm



All simulations in VibeSpin sample the Boltzmann distribution $P(s) \propto \exp(-\beta E(s))$ via the Metropolis-Hastings algorithm. For a pedagogical introduction, see Hastings [[5]](#Bibliography).
See also the [Metropolis–Hastings algorithm article on Wikipedia](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm).

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


For a pedagogical introduction to thermodynamic observables in spin models, see Huang [[6]](#Bibliography).
See also the [Magnetization](https://en.wikipedia.org/wiki/Magnetization), [Magnetic susceptibility](https://en.wikipedia.org/wiki/Magnetic_susceptibility), and [Heat capacity](https://en.wikipedia.org/wiki/Heat_capacity) articles on Wikipedia.

Temperature-sweep simulations also compute the **entropy** by integrating the specific-heat curve downward from a high-temperature reference:

$$S(T) = S_{\mathrm{ref}} - \int_T^{T_{\mathrm{ref}}} \frac{C_v(T')}{T'}\,dT'.$$

The highest simulated temperature serves as the reference point. For clock models the absolute high-temperature limit is $S_{\mathrm{ref}} = \ln q$ per site (in units of $k_B$), corresponding to equipartition over all $q$ orientations.

Finally, the **integrated autocorrelation time** $\tau_{\mathrm{int}}$, extracted from the magnetization time series, quantifies how many sweeps separate statistically independent samples [[12]](#Bibliography).
 Near a critical point $\tau_{\mathrm{int}}$ diverges — the hallmark of critical slowing down — and its magnitude directly governs the statistical efficiency of the Monte Carlo run. To ensure measurements are strictly taken from the stationary distribution, VibeSpin uses a **Two-Start Convergence** equilibration routine: at each temperature point, two simulations are launched—one from a random (disordered) state and one from a fully aligned (ordered) state. The burn-in proceeds in chunks until the absolute magnetization traces of both trajectories converge and stay within a common equilibrium band (typically defined by the standard deviation of the combined traces). This non-equilibrium diagnostic guarantees that the system has "forgotten" its initial state before any thermodynamic measurements are recorded.

### Spatial Diagnostics


For a review of spatial diagnostics and correlation functions, see Goldenfeld [[7]](#Bibliography).
See also the [Correlation function](https://en.wikipedia.org/wiki/Correlation_function_(statistical_mechanics)) and [Structure factor](https://en.wikipedia.org/wiki/Structure_factor) articles on Wikipedia.


### Topological Diagnostics


For the BKT transition and topological diagnostics, see Kosterlitz and Thouless [[2]](#Bibliography).
See also the [BKT transition](https://en.wikipedia.org/wiki/Berezinskii%E2%80%93Kosterlitz%E2%80%93Thouless_transition), [Vortex](https://en.wikipedia.org/wiki/Vortex), and [Superfluid stiffness (Helicity modulus)](https://en.wikipedia.org/wiki/Superfluid_stiffness) articles on Wikipedia.

## 4. The Wolff Cluster Algorithm

### Motivation: Critical Slowing Down

The Metropolis single-spin-flip algorithm becomes increasingly inefficient as a continuous phase transition is approached. Near the critical point the correlation length $\xi$ diverges, and the spin configurations develop large coherent domains whose characteristic size $\xi$ sets the natural unit of any proposed single-site change. Because a single flip disturbs only one spin at a time, the algorithm must perform $O(\xi^z)$ sweeps to decorrelate the system from one independent sample to the next, where the dynamic critical exponent $z \approx 2.17$ for the 2D Ising model. This quadratic growth of autocorrelation time with lattice size — the critical slowing down — makes precise equilibrium measurements near $T_c$ computationally expensive with Metropolis alone.

The Wolff cluster algorithm eliminates critical slowing down by operating at the scale of the correlated domain rather than the individual spin. Rather than proposing a single flip, it grows an entire correlated cluster and flips it as a single collective move. The dynamic exponent drops to $z \approx 0.25$ for the 2D Ising model — an order-of-magnitude reduction in autocorrelation time at criticality.

### Ising Wolff: Fortuin-Kasteleyn Construction


The theoretical basis for the Ising Wolff algorithm is the Fortuin-Kasteleyn (FK) random-cluster representation, which maps the Ising partition function onto a bond-percolation problem. For the FK construction, see Fortuin and Kasteleyn [[8]](#Bibliography).
See also the [Random cluster model (Fortuin–Kasteleyn representation)](https://en.wikipedia.org/wiki/Random_cluster_model) article on Wikipedia.

$$P_{\mathrm{add}} = 1 - e^{-2\beta J}.$$

After the cluster $\mathcal{C}$ is fully grown, all spins in $\mathcal{C}$ are flipped simultaneously: $\sigma_i \to -\sigma_i$ for all $i \in \mathcal{C}$. The bond probability is derived precisely so that this collective move satisfies detailed balance without any rejection step — the cluster flip is always accepted. This zero-rejection property, combined with the divergence of the mean cluster size $\langle|\mathcal{C}|\rangle \sim \xi^{d_f}$ at $T_c$ (where $d_f$ is the fractal dimension of the FK cluster), is the mechanism behind the dramatic acceleration near criticality.

### XY and Clock Wolff: The Reflection Trick


For the Wolff-Evertz generalization to $O(2)$ models, see Wolff [[9]](#Bibliography) and Newman and Barkema [[10]](#Bibliography).
See also the [Wolff algorithm](https://en.wikipedia.org/wiki/Wolff_algorithm) article on Wikipedia.

$$P_{\mathrm{add}} = 1 - e^{-2\beta J\,\sigma_i \sigma_j}.$$

Once the cluster is formed, every cluster spin is reflected through the hyperplane perpendicular to $\hat{r}$:

$$\mathbf{s}_i \to \mathbf{s}_i - 2(\mathbf{s}_i \cdot \hat{r})\,\hat{r}.$$

This reflection preserves the Euclidean norm $|\mathbf{s}_i| = 1$ exactly, requires no renormalisation, and satisfies detailed balance for the pure Heisenberg exchange term $-J \mathbf{s}_i \cdot \mathbf{s}_j$. For the **continuous clock model**, which includes an additional crystal-field anisotropy $-A\cos(q\theta_i)$, the reflection symmetry required by the FK bond construction holds only for the exchange part of the Hamiltonian; the anisotropy term breaks this symmetry and falls outside the standard Wolff-Evertz derivation. The Wolff kernel therefore satisfies detailed balance exactly only when $A = 0$, and converges to an approximation of the correct equilibrium distribution for $A > 0$. Its use in the clock model is most appropriate when $A \ll J$, or as a diagnostic in the XY-like regime.

### Practical Semantics

One call to a Wolff `step()` constitutes one cluster sweep, meaning a single cluster is grown and flipped. This differs from the Metropolis convention, where one `step()` comprises $N^2$ single-spin-flip attempts. The two schemes are therefore not directly comparable on a per-step basis near $T_c$; the relevant comparison is per unit of computational time or per independent sample. Because the mean cluster size scales with the correlation length, the cost per cluster sweep also scales with $\xi$, but the autocorrelation time in units of sweeps falls far faster than it rises in cost, resulting in a substantial net gain precisely where it is most needed.

The `parallel=True` flag is silently ignored when `update='wolff'`. Cluster growth is a sequential depth-first search whose frontier depends on each newly added site; it cannot be decomposed into independent sublattices and is therefore incompatible with the checkerboard parallelisation strategy. The Wolff algorithm is inherently a single-threaded traversal, and its performance advantage over Metropolis is algorithmic rather than hardware-parallel.


## Bibliography

[[1]](#Bibliography) L. Onsager, "Crystal Statistics. I. A Two-Dimensional Model with an Order-Disorder Transition," *Physical Review*, vol. 65, no. 3-4, pp. 117–149, 1944. [APS Open Access](https://journals.aps.org/pr/abstract/10.1103/PhysRev.65.117)

[[2]](#Bibliography) J. M. Kosterlitz and D. J. Thouless, "Ordering, metastability and phase transitions in two-dimensional systems," *Journal of Physics C: Solid State Physics*, vol. 6, no. 7, pp. 1181–1203, 1973. [IOP Open Access](https://iopscience.iop.org/article/10.1088/0022-3719/6/7/010)

[[3]](#Bibliography) N. D. Mermin and H. Wagner, "Absence of Ferromagnetism or Antiferromagnetism in One- or Two-Dimensional Isotropic Heisenberg Models," *Physical Review Letters*, vol. 17, no. 22, pp. 1133–1136, 1966. [APS Open Access](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.17.1133)

[[4]](#Bibliography) J. Lapilli, P. Pfeifer, and C. Wexler, "Universality away from critical points in two-dimensional phase transitions," *Physical Review Letters*, vol. 96, no. 14, 140603, 2006. [APS Open Access](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.96.140603)

[[5]](#Bibliography) W. K. Hastings, "Monte Carlo sampling methods using Markov chains and their applications," *Biometrika*, vol. 57, no. 1, pp. 97–109, 1970. [Oxford Academic](https://academic.oup.com/biomet/article/57/1/97/252073)

[[6]](#Bibliography) K. Huang, "Statistical Mechanics," 2nd Edition, Wiley, 1987. [Statistical Mechanics lecture notes, John Cardy, Oxford (Archive)](https://arxiv.org/pdf/0807.3472.pdf)

[[7]](#Bibliography) N. Goldenfeld, "Lectures on Phase Transitions and the Renormalization Group," Addison-Wesley, 1992. [Internet Archive (Open Access)](https://archive.org/details/lecturesonphaset0000gold)

[[8]](#Bibliography) C. M. Fortuin and P. W. Kasteleyn, "On the random-cluster model. I. Introduction and relation to other models," *Physica*, vol. 57, no. 4, pp. 536–564, 1972. [Elsevier Open Access](https://doi.org/10.1016/0031-8914(72)90045-6)

[[9]](#Bibliography) U. Wolff, "Collective Monte Carlo Updating for Spin Systems," *Physical Review Letters*, vol. 62, no. 4, pp. 361–364, 1989. [APS Open Access](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.62.361)

[[10]](#Bibliography) M. E. J. Newman and G. T. Barkema, "Monte Carlo Methods in Statistical Physics," Oxford University Press, 1999. [Lecture Notes Summary (H. G. Katzgraber)](https://arxiv.org/abs/0905.1629)

[[11]](#Bibliography) J. Villain, "Theory of one- and two-dimensional magnets with an easy magnetization plane. II. The planar, classical, two-dimensional magnet," *J. Phys. France* 36, 581-590 (1975). [Open Access](https://doi.org/10.1051/jphys:01975003606058100)

[[12]](#Bibliography) A. D. Sokal, \"Monte Carlo Methods in Statistical Mechanics: Foundations and New Algorithms,\" lecture notes (1989), published in *Functional Integration: Basics and Applications* (C. DeWitt-Morette, P. Cartier, A. Folacci, eds.), Springer, 1997, pp. 131–192. [Springer Link](https://link.springer.com/chapter/10.1007/978-1-4899-0319-8_6)
