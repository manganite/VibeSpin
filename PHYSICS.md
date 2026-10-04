# Physics and Algorithm Guide: VibeSpin

This document details the theoretical foundations, physical models, and algorithmic requirements of the VibeSpin framework.

## 1. Physical Models

VibeSpin implements three foundational 2D lattice spin models, each defined by its Hamiltonian and state space. They span the full spectrum from discrete to continuous on-site symmetry, and together they cover the major universality classes accessible in two dimensions.



### Ising Model

For a thorough introduction to the Ising model and its exact solution in two dimensions, see Onsager [[1]](#Bibliography). See also the [Ising model article on Wikipedia](https://en.wikipedia.org/wiki/Ising_model) for a modern summary.

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

The **discrete** form evaluates the same interaction directly on integer state indices $k_i \in \{0,\dots,q-1\}$ using precomputed cosine lookup tables, eliminating per-site trigonometric calls. It is the clock model proper, and the clock scripts use it by default; `--continuous` selects the anisotropic form instead. The two are different Hamiltonians. At $A = 0$ the continuous form is exactly the XY model, and it approaches the discrete model only as $A \to \infty$, so its transition temperatures depend on $A$ and literature values for the discrete model do not apply to it.

The phase behaviour of the discrete model depends on $q$. For $q = 2$ it is the Ising model with $T_c = 2J/\ln(1+\sqrt{2}) \approx 2.269\,J$. For $q = 3$ it maps onto the 3-state Potts model with coupling $3J/2$, giving $T_c = 3J/[2\ln(1+\sqrt{3})] \approx 1.492\,J$. For $q = 4$ it decouples into two Ising models with coupling $J/2$, so $T_c = J/\ln(1+\sqrt{2}) \approx 1.135\,J$. For $q \geq 5$ there are two BKT transitions: below $T_1$ the spins lock to one of the $q$ directions with true long-range order, between $T_1$ and $T_2$ the discreteness is irrelevant at long distances and correlations decay algebraically as in the XY model, and above $T_2$ they decay exponentially. For $q = 6$, Tomita and Okabe [[13]](#Bibliography) obtain $T_1 = 0.7014(11)\,J$ and $T_2 = 0.9008(6)\,J$ from a cluster study, against the earlier Monte Carlo estimates $T_1 = 0.68(2)\,J$ and $T_2 = 0.92(1)\,J$ that they quote; the scripts mark the older pair as approximate crossovers.

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
 Near a critical point $\tau_{\mathrm{int}}$ diverges, the hallmark of critical slowing down, and its magnitude directly governs the statistical efficiency of the Monte Carlo run. To ensure measurements are strictly taken from the stationary distribution, VibeSpin uses a **Two-Start Convergence** equilibration routine: at each $(T, \text{seed})$ point, two independent simulations are launched: one from a random (disordered) state and one from a fully aligned (ordered) state. The burn-in proceeds in chunks until both smoothed magnetization traces satisfy a mutual cross-band test: the random-start trace must lie within a band of $\pm k$ standard deviations of the ordered-start tail mean, and the ordered-start trace must simultaneously lie within $\pm k$ standard deviations of the random-start tail mean. A sigma floor prevents band collapse when either trace is nearly variance-free.

The two runs use independent random-number streams. With a shared seed they draw identical random numbers, and below $T_c$ the chains couple: their magnetization traces become perfectly correlated and pass the cross-band test without having relaxed independently.

This protocol includes **Quasi-Steady Stuck Detection** for domain-wall trapping in the ordered phase of models with a discrete order parameter. A random start can strand in a metastable state, for example a stripe of two Ising domains spanning the periodic lattice, whose lifetime grows rapidly with system size. The detector fires when the ordered-start trace has a small tail variance while the two smoothed traces still fail the cross-band test. Below the ordering temperature of the Ising and discrete clock models such a point is accepted and measured on the ordered start, which samples the equilibrium ordered state. The detector checks only the variance of the ordered trace, so it cannot distinguish a stranded random start from one that is still relaxing slowly. For the XY model, whose random start can take thousands of sweeps to shed its vortices near $T_{\mathrm{BKT}}$ but has no metastable domain states, and for all temperatures above ordering, the detector is therefore switched off and the pair runs until it converges or reaches the step cap. To maximize hardware utilization, these points are parallelized individually across the worker pool, ensuring full CPU throughput even for single-seed sweeps.

### Spatial Diagnostics


For a review of spatial diagnostics and correlation functions, see Goldenfeld [[7]](#Bibliography).
See also the [Correlation function](https://en.wikipedia.org/wiki/Correlation_function_(statistical_mechanics)) and [Structure factor](https://en.wikipedia.org/wiki/Structure_factor) articles on Wikipedia.


### Topological Diagnostics


For the BKT transition and topological diagnostics, see Kosterlitz and Thouless [[2]](#Bibliography).
See also the [BKT transition](https://en.wikipedia.org/wiki/Berezinskii%E2%80%93Kosterlitz%E2%80%93Thouless_transition), [Vortex](https://en.wikipedia.org/wiki/Vortex), and [Superfluid stiffness (Helicity modulus)](https://en.wikipedia.org/wiki/Superfluid_stiffness) articles on Wikipedia.

## 4. The Wolff Cluster Algorithm

### Motivation: Critical Slowing Down

The Metropolis single-spin-flip algorithm becomes increasingly inefficient as a continuous phase transition is approached. Near the critical point the correlation length $\xi$ diverges, and the spin configurations develop large coherent domains whose characteristic size $\xi$ sets the natural unit of any proposed single-site change. Because a single flip disturbs only one spin at a time, the algorithm must perform $O(\xi^z)$ sweeps to decorrelate the system from one independent sample to the next, where the dynamic critical exponent $z \approx 2.17$ for the 2D Ising model. This quadratic growth of autocorrelation time with lattice size, the critical slowing down, makes precise equilibrium measurements near $T_c$ computationally expensive with Metropolis alone.

The Wolff cluster algorithm drastically reduces critical slowing down by operating at the scale of the correlated domain rather than the individual spin. Rather than proposing a single flip, it grows an entire correlated cluster and flips it as a single collective move. The dynamic exponent in cluster-step units is $z^{\mathrm{cs}} \approx 0.5$ asymptotically. However, a single cluster-step flips an $O(L^{\gamma/\nu})$-size cluster (where $\gamma/\nu = 7/4$ for the 2D Ising model), so normalizing to equivalent-sweep units yields $z^{\mathrm{sw}} = z^{\mathrm{cs}} - 1/4 \approx 0.25$, more than an order-of-magnitude reduction compared to Metropolis. Measured values of $z^{\mathrm{cs}}$ at finite lattice sizes (e.g., $L = 16$–$128$) can deviate from the asymptotic value due to finite-size effects.

### Ising Wolff: Fortuin-Kasteleyn Construction


The theoretical basis for the Ising Wolff algorithm is the Fortuin-Kasteleyn (FK) random-cluster representation, which maps the Ising partition function onto a bond-percolation problem. For the FK construction, see Fortuin and Kasteleyn [[8]](#Bibliography).
See also the [Random cluster model (Fortuin–Kasteleyn representation)](https://en.wikipedia.org/wiki/Random_cluster_model) article on Wikipedia.

$$P_{\mathrm{add}} = 1 - e^{-2\beta J}.$$

After the cluster $\mathcal{C}$ is fully grown, all spins in $\mathcal{C}$ are flipped simultaneously: $\sigma_i \to -\sigma_i$ for all $i \in \mathcal{C}$. The bond probability is derived precisely so that this collective move satisfies detailed balance without any rejection step: the cluster flip is always accepted. This zero-rejection property, combined with the divergence of the mean cluster size $\langle|\mathcal{C}|\rangle \sim \xi^{d_f}$ at $T_c$ (where $d_f$ is the fractal dimension of the FK cluster), is the mechanism behind the dramatic acceleration near criticality.

**Unit Convention.** Reported measurements of $\tau_{\mathrm{int}}$ and the dynamic exponent $z$ depend critically on how a "step" is defined. In cluster-step units (one cluster-flip as one step), $\tau_{\mathrm{int}}$ and $z$ are measured directly from the Wolff trajectory. To convert to sweep-equivalent units (matching Metropolis, where one step = $N^2$ single-flip attempts), we normalize by the mean cluster size: $\tau^{\mathrm{sw}}_{\mathrm{Wolff}} = \tau^{\mathrm{cs}}_{\mathrm{Wolff}} \times \langle C \rangle / L^2 \sim \tau^{\mathrm{cs}}_{\mathrm{Wolff}} \times L^{7/4} / L^2 = \tau^{\mathrm{cs}}_{\mathrm{Wolff}} \times L^{-1/4}$. This gives $z^{\mathrm{sw}} = z^{\mathrm{cs}} - 1/4$, an exact relation for the 2D Ising model.

### XY and Clock Wolff: The Reflection Trick


For the Wolff-Evertz generalization to $O(2)$ models, see Wolff [[9]](#Bibliography) and Newman and Barkema [[10]](#Bibliography).
See also the [Wolff algorithm](https://en.wikipedia.org/wiki/Wolff_algorithm) article on Wikipedia.

A reflection axis $\hat{r}$ is drawn uniformly from the unit circle, and $\sigma_i = \mathbf{s}_i \cdot \hat{r}$ denotes the projection of spin $i$ onto it. A bond between neighbours is activated with probability

$$P_{\mathrm{add}} = 1 - e^{\min(0,\,-2\beta J\,\sigma_i \sigma_j)},$$

which vanishes unless the two projections share a sign.

Once the cluster is formed, every cluster spin is reflected through the hyperplane perpendicular to $\hat{r}$:

$$\mathbf{s}_i \to \mathbf{s}_i - 2(\mathbf{s}_i \cdot \hat{r})\,\hat{r}.$$

This reflection preserves the Euclidean norm $|\mathbf{s}_i| = 1$ exactly, requires no renormalisation, and satisfies detailed balance for the planar exchange term $-J\, \mathbf{s}_i \cdot \mathbf{s}_j$. For the **continuous clock model**, which adds the crystal-field anisotropy $-A\cos(q\theta_i)$, the bond construction sees only the exchange part. A chain built from these reflections alone samples the $A = 0$ Hamiltonian, which is the XY model, whatever value of $A$ is set. `ClockSimulation` therefore rejects `update='wolff'` for $A \neq 0$ with a `ValueError` rather than producing XY statistics under a clock label.

The **discrete clock model** admits an exact cluster update because reflections can be restricted to its symmetry group. The axis is drawn from the $q$ mirror axes of the regular $q$-gon, at angles $\varphi_m = \pi m / q$ for $m = 0, \dots, q-1$. Reflecting $\theta_s = 2\pi s/q$ about $\varphi_m$ gives $2\varphi_m - \theta_s$, the allowed state $s' = (m - s) \bmod q$, so no spin ever leaves the discrete state space. Tomita and Okabe [[13]](#Bibliography) restrict the embedding axis to the same set of directions for their cluster study of the clock model. The projection perpendicular to the mirror axis is $\sigma_i = \sin(\theta_{s_i} - \varphi_m)$, the bond probability is the same expression as above, and because each reflection is an involution chosen with probability $1/q$ in both directions, detailed balance holds for the full discrete Hamiltonian. A single-site cluster can reach every state $s'$ from $s$, so the chain is ergodic. The test suite checks both properties on the exact transition matrix of a $2 \times 2$ lattice.

### Practical Semantics

One call to a Wolff `step()` constitutes one cluster sweep, meaning a single cluster is grown and flipped. This differs from the Metropolis convention, where one `step()` comprises $N^2$ single-spin-flip attempts. The two schemes are therefore not directly comparable on a per-step basis near $T_c$; the relevant comparison is per unit of computational time or per independent sample. Because the mean cluster size scales with the correlation length, the cost per cluster sweep also scales with $\xi$, but the autocorrelation time in units of sweeps falls far faster than it rises in cost, resulting in a substantial net gain precisely where it is most needed.

The `parallel=True` flag is silently ignored when `update='wolff'`. Cluster growth is a sequential depth-first search whose frontier depends on each newly added site; it cannot be decomposed into independent sublattices and is therefore incompatible with the checkerboard parallelisation strategy. The Wolff algorithm is inherently a single-threaded traversal, and its performance advantage over Metropolis is algorithmic rather than hardware-parallel.


## 5. Statistical Uncertainty Estimation

Monte Carlo time series are not sequences of independent draws. Each configuration is generated by a Markov chain whose proposal depends on the current state, so successive measurements are correlated over a characteristic decorrelation scale set by $\tau_{\mathrm{int}}$. Treating $N$ correlated measurements as $N$ independent samples underestimates the true statistical error by a factor of $\sqrt{2\tau_{\mathrm{int}}}$. Correct uncertainty quantification requires accounting for this serial correlation explicitly. For the foundational treatment, see Sokal [[12]](#Bibliography).

### Blocking Analysis

The standard technique for estimating the error on an autocorrelated mean is the blocking (or "batch means") method. Successive measurements are grouped into blocks of size $b$, and the block means are treated as approximately independent if $b \gg \tau_{\mathrm{int}}$. As $b$ doubles from 2 towards $N/2$, the variance of the block means falls, but more slowly than $1/b$ while the blocks are shorter than the correlation time, so the standard-error estimate $\mathrm{Var}(\bar{x}_{\text{block}})/(N/b)$ rises and then levels off once each block spans several decorrelation times [[14]](#Bibliography). VibeSpin takes the first block size at which two successive estimates agree within 5 %, and the largest block size if none do. The plateau estimate is the autocorrelation-aware standard error on the time-series mean.

For a block containing $\lfloor N/b \rfloor$ independent blocks of $b$ measurements each, the standard error on the mean is

$$\sigma_{\bar{x}} = \sqrt{\frac{\mathrm{Var}(\bar{x}_{\text{block}})}{N/b}},$$

where $\mathrm{Var}(\bar{x}_{\text{block}})$ is the variance of the block means. Under the plateau condition this reduces to $\sigma_{\bar{x}} \approx \sigma_x \sqrt{2\tau_{\mathrm{int}} / N}$, recovering the exact asymptotic formula.

The standard-error estimate is itself a statistic built from $n_b = N/b$ block means, with a relative uncertainty of about $1/\sqrt{2(n_b - 1)}$. For strongly correlated series the plateau often lies at a handful of large blocks, and a Gaussian interval built on so few values undercovers. Confidence intervals therefore use the Student-t quantile with $n_b - 1$ degrees of freedom. On autoregressive test series with $\tau_{\mathrm{int}}$ between 1 and 100 and 2 000 to 20 000 samples, the nominal 68 % interval then covers the true mean in 64 to 71 % of runs, against 54 to 69 % with the Gaussian quantile. When $N$ is not much larger than $100\,\tau_{\mathrm{int}}$ the plateau may not be reached at all, and the error is then underestimated; the wider t interval compensates for part of this, not all.

### Effective Sample Size

The effective sample size $N_\mathrm{eff}$ summarizes how many statistically independent observations the run is worth, accounting for serial correlation:

$$N_\mathrm{eff} = \frac{N}{2\,\tau_{\mathrm{int}}}.$$

$N_\mathrm{eff}$ is bounded above by $N$ and approaches it only when successive measurements are uncorrelated. Near a critical point, $\tau_{\mathrm{int}}$ diverges as a power of $L$ (critical slowing down), so $N_\mathrm{eff} \ll N$ even for long runs. The ratio $N_\mathrm{eff} / N$ is a direct diagnostic of simulation efficiency and guides decisions about run length and algorithm choice.

### Confidence Intervals and Conventions

VibeSpin reports uncertainty intervals at a default confidence level of 68%, the one-sigma Gaussian equivalent. Under this convention the interval $[\bar{x} - \sigma_{\bar{x}},\, \bar{x} + \sigma_{\bar{x}}]$ contains the true mean with probability 0.68 for large samples from a normal distribution; where the error is estimated from few block means or seeds, the half-width is the corresponding Student-t multiple of $\sigma_{\bar{x}}$. This matches the standard physics convention of quoting a single symmetric error bar and is set by the constant `DEFAULT_CONFIDENCE_LEVEL = 0.68` in `utils/statistics.py`.

When a time series has zero variance (a fully ordered configuration, for example), $\tau_{\mathrm{int}}$ and the standard error are both undefined: a constant window says nothing about the fluctuations it did not sample. VibeSpin handles this explicitly: the `ZeroVarianceAutocorrelationError` exception is caught, and the uncertainty fields (`err`, `ci_low`, `ci_high`, `tau_int`, `n_eff`) are stored as `NaN` rather than silently defaulting to zero or propagating exceptions into the aggregation layer. The point estimate itself stays defined, so the susceptibility of a frozen trace is 0 with an undefined error.

### Single-Seed and Multi-Seed Aggregation

For a single seed at fixed temperature, VibeSpin estimates uncertainty directly from the measured trajectory via blocking. This gives a statistically valid error bar even with one seed, provided the time series is long enough to resolve a blocking plateau. For multi-seed sweeps, the uncertainty of the seed mean is the standard error of the seed values, $\sqrt{\mathrm{Var}_{\text{between}}/n_{\text{seeds}}}$, with a Student-t interval on $n_{\text{seeds}} - 1$ degrees of freedom. Independent seeds already scatter by their own statistical noise, so the between-seed spread contains the within-seed blocking error; adding the two would count that noise twice and inflate the error by about $\sqrt{2}$ when the seeds agree. The average within-seed blocking error is still stored as a diagnostic, and a single seed falls back to it.

The entropy is integrated from the specific heat with the trapezoidal rule, so each interior temperature enters the two trapezoids it bounds. Its error contributes once, with half the sum of the adjacent spacings as weight, to every entropy value below it. For multi-seed sweeps the $\tau_{\mathrm{int}}$ interval is the 16th to 84th percentile of the per-seed values, and the `tau_interval_unstable_flag` marks temperatures where that interval is wide compared with $\tau_{\mathrm{int}}$; a single seed has no such interval.

### Regime Caveats

In the deep ordered phase (low $T$), configurations can be nearly frozen. The autocorrelation estimate is then either undefined (zero variance) or non-positive, and $N_\mathrm{eff}$ is set to `NaN` in both cases rather than reporting an artificially inflated value; finite estimates are clipped to $1 \le N_\mathrm{eff} \le N$. Near a continuous phase transition, $\tau_{\mathrm{int}} \propto L^z$ grows rapidly with system size, and the Wolff cluster algorithm is strongly preferred to reduce $z$ from roughly 2.2 to roughly 0.25 and restore statistical efficiency. For derived observables like susceptibility $\chi$ and specific heat $C_v$, which are nonlinear functions of the measured time series (and are therefore not simply the mean of an ergodic chain), VibeSpin supports both an autocorrelation-aware blocking estimator (the default) and an optional block bootstrap over per-block estimates. Each per-block estimate measures fluctuations about its own block mean and is biased low, so the bootstrap interval is taken as the spread of the resampled means around their own centre and placed around the full-series value.

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

[[13]](#Bibliography) Y. Tomita and Y. Okabe, "Probability-changing cluster algorithm for two-dimensional XY and clock models," 2002. [arXiv:cond-mat/0202161](https://arxiv.org/abs/cond-mat/0202161)

[[14]](#Bibliography) H. Flyvbjerg and H. G. Petersen, "Error estimates on averages of correlated data," *Journal of Chemical Physics*, vol. 91, no. 1, pp. 461–466, 1989. [AIP Publishing](https://doi.org/10.1063/1.457480)
