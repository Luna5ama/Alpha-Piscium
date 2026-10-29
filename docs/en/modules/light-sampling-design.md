# Light Sampling Design

Language: English | [简体中文](../../sc/modules/light-sampling-design.md)

This records the Alpha Piscium light-sampling discussion of 2026-10-29. The agreed direction is ready for a first
algorithm experiment: a compact light BVH combined with directional blocker-distance statistics shared by shading
bricks. It is a candidate design, not an implemented or performance-validated rendering path.

## Scope and decisions

- Target a pure Iris shaderpack. Do not depend on modifying Iris or a companion mod to supply light geometry.
- Focus on NEE light selection and visibility guidance. Model geometry, per-quad storage, UVs, cutout evaluation,
  and exact sampling of a selected emitter's surface are deferred.
- Treat most Minecraft emissive blocks as omnidirectional for importance estimates: point or sphere approximations
  are sufficient initially; a cube is a more accurate later model. Emission cones, emission-direction vMF lobes,
  and position-normal cross-covariance are outside the first experiment.
- Receiver normals, BRDFs, and outgoing directions still affect importance. Omnidirectional emission does not
  remove these terms.
- Separate light selection from conditional sampling on the selected light. A cheap proposal approximation must
  not silently redefine the actual light contribution.
- Use visibility statistics to change branch probabilities first. Additional stochastic subtree pruning is deferred.
- Evaluate ordinary NEE before integrating ReSTIR, temporal reuse, or denoising.

## Data ownership and flow

The light BVH belongs to the emitter population. Visibility statistics belong to the receiver's spatial brick.
They are independent structures; a light-tree node does not own a copy of every receiving brick's statistics.

| Order | Stage | Purpose |
|-------|-------|---------|
| 1 | Light extraction and hierarchy construction | Build an emitter-only hierarchy; extraction details remain deferred |
| 2 | Visibility-statistics preparation | Sample valid shading locations in each active brick and collect directional first-blocker distances |
| 3 | Light-tree traversal | Combine flux, geometric/BRDF importance, and the brick's visibility estimate to choose a leaf |
| 4 | Actual light evaluation | Sample the chosen emitter as required, trace real visibility, and divide by the actual sampling probability |
| 5 | Statistics update | Initially collect and freeze statistics before evaluation; incremental updates are a later experiment |

Existing code anchors are [voxel storage](../../../shaders/techniques/voxel/Voxelization.glsl),
[voxel tracing](../../../shaders/techniques/voxel/VoxelTrace.glsl), the
[current GI estimator contracts](global-illumination.md#estimator-contracts), and
[program registration](../../../scripts/programs.main.kts). These are integration context, not an assertion that this
light sampler already exists. No new pass placement, resource binding, or user setting has been chosen.

## Light BVH

### Initial representation

Start with a compact binary spatial hierarchy containing emissive instances only. A Morton/LBVH construction is a
baseline candidate, not a claim of optimal sampling quality or GPU update cost. Leaves identify lights; internal
nodes aggregate spatial bounds and total flux. Exact packing, construction, refitting, and emitter identifiers
remain implementation decisions.

Do not directly reuse the voxel occupancy hierarchy as the light tree. Its occupancy role and 64-way branching do
not establish an efficient light-sampling layout. Wider branching reduces depth but increases child reads and
importance evaluations.

For an omnidirectional point-light approximation, a cluster's ideal unoccluded scalar importance is proportional to

$$
I_C(x)=\sum_{\ell\in C}
\frac{\Phi_\ell}{4\pi r_\ell^2}
f_r(x,\omega_\ell,\omega_o)\max(n\cdot\omega_\ell,0).
$$

A node stores an approximation to this sum, not its exact value. The scalar treatment of colored flux and the
near-field distance regularization are still to be selected. AABB-plus-flux is the initial simple candidate; spatial
Gaussian summaries remain an alternative if they beat spending the same cost on more detailed traversal.

### Query-dependent refinement

A distant compact cluster can use a cheap aggregate. A nearby or broad cluster, a cluster crossing the receiver
horizon, or one spanning a narrow BRDF lobe may need refinement before normalizing competing branch weights.
Refining only after choosing the wrong parent cannot recover probability already lost at that parent.

A visibility query must account for the cluster's angular footprint and distance range. Evaluating only its center
can suppress a visible descendant behind a doorway or confuse lights before and behind a wall. The initial
heuristic is weak visibility modulation for broad clusters and stronger modulation after they become compact.
The compactness rule and modulation strength are experimental parameters, not settled thresholds.

## Directional blocker-distance statistics

### Meaning and sampling population

For brick B, let D(x, omega) be the distance to the first blocker from position x in direction omega. The desired
quantity is a survival probability

$$
S_B(\omega,r)=P(D\ge r\mid B,\omega).
$$

This probability averages over the chosen origin population and angular filtering. Visibility at one exact origin
and direction is deterministic; the variance describes pooling, not randomness in the scene.

The first candidate uses representative actual NEE shading locations within each 16 x 16 x 16 brick. This is an
initial experimental choice: the user left center versus multiple-origin sampling open.

| Origin population | Meaning and limitation |
|-------------------|------------------------|
| Brick center | Center visibility only; the center can be inside solid geometry or across a wall from a query |
| Multiple valid positions | Visibility averaged over those positions; separate rooms can still be mixed |
| Sampled actual shading locations | Workload-relevant average; depends on the view/path workload and needs updates |

Increasing SH degree cannot recover origin information already discarded by sharing one record across a wall.
For an actual query, direction and distance are measured from its shading position, while the statistics still
represent the brick's pooled population.

### Bounded distance transform and two moments

Use a monotone transform with a positive scale s:

$$
T(D)=\frac{D}{D+s},\qquad T(\infty)=1.
$$

The scale is not a hard distance cutoff. Store directional first and second raw moments of T:

$$
m_1(\omega)=E[T],\qquad m_2(\omega)=E[T^2].
$$

Represent each moment field with low-order SH. Maximum degree 2 gives 9 coefficients per moment: 18 FP32 values,
or 72 bytes per brick, excluding counts, confidence, history, and update storage. Precision and SH degree remain
experimental choices. The transform compresses differences between far distances; the scale therefore affects
useful distance resolution.

At query distance r, reconstruct mu and the variance, then use a Cantelli/VSM-style estimate:

$$
t=T(r),\qquad \mu=m_1,\qquad \sigma^2=m_2-\mu^2,
$$

$$
\widehat S_B(\omega,r)=
\begin{cases}
1,&t\le\mu,\\
\frac{\sigma^2}{\sigma^2+(t-\mu)^2},&t>\mu.
\end{cases}
$$

For true moments of a defined distribution, the expression is an upper bound on its survival probability.
Finite sampling, truncated SH, and pooling over origins do not make it a guaranteed visibility bound for the
current shading point. Two moments do not uniquely determine a distance distribution, its tail, or escape
probability. This is a proposal heuristic; a positive tail is not a substitute for sampling support.

Reconstruction can produce invalid moments. For a variable in [0, 1], feasible moments satisfy

$$
0\le m_1\le1,\qquad m_1^2\le m_2\le m_1.
$$

Any projection into that feasible set is a numerical repair, not restoration of a conservative scene bound.
Zero variance permits a zero tail beyond the mean; the full sampler must still retain required support.

### Trace outcomes and training

A confirmed first hit is a distance observation. An unobstructed trace stopped at a light or finite limit R only
establishes D > R; it is right-censored data, not a hit at R and not proof of D = infinity. Step exhaustion is not
free-space evidence. Exiting the voxel grid only inherits an explicitly chosen boundary model; it does not prove
that the Minecraft world is empty beyond the grid.

For the first experiment, use dedicated uniform-sphere probes from sampled shading locations, build the statistics,
then freeze them and evaluate with separate query samples. Synthetic scenes should define their outside geometry
so that exact hits and true misses are distinguishable.

Existing NEE rays are sampled with a light-dependent angular distribution and often stop at a selected light.
Naively accumulating them into SH changes the intended statistics. Reusing those rays requires an appropriate
sampling-weight or fitting treatment and handling of censored observations; this is deferred.

Later online updates must keep the statistics used to generate a sample consistent with its recorded probability.
Previous-frame read state and separate current accumulation are a possible implementation, not a selected resource
layout. Geometry edits, newly active bricks, workload changes, and history invalidation remain later experiments.

## Sampling probabilities and estimator

Let q0 be the baseline tree distribution with support wherever a light can contribute. Let qv use visibility-modulated
node scores:

$$
w_C=I_C\,\widetilde v_C,\qquad
p_v(C\mid P,x)=\frac{w_C}{\sum_{j\in children(P)}w_j}.
$$

Here the visibility factor includes the chosen treatment of cluster extent and statistical confidence. A branch
with no usable statistics uses neutral visibility. If all child scores vanish, a normalized supported branch rule
must still be defined, using the baseline probabilities. Unknown visibility must not become a proof of zero
contribution.

Each tree's leaf PMF is the product of its branch probabilities. Preserve exploration by mixing the complete
distributions:

$$
q(\ell\mid x)=\alpha q_0(\ell\mid x)+(1-\alpha)q_v(\ell\mid x).
$$

Alpha = 0.1 is a starting experiment value, not a tuned default. This provides

$$
q(\ell\mid x)\ge\alpha q_0(\ell\mid x).
$$

Mixing baseline probabilities separately at each tree level does not provide the same leaf-level guarantee;
its worst-case relative protection can decay with depth. Evaluate the full mixture PMF for the selected leaf,
including both components, rather than dividing only by the probability under the selected technique.

For a discrete point-light experiment with true contribution C_l, use C_l / q(l). When surface sampling is added,
include its conditional density and the appropriate measure conversion. Real visibility and the actual emitter
model belong in C_l; the learned survival estimate belongs only in the proposal. An inaccurate proposal affects
variance without biasing this estimator when support and probabilities are correct. This statement does not by
itself establish correctness of later ReSTIR reuse or of existing trace-termination approximations.

Additional roulette would require dividing surviving contributions by their continuation probability. Pruning
random subtrees and merely renormalizing the survivors is not an established estimator. A shared random distance
cutoff also correlates pruning events; their marginal survival probabilities cannot be multiplied as if independent.
Roulette can save work but increases variance per attempt, so it needs an equal-time evaluation of its own.

## First experiment

Hold the light tree, emitter powers, scene, receiver queries, and training-ray population fixed across variants.

| Variant | Visibility representation | Question |
|---------|---------------------------|----------|
| A | None | Baseline sampling variance and cost |
| B | High-resolution directional and distance histogram per brick | Whether spatially shared distance statistics are useful |
| C | SH first and second moments from the same training samples | Whether the proposed compression preserves that benefit |

B is an expensive experimental reference, not a production feature. If B helps and C does not, investigate
representation loss. If B also fails, inspect origin pooling, cluster queries, and visibility weighting before
increasing SH degree. Freeze training first; do not let each variant's feedback generate a different training set.

Use three initial scenes:

- Unoccluded near and far lights: expose pure overhead and unnecessary variance.
- A wall dividing two rooms inside one 16-cubed brick: expose lost origin dependence.
- A small doorway with a bright distant light: expose over-suppression and extreme contribution weights.

Measure the raw estimator against an all-light sum in the discrete test. Check PMF normalization, convergence
across independent seeds, error at equal sample count, and the tail of contribution weights. A mean within Monte
Carlo uncertainty is evidence, not a proof of unbiasedness. Visibility hit rate is diagnostic: sampling more visible
dim lights can improve it while worsening the lighting estimate.

An offline experiment can establish statistical value. GPU acceptance must measure equal-time error with probe
generation, statistics construction/update, tree traversal, and final shadow traces included. CPU timing does not
establish GLSL/Iris performance. Add movement, geometry changes, cold starts, and ReSTIR only after the static
sampling behavior is understood.

## Deferred alternatives and open choices

- Spatial SG summaries can improve directional BRDF integration but cost more arithmetic and can lose spatial
  anisotropy or multimodality. Compare them with equal-cost bounded lookahead; emission-direction SGs are not
  a priority under the omnidirectional-light assumption.
- Learned split corrections, neural proposals, and ReGIR are later alternatives. They do not remove the need for
  a valid base proposal, explicit probability evaluation, or accounting for update cost.
- For a fixed scalar subtree estimator Z_C and equal branch cost, minimizing second moment gives branch
  probabilities proportional to sqrt(E[Z_C^2]), not necessarily E[Z_C]. This is related to existing variance-aware
  guiding, not a claimed new algorithm, and does not directly transfer to the final ReSTIR estimator.
- Sampling origins, SH degree/precision, distance scale, probe budget, confidence treatment, cluster-footprint
  queries, exploration weight, and update cadence remain experimental choices.
- Emitter extraction, tree packing/construction/update, exact cube/surface sampling, transparency, UVs, and cutout
  treatment remain deferred implementation contracts.

The next development step is the controlled sampling experiment. This record does not report an
implementation, a GPU benchmark, or visual acceptance.

## References

- [PBRT light sampling](https://www.pbr-book.org/4ed/Light_Sources/Light_Sampling):
  hierarchical light selection and leaf probabilities.
- [PBRT Monte Carlo efficiency](https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency):
  mixture sampling, MIS, and roulette.
- [GPU Gems 3 variance shadow maps](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps):
  two-moment visibility estimates and the one-sided variance bound.
- [Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields](https://research.nvidia.com/publication/2019-05_dynamic-diffuse-global-illumination-ray-traced-irradiance-fields):
  related directional distance-moment visibility representation.
- [NEE++](https://diglib.eg.org/items/54459a33-0ee3-4411-8b4c-7bb7cb1612ed):
  visibility caching for direct-light sampling and probabilistic work reduction.
- [Hierarchical Light Sampling with Accurate Spherical Gaussians](https://doi.org/10.1145/3680528.3687647):
  the richer node-summary alternative discussed.
- [Variance-Aware Path Guiding](https://graphics.cg.uni-saarland.de/papers/rath-2020-siggraph-guiding-paper.pdf):
  accounting for the second moment of nested sampling.
