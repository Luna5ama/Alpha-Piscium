# Light Sampling Design

Language: English | [简体中文](../../sc/modules/light-sampling-design.md)

This records Alpha Piscium light-sampling decisions, candidates awaiting validation, and experimental evidence.
The initial design and CPU experiments of 2026-10-29 remain below. Their implementation status and proposed next
steps describe that earlier work; the current decisions below take precedence where they differ.

## Discussion and experiment responsibilities (2026-10-30)

- The current discussion maintains this document and its Chinese counterpart, recording confirmed decisions,
  rationale, hypotheses, and the questions subsequent experiments should answer.
- Experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` owns candidate implementation, runtime captures, timings,
  and image comparisons. Its code stays on `2.0/light-sampling`. A discussed candidate does not automatically
  authorize starting an experiment.
- Experimental results identify the source commit, configuration, scenes, and limitations before informing design
  updates. Record selection, implementation, and validated benefit separately.

Design, experiment sources, and tracked records are consolidated on `2.0/light-sampling`. The migration
preserves experiment commit `5c022ba1` and the later design updates from `2.0/dev`. The runtime entry points
are [LightSampling](../../../shaders/techniques/debug/LightSampling.glsl),
[LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl), and
[option registration](../../../scripts/options.main.kts).

## Current selection and backlog (2026-10-30)

| Item | Confirmed choice |
|------|------------------|
| Conditional emitter sampling | Retain projected-solid-angle cube sampling without repeating its adoption comparison |
| Node weights | Keep the original geometric weights; defer the revised weight candidate |
| Spatial grouping | Retain Morton spatial grouping; Gilbert was rejected and its experimental implementation removed |
| Branching | Default to four-way, retaining binary and eight-way alternatives; all run without geometric lookahead |
| Lookahead | Return to the backlog; revisit only in an explicit quality versus traversal-cost comparison |
| Visibility guidance | Exclude from the current runtime baseline; defer further discussion at the user's request |
| H-PLOC | On hold at the user's request; retain the global cross-brick scope if reopened |

These choices apply to the experimental path; the direct-lighting experiment itself remains disabled by default.
Default settings and lookahead-option removal are in `f9dc71bf`; the five-scene wide-tree experiment record is in
`3e7f66b9`. Four-way branching is the selected tradeoff, not the lowest-variance result: binary two-level lookahead
had lower variance in both reported luminance regions in that round. A 32-frame variance window is not error
acceptance against a high-sample reference.

## Architecture reassessment (2026-10-30)

The [architecture reassessment](light-sampling-reassessment.md) revisits whether per-query light-tree traversal
fits this workload, separating topology, receiver-dependent importance, and shared computation. Following the
user's correction, regional estimates target bounded sets of tree clusters rather than individual lights;
standalone small-candidate RIS is not a scalable replacement, and ReGIR is a complementary direction.
The record expands cluster weights, regional cuts, visibility statistics, and their coupling to topology updates.
Its diagnostic proposals do not accept a replacement, change the baseline above, or start experiments.
Global cross-brick clustering remains the agreed H-PLOC scope.
The current discussion focuses on weight costs and cut semantics; visibility and H-PLOC are deferred.

## Global H-PLOC clustering candidate

This candidate is currently on hold. The following preserves its scope and research context.

The confirmed scope allows global emitter clustering across 16³ brick boundaries. Bricks own emitter extraction,
not the final light-tree grouping. This decision is recorded in experiment commit `5c022ba1`; no H-PLOC builder
has been implemented.

The existing experiment already has one global light tree over the current voxel volume. The binary option
produces a global binary tree; the current default is four-way. Construction first creates a complete subtree
for each brick, then combines brick roots through a fixed Morton hierarchy. Ancestors spanning bricks already
exist, but a local cluster cannot first group only some emitters from two neighboring bricks while leaving
their remaining emitters outside. Here, global clustering removes that complete-brick-subtree constraint; it
does not introduce the first global root. H-PLOC candidate merges remain bounded by Morton ranges and local
neighborhoods.

The candidate for discussion is a globally Morton-ordered emitter list, H-PLOC binary construction, and conversion
to a compact four-way tree for the existing sampler. [Section 3.4 of the H-PLOC paper](https://gpuopen.com/download/HPLOC.pdf#page=8)
already provides wide-tree conversion: repeatedly open the internal child with the largest surface area until
the target child count is reached or all children are leaves. The paper's complete pipeline outputs a four-way
tree, but first constructs a binary tree and then runs a separate conversion kernel. Binary construction and
wide conversion can each use one kernel launch. This is neither native four-way clustering nor a single
dispatch for the whole preparation pipeline. Removing the intermediate binary topology requires further
algorithm changes, rather than setting a branching parameter to four.

The paper's ray-tracing results do not establish lower light-sampling variance here.
A proposed first comparison retains the current emitter set, cube sampling, and
node weights, with visibility guidance and lookahead disabled. This is a discussion starting point, not an
experiment assignment.

Questions to resolve next:

- Merge criterion: start with merged-AABB surface area or introduce a light-sampling objective. The latter also
  changes the topology objective and needs an explicit attribution argument.
- Binary-to-four-way conversion: the paper's largest-surface-area expansion rule is a first-version candidate;
  verify output layout and scheduling when converting temporary binary indices into contiguous children.
- GPU execution: verify global Morton ordering, scratch capacity, subgroup capabilities, and cross-workgroup
  publication. Do not assume the paper's execution mechanisms already exist in the GLSL path.
- Experiment criteria: record all preparation and construction costs, initial-sampling time, raw sample quality,
  and PDF/emitter-set consistency. Comparing only the core construction kernel is insufficient, and tighter
  grouping does not itself establish lower noise.

### Expected sampling-quality impact

There are no H-PLOC sampling-quality measurements yet. It is a topology candidate worth comparing, with a
hypothesized benefit. Current node weights approximate subtree contribution using aggregate emission mass,
the AABB center, bounding-sphere radius, and receiver normal. Clustering could reduce branch-probability error
by avoiding broad bounds across empty space or grouping nearby emitters across brick boundaries separately.
Near-field, sparse, or uneven emitter distributions are plausible beneficiaries, not guaranteed wins for every
receiver.

H-PLOC's area-based merge criterion does not use the receiver, normal, emission mass, or actual visibility.
Smaller area does not guarantee a better contribution estimate, nor does it identify lights separated by a wall.
Current leaves are equal-sized unit cubes and the baseline already uses Morton spatial grouping. The paper's
BVH-quality benefits for differently sized geometry cannot predict a noise-reduction percentage here.
With conditional emitter sampling and actual contribution definitions fixed, and support and PDFs correct,
topology changes affect sampling variance while preserving the target mean.

The existing five-scene spatial-grouping comparison (record commit `181878b9`) was itself mixed: without
lookahead, dark-region variance rose about 14% in scene 1 and fell about 7% in scene 2. This motivates studying
grouping but is not evidence of H-PLOC benefit. Judge the first comparison by sampling error/variance and
contribution tails with the same emitter set, four-way branching, and node weights; then include all build costs
in an equal-time comparison. Tree area, SAH, and visible-hit rate are explanatory diagnostics, not substitutes
for sampling-quality measurements or evidence that H-PLOC can replace lookahead.

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

## Experiment results on 2026-10-29

The first CPU experiment is complete. The degree-2 SH candidate does not show stable benefit under these settings.
This is a statistical result for synthetic scenes, not an Iris/GPU performance result. Keep visibility guidance as
an experimental direction; prioritize the geometric node approximation and bounded lookahead before adopting
this SH representation.

### Protocol and controls

[The sampler experiment](../../../scripts/light_sampling_experiment.py) uses 512 omnidirectional point lights,
256 diffuse receiver queries, opaque AABB occluders, and an all-light reference in each scene. Each pooled training
seed uses 64 independent origin positions and 8,192 equal-solid-angle direction-bin midpoints: 524,288 probes per
scene. The angular grid is deterministic quadrature. Training seeds are 11, 29, and 47; alpha is 0.1 and the bounded
distance scale is 16 blocks. Computation uses float64, Python 3.14.7, and NumPy 2.4.6.

The high-resolution reference retains raw distances within each angular bin and queries their empirical survival
function. A fourth representation retains two moments in those same bins, without SH, to isolate two-moment
approximation from angular compression. All variants use the same training distances, tree, and receiver queries.

Nodes use a flux-weighted center and a sphere enclosing descendant light positions. Visibility is neutral while
radius/distance exceeds 0.5. Compact nodes average five directions over the sphere's angular footprint at the nearest
sphere distance, with modulation strength (1 - radius/distance) squared. These are concrete initial heuristics,
not an established optimal cluster query.

The metric below is the sum of exact single-sample variances across receiver queries, divided by the baseline's
sum. At equal samples per query it is also the expected image MSE ratio. Ranges span the three training seeds,
not confidence intervals over scenes. Construction and lookup costs are excluded.

| Scene | Distance CDF | Binned two moments | SH two moments |
|-------|--------------|--------------------|----------------|
| Unoccluded | 1.003 | 1.003 | 1.000 |
| Two rooms sharing a brick | 1.069–1.094 | 0.987–0.992 | 1.031–1.219 |
| Bright distant light through a doorway | 0.968 | 0.973–0.975 | 0.999–1.005 |

### Diagnosis

[The saved-data diagnostics](../../../scripts/light_sampling_diagnostics.py) separate three sources of loss:

- With probe origins matched exactly to each query, the distance-CDF variance ratio in the two-room scene changes
  to 0.948. Pooling origins is therefore a material source of loss in this fixture. This control is not a proposed
  per-pixel cache.
- Keeping high angular resolution while retaining only two moments is more stable here than projecting the
  moments into degree-2 SH. The result applies to this SH fit, feasible-moment repair, scale, and query rule;
  it does not reject every possible SH or moment representation.
- Summing child scores two levels deeper, with no visibility guidance, gives ratios of 0.577, 0.815, and 0.762
  in the open, two-room, and doorway scenes. Node importance approximation is already a substantial source of
  variance.

With that same two-level lookahead, binned moments give 0.789–0.796 in the two-room scene and 0.691–0.695 in the
doorway scene, an additional reduction relative to the matching geometry-only controls. SH gives 1.712–1.719 in
the doorway scene, roughly 2.25 times the variance of its geometry-only lookahead control. More accurate tree
evaluation can expose errors that coarse-node modulation previously hid.

Complete leaf aggregation is an expensive reference control. With matched origins and a high-resolution distance
CDF, its ratios are 0.075 for two rooms and 0.0069 for the doorway. With exact visibility they are 0.062 and 0.0034.
These controls demonstrate remaining statistical opportunity; they require unavailable cheap knowledge or
all-light work and are not production sampling algorithms.

### Correctness evidence and limits

The pooled and matched-origin runs each execute 160 Monte Carlo checks with 65,536 samples per check. The maximum
absolute mean deviation is 2.35 analytic standard errors. Tree-generated PDFs match enumerated mixture PMFs;
PMFs normalize and retain the alpha-times-baseline support bound. Independently enumerated box-face intersections
agree on 128 sampled visibility connections per scene, with explicit visible and blocked doorway cases. A
two-light mutation using an unsupported conditional PDF produces 2 instead of the correct 11 and is rejected.

The degree-2 SH candidate did not pass the benefit comparison. No shader code or old production path was replaced.
FP32 behavior, dynamic updates, ReSTIR integration, target-Iris compilation, visual acceptance, and equal-time GPU
benefit remain unverified. The next experiment should compare bounded geometric lookahead with a compact positive
directional representation; current results do not justify immediately integrating this SH layout into the pack.

### Reproduction and artifacts

Run from the repository root with the declared dependencies installed, using fresh output directories:

~~~powershell
python scripts/light_sampling_experiment.py --output builds/light-sampling/20261029-pooled-controls
python scripts/light_sampling_experiment.py --output builds/light-sampling/20261029-origin-controls --origin-mode query-matched --training-seeds 11
python scripts/light_sampling_diagnostics.py --inputs builds/light-sampling/20261029-pooled-controls builds/light-sampling/20261029-origin-controls --output builds/light-sampling/20261029-tree-controls
~~~

These commands were executed. The scripts use the lowest process priority on Windows; their elapsed times are not
CPU performance benchmarks. Each run saves source snapshots and hashes, settings, light and occluder inputs,
training samples, full leaf PMFs, and numerical results.

Local generated artifacts are ignored by Git and can be recreated with the commands above:

- [Pooled results and checks](../../../builds/light-sampling/20261029-pooled-controls/results.json).
- [Matched-origin results and checks](../../../builds/light-sampling/20261029-origin-controls/results.json).
- [Tree and leaf-aggregation controls](../../../builds/light-sampling/20261029-tree-controls/diagnostics.json).
- [Pooled variance maps](../../../builds/light-sampling/20261029-pooled-controls/variance-maps.png).

The original three-variant run and the lower-sample smoke run are retained separately under
[local experiment outputs](../../../builds/light-sampling/); they are not silently replaced by the later controls.

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

The next experiment should address the node-approximation and directional-representation losses identified above.
The offline implementation does not establish a runtime sampler, GPU benefit, or visual acceptance.

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
