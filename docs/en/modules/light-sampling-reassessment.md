# Light Sampling Architecture Reassessment

Language: English | [简体中文](../../sc/modules/light-sampling-reassessment.md)

Research record, 2026-10-30 through 2026-10-31. This extends the [design record](light-sampling-design.md), following the request to
reconsider light BVHs and alternative sampling methods from first principles. It records research and proposed
comparisons, not an implemented replacement or authorization to start experiments.

## Assessment

The evidence does not rule out light BVHs. It does question the cost and quality of making every shading point
follow one irreversible path through approximate aggregate weights. Three choices are independent: how lights
are grouped, how receiver-dependent importance is estimated, and how that work is shared between queries.
H-PLOC principally changes the first; it cannot settle the other two.

The user's follow-up rejects per-region, per-light tables as too fine and expensive, favoring tree clusters as
the estimation unit. Fixed-budget global-candidate RIS is not a suitable replacement at large light counts;
ReGIR is a complementary sharing/resampling direction that can use a tree-based proposal. The earlier ranking
mixed these different levels. This revision removes the dense-table and standalone-RIS experiment priorities.

The focus is now cluster importance, receiver-region cuts and visibility, and the global clustering that
supports them. Keep the accepted Morton four-way/cube baseline. Global cross-brick H-PLOC remains a candidate,
not an accepted quality improvement. MIS is a separate proposal-combination choice. Full per-leaf scans remain
small diagnostic references, not a second production architecture alongside spatial tables. Experiment startup
and scope follow the user's latest instructions to the designated experiment chat; this discussion maintains
the handoff documents, and temporal-reuse implementation is outside the current task.

Current discussion covers the costs of near/far-distance weights and spatial-moment/SG weights, tree cuts,
shared upper-tree work, and receiver-region choices. At the user's request, visibility discussion is deferred and
H-PLOC is on hold. Their existing sections below are backlog context, not the next scheduled experiments.
The user has now added all three weight variants to the next experiment candidates: near/far-distance mixing,
flux-weighted center and spatial variance with a simple spherical weight, and full diffuse SG weighting.

## What the existing evidence establishes

The experiment branch was inspected at `5c022ba1`, including the geometry record at `4920b6d9`, spatial grouping
record at `181878b9`, and wide-tree record at `3e7f66b9`. These records and experimental shaders belong to
`2.0/light-sampling`, alongside this research record. The original discussion is
`01a18648-f66e-78e0-8d01-cd906ca45429`.

| Condition or observation | Consequence for this reassessment |
|--------------------------|-----------------------------------|
| Pure Iris, software voxel tracing, unit-cube emitters, direct diffuse isolation | Hardware ray-tracing paper timings and glossy-heavy quality wins are not predictions for this path. |
| Five wide-tree scenes contain 3,921 / 2,111 / 2,194 / 908 / 877 emitters | These counts are not an upper bound; design for growth rather than approving O(regions * lights) storage from this sample. |
| Four-way trees have unweighted mean leaf depths of about 8.9–11.0, with many two-child nodes | Four-way capacity does not imply a balanced depth of approximately log4(N), nor establish sampled traversal cost. |
| Cube sampling fixed much of the old bounding-cone rejection | Retain it; do not attribute that improvement to topology or visibility guidance. |
| Lookahead improved quality at extra cost; spatial regrouping had mixed results | Aggregate importance is a plausible bottleneck. These observations do not prove H-PLOC will help. |
| The rejected revised weight changed both emission mass and the geometry formula | It did not isolate either change and cannot reject every improved node-weight model. |
| Earlier images improved over large dark regions while bright areas could regress | Whole-image HDR variance alone can reverse the useful perceptual conclusion. |
| Initial-pass timing includes selection, cube sampling, material work, and tracing | It is not a measurement of BVH traversal alone. Tree construction is separately charged. |

The wide-tree experiment used the original 1920×1080 presets; the geometry suite used 720p presets. Do not
combine their milliseconds into one ranking. The recorded 32-frame variances are not reference MSE. Some
wide-tree cases have incomplete pixels; retain their counts as correctness evidence instead of treating the
shared valid mask as proof that every sample was valid.

The 2026-10-31 review also confirmed out-of-bounds and competing audit writes at 1080p, detailed below.
That round's image-quality and PDF acceptance need corrected captures; its old audit pass is insufficient
correctness evidence. The selected defaults remain unchanged.

Source inspection also matters. The experiment multiplies actual branch probabilities, but has no general
reverse PMF query for an arbitrary emitter. A material ID does not uniquely identify an emitter instance.
After a light is selected, a different first-hit emitter returns zero; the BRDF baseline instead receives
emission from its first hit. This is valid selected-emitter visibility testing, but it explains one difference
between the proposals. Reusing that different emitter with the selected emitter's PDF would be incorrect.

## Review of original scene evidence (2026-10-31)

At the user's request, this review checks original images, statistics, capture settings, and source, and adds
offline numerical reconstruction without starting a GPU experiment. The
[review record](../../../experiments/light-sampling/20261031-evidence-review.json) preserves source hashes,
results, and limits. The [review script](../../../scripts/light_sampling_evidence_review.py) reconstructs all
light-selection probabilities from saved trees and audits. Run it from the repository root to reproduce the
analysis; it requires historical binaries in the ignored build artifacts, not just the Git-tracked summaries.

### Distinguishing aggregate and single-light errors

The numerical review uses the old 720p geometry suite, whose audit addressing is valid: room1, room2,
basilica, gishadow, and tunnel, with 515, 785, 1093, 72, and 1874 lights. It compares binary `cube-plain`
and `cube-ahead`, not a new measurement of the current four-way baseline. At most 64 fixed audit pixels per
scene yield 311 valid receivers and 622 records across the two configurations, with matching positions and
normals. Checks cover manifest hashes, revisions/settings, tree mass conservation, probability normalization,
and the GPU-selected emitter's PMF; maximum relative PMF discrepancy is 0.013%. These receivers are not a
random survey of the scenes.

Construct two references: first normalize **the exact same original leaf formula** across all emitters to
isolate hierarchical aggregation; then analytically integrate each cube's hemisphere-clipped projected solid
angle and multiply by the same leaf mass. The latter is an unoccluded, uniform-emission geometric reference.
It still omits actual visibility, texture emission variation, and conditional-sampling error, so it is not
a converged image reference.

The second-moment factor below is `sum(p_cube² / q)`, with 1 for this geometric reference's ideal distribution.
It is not a measured image-variance ratio. Every entry is a receiver median; ratios of these medians are not
per-receiver or whole-image gains.

| Scene | Tree probability for cubes entirely behind the receiver: original → lookahead | Second moment: original tree | Second moment: flat leaf formula | Second moment: lookahead |
|-------|------------------------------------------------------------------------------|------------------------------|----------------------------------|--------------------------|
| room1 | 11.0% → 6.0% | 1.635 | 1.030 | 1.424 |
| room2 | 32.2% → 2.6% | 1.980 | 1.014 | 1.214 |
| basilica | 17.3% → 2.4% | 2.832 | 1.020 | 1.646 |
| gishadow | 0.12% → 0.05% | 2.136 | 1.075 | 1.197 |
| tunnel | 19.5% → 2.6% | 7.409 | 1.018 | 2.027 |

At most audited receivers, aggregation substantially distorts an otherwise useful per-light geometric
estimate. Back-hemisphere probability is a more direct failure: with `delta = center - receiver` and
`halfSize = (max - min) / 2`, the condition
`dot(normal, delta) + dot(abs(normal), halfSize) <= 0` puts the entire emitter behind the diffuse receiving
plane. Its contribution is zero without making any inference about occlusion. The existing score inflates
cosine by the bounding-sphere radius and retains a positive floor, potentially giving such emitters or
clusters substantial weight.

**Medians must not hide the single-light formula's near-field tail.** In raw-buffer coordinates, gishadow
`(400, 405)` and tunnel `(720, 585)` are counterexamples: the flat leaf formula still assigns 92.7%/95.4%
of its probability to back-hemisphere lights, with second-moment factors 13.70/22.28. Conditioning just the
flat distribution on front-side lights reduces these geometric-reference factors to 1.001/1.030. This is
an offline counterfactual, not a measured tree modification. The original tree assigns 92.6%/27.8% to those
back-side lights, so merely getting closer to the original leaf formula does not always mean improvement.
Tunnel's flat-formula factor has p90 3.03 and maximum 22.28; better aggregation does not replace near-light
and horizon checks.

The depth-wise KL decomposition also argues against always sharing only the top levels. Relative to the
original leaf formula, about 92% of room2's discrepancy occurs in the top three levels, and 74% for tunnel;
room1 has only 19% there, with 59% at depths 9 and 10. Root depth is zero. This decomposes distribution
approximation, not illumination error. Further cut research should refine problematic branches for the
receiver instead of assuming a fixed top cut covers the problem. Whether shared receiver regions preserve
those distinctions remains unmeasured.

### What the images actually lose

Rechecking the [old 720p image comparison](../../../builds/light-sampling/20261029-geometry-matrix/comparison.html)
and linear variances identifies both near-light tails and losses over substantial receiving surfaces, beyond
noise appearance alone. These are existing 32-frame statistics with same-scene BRDF variance equal to 1:

| Scene / region | Original tree + cube | Same tree + two-level lookahead | Interpretation |
|----------------|----------------------|---------------------------------|----------------|
| tunnel floor ROI | 3.660 | 0.508 | Branch estimates alone substantially change surface noise with emitters and conditional sampling fixed |
| tunnel brightest 1% | 50.890 | 10.921 | A large bright tail remains after lookahead; dark-region gains cannot replace near-light acceptance |
| gishadow wall ROI | 0.271 | 0.135 | Light selection already beats BRDF in meaningful regions; this does not reject the whole light-BVH route |
| room2 dark half | 5.732 | 3.537 | It still trails after geometric improvement; current counters do not isolate the remaining causes |

These are sample variances, not reference-image MSE. Room2's short-window means also differ from BRDF by
about +8.8%/+7.8%, insufficient to distinguish rare samples from bias. Existing `visible` counters combine
multiple rejection reasons and cannot establish that all residual trouble is occlusion. For example,
tunnel's original tree accepts the selected emitter for only 3.72% of receivers; that alone does not separate
occlusion, non-emissive hits, proxy-geometry mismatch, or other rejections.

The old visibility matrix actually covers five scenes but uses the old bounding-sphere cone sampler, not
the retained cube sampler. Relative to each matching visibility-off control, no-lookahead variance is about
0.68 times for room1, 1.37–1.43 for room2, and 1.16–1.22 for tunnel; guidance with lookahead makes tunnel
1.47–1.61 times worse. This supports rejecting that old model as a universal improvement, not a claim that
adding visibility fixes the remaining noise or that all visibility methods fail.

### The 1080p audit fault and evidence limits

Historical and current audit addressing in
[LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl) starts at `(80, 45)`,
steps by `(160, 90)`, and indexes 64 records with a row width of 8, without limiting the grid's upper bounds.
The [resource declaration](../../../scripts/shaders.properties) allocates only 4096B. At 720p this produces
64 writers; at 1080p it produces 144, reaches slot 99, creates competing writes in 28 valid slots, and sends
52 writers beyond the allocation.

The check covers historical wide-tree revision `e9a0dfd0`, actual image and buffer sizes, and audit enabled
in all 20 quality captures. The old 1080p PDF audit is therefore unreliable, and image-quality acceptance
requires corrected captures. This does not attribute noise or incomplete pixels to the fault. All 40 timing
captures have audit disabled, so this defect does not directly invalidate their timings. The initial pass
still includes light selection, conditional sampling, and software tracing; its total does not isolate tree
traversal cost. This review records the defect without changing shaders.

### Narrowing the next work

1. Correct audit addressing before new runtime captures, checking unique writers, capacity, and actual
   resolution. Do not reuse the old 1080p pass as evidence.
2. Retain the three queued weight candidates, recording aggregate distributions, back-hemisphere probability,
   and near-light tails separately. Run cheap independent ablations first: fix mass, distance,
   topology, and conditional sampling, changing only exact half-space rejection and cosine estimation.
   The completed prompt now runs this as independent A/B ablations before the three weights; see the
   [experiment prompt](light-sampling-weight-experiment-prompt.md#preliminary-diagnostics-and-independent-ablations)
   for formulas and ordering, without crossing the ablations with those weights. An entirely back-side
   cluster can safely receive zero, but mixed clusters and all-zero child weights need explicit behavior;
   retries or incorrect PDFs must not conceal failures.
3. Judge a cut by whether it refines the branches that actually fail. Fixed top-level sharing has no general
   support here. A more complex SG or a different builder does not replace this diagnosis.
4. Separate remaining rejection reasons, and compare unoccluded and actually visible per-emitter
   contributions/second moments at a few real receivers before prioritizing visibility. The near-light
   bright/dark complementarity also motivates retaining one-sample BRDF/light MIS as a control, with no
   measured benefit claimed yet.

The evidence supports fixing importance estimates and does not establish that light BVHs are unsuitable.
These numbers come from controlled reanalysis of the old binary tree. The current four-way tree needs the
same diagnostics after the audit fix; it cannot inherit these percentages.

## Separate the sampling objective from the tree

For a fixed shading point, let the sampled emitter be l with probability q_l. Let Z_l be its conditional
contribution estimate, including cube sampling and actual visibility, but before division by q_l. Then

$$
\widehat L=\frac{Z_l}{q_l},\qquad
E[\widehat L^2]=\sum_l\frac{E[Z_l^2]}{q_l},\qquad
q_l^*\propto\sqrt{E[Z_l^2]}.
$$

This scalar derivation holds for fixed conditional samplers and positive support; RGB needs an explicit scalar
error objective. Expected contribution is optimal when the conditional estimator has no remaining variance.
For the illustrative model Z_l=A_l V with Bernoulli visibility probability v_l, the optimum is
q_l proportional to A_l sqrt(v_l), rather than automatically A_l v_l. This is not a fitted model of our scenes;
it explains why hit rate or estimated mean visibility alone is not the quality objective. Moments of blocker
distance are not moments of Z_l.

There is also a useful topology test. Suppose a node's importance were exactly the sum of some desired positive
leaf scores a_l(x). Selecting children in proportion to those sums gives

$$
q_l(x)=\prod_{(P\to C)\text{ on path to }l}
\frac{\sum_{j\in C}a_j(x)}{\sum_{j\in P}a_j(x)}
=\frac{a_l(x)}{\sum_j a_j(x)}.
$$

Topology disappears from this probability. The practical problem is that the current AABB/center/radius/flux
score is not additive: a parent can receive a very different score from the sum of its children. Early branch
errors survive even if later levels and cube sampling are accurate. Better grouping, better aggregate weights,
selective refinement, and avoiding aggregate decisions are distinct ways to address that approximation.

## Distinct design dimensions

| Route | What changes | Final visibility rays per query | Main limitation |
|-------|--------------|---------------------------------|-----------------|
| One-sample BRDF/light MIS | Combine complementary proposals with the correct mixture density | One | Needs reverse light PMF and conditional density evaluation; does not make either proposal cheap. |
| Receiver-region tree cut | Cache a bounded partition of the light tree, select a cluster, then a light | One | Per-query cluster scoring and region mismatch; no automatic visibility knowledge. |
| RIS / ReGIR | Resample candidates and optionally share their generation in world space | One in the proposed one-final-sample usage | Depends on source coverage; complementary to a tree, not a direct replacement for it. |
| Improved light tree | Better near-field weights, SG aggregates, or shared refinement | One if kept to one final sample | Still uses aggregate approximations; some published splitting methods actually trace multiple lights. |

### One-sample MIS: a useful control before replacing the structure

Choose the BRDF or light technique randomly, generate one sample, and divide by the full mixture density in a
common emitter-surface measure. With technique probability alpha, this is
q_mix=alpha*q_BRDF+(1-alpha)*q_light. A BRDF ray that hits no emitter is a zero sample. No second shadow ray is
required. This is the one-sample use of [MIS](https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency#MultipleImportanceSampling),
not averaging two separately rendered images.

Our evidence motivates this control: BRDF sampling can cover a nearby emitter's large angular footprint,
whereas light selection can reach small distant emitters that BRDF sampling rarely finds. It is not guaranteed
to beat whichever pure technique is best at every pixel. The current experiment lacks the inverse queries:
identify a BRDF-hit cube, evaluate its tree PMF at the receiver, and evaluate the cube sampler's conditional
density for that hit. Area and solid-angle PDFs must be converted consistently, including the cube sampler's
small-projection branch. [PBRT's light sampler interface](https://www.pbr-book.org/4ed/Light_Sources/Light_Sampling)
is a useful reference for the reverse-PMF requirement.

### Correcting the table and resampling classification

A dense regional distribution needs O(RN) entries for R receiving regions and N lights. Merely replacing each
light with every internal tree node still needs O(RN) storage. The useful change is a bounded selection of
clusters per region, not a different name for the entries. The earlier full-leaf diagnostic and proposed spatial
table also shared the same underlying scoring idea: one was a per-point reference, the other a cached regional
approximation. They should not have been presented as independent algorithmic directions. MIS remains separate.

For M independent uniform candidates among N lights, the chance of including one particular important light
is 1-(1-1/N)^M, approximately M/N for small M/N. Resampling cannot select a missing candidate. This invalidates
fixed-small-M global RIS as our scalable replacement; it does not establish a universal light-count limit for
the RIS operator. A good tree or cluster proposal can supply its candidates.

[ReGIR](https://cwyman.org/papers/rtg2-manyLightReGIR.pdf) shares candidate preparation in world space and reweights
at actual receivers. It can be combined with that proposal, independently of choosing Morton or H-PLOC topology.
The [RTXDI integration guide](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md) distinguishes this
machinery from screen-space ReSTIR. Coverage, correlation, and update costs still matter; the original chapter
also reports slight bias. If this separate direction is pursued, derive the estimator weight rather than treating
a reservoir weight as an exact PDF; [GRIS](https://research.nvidia.com/labs/rtr/publication/lin2022generalized/)
provides the relevant framework. It is not a prerequisite for the cluster-weight research below.

## Receiver regions and tree clusters

### Share a bounded cut, with complete coverage

A tree cut is a frontier of nodes: every root-to-leaf path crosses exactly one selected node. The nodes need
not be at the same depth. For example, if root children are A and B and A is expanded into A1 and A2, then
{A1, A2, B} is a cut. It partitions the leaves without modifying or pruning the global tree.

| Term | Meaning here |
|------|--------------|
| Tree cut / light cut | The node frontier or light-cluster partition itself. |
| [Original Lightcuts](https://www.graphics.cornell.edu/~bjw/lightcuts.pdf) | An illumination algorithm that chooses a cut using an error criterion and sums representative-light approximations for its clusters. |
| Stochastic Lightcuts | Uses randomized light representatives and probability correction inside cut clusters; the referenced algorithm still evaluates multiple cluster samples. |
| Regional cut used as a proposal | Select one cluster from the cut, then one light from it, and use the full probability. The cut is not evaluated as a sum of representative lighting estimates. |

For each active receiving region R, keep a cut K_R through the global light tree. Its nodes are disjoint subtrees
whose leaves cover the complete emitter set. Nearby or difficult clusters may be split deeply, while distant
compact clusters stay coarse. Reaching a size budget stops further splitting; it does not drop the remaining
lights. Storing only the most important clusters would instead require a separate full-support proposal.

With at most K clusters per region, regional storage is O(RK), in addition to the O(N) global tree. K is a memory
and quality budget, not a scene-independent accuracy guarantee. A cut must not contain both a node and its
descendant. Store references to global nodes, rather than copying their bounds and light populations per region.

| Order | Stage | Purpose |
|-------|-------|---------|
| 1 | Region preparation | Select a bounded cut from the current tree, using valid representative shading locations; share the partition and receiver-independent terms. |
| 2 | Cluster selection | At the actual point and normal, score the cut nodes and normalize their probabilities. |
| 3 | Conditional light selection | Select a leaf inside the chosen cluster using a defined conditional sampler. |
| 4 | Conditional cube sampling and tracing | Use the retained cube sampler, trace the true visibility once, and divide by the full sampling probability. |

$$
q_l(x)=P(C\mid x,R)P(l\mid C,x),\qquad
p(l,y\mid x)=q_l(x)p(y\mid l,x).
$$

[Adaptive Direct Illumination Sampling (2016)](https://cgg.mff.cuni.cz/~jaroslav/papers/2016-directillum/2016-vevoda-directillum-abstract.pdf)
already uses regional cuts, query-dependent cluster probabilities, and power-proportional light selection
inside the selected cluster. It samples one light; this is a closer precedent than adapting a multi-light
lightcut estimator. Our variants could retain geometric traversal inside the chosen subtree, or use power-only
selection if the cut resolves the important geometric variation. The latter saves importance evaluations but
may lose quality within broad clusters; it needs a separate comparison.

Cut sharing removes repeated partition search, not necessarily repeated probability evaluation. Recomputing K
weights per point plus a subtree traversal can cost more than the present root-to-leaf traversal. Sharing a
single scalar PMF instead would save that work but reintroduce position/normal mismatch. Do not conflate these
two tradeoffs. A receiver region is not automatically one 16³ extraction brick, and a point inside solid geometry
is not a representative receiver. Which regions to use and the cut budget remain design questions.

It is reasonable to describe this as sharing the result of expanding the upper tree into a frontier. It does
not share a single randomly chosen upper branch across all points. Each point still evaluates or obtains its
own cluster probabilities and independently selects a cluster. Directly normalizing frontier-node weights
also changes the approximation: it need not reproduce the product of the original upper-level probabilities,
because aggregate weights are not additive.

A cost counterexample is a full four-way tree expanded by two levels. Ordinary single-path sampling evaluates
4+4=8 child weights in those levels; scoring all 16 frontier clusters takes 16 weights. A shared cut may improve
importance estimates and remove dependent traversal steps, but it does not automatically reduce arithmetic or
node reads. Compared with per-query adaptive cut construction, the partition search is amortized; compared
with the current single-path baseline, region preparation is additional work. If every region uses the same
cut, store that cut once rather than duplicate it per region.

### Sharing cuts in world space and screen space

This round discusses both region types and both visible-surface and indirect-hit queries. The user has not
restricted the consumers or selected an implementation. Regions group receivers; the global light tree and
its emitter clusters remain shared. Each region stores a complete cut, and each receiver independently picks
one cluster, one light, and one conditional emitter sample. K frontier nodes do not mean K shadow rays per pixel.

Both arrangements have precedents. Section 3.3 of
[Real-Time Stochastic Lightcuts](https://www.cemyuksel.com/research/stochasticlightcuts/realtime_stochastic_lightcuts.pdf)
shares a cut across a pixel block and samples independently per pixel. Its multiple-cluster samples and
interleaved reconstruction are not evidence of a gain in our single-light, unfiltered path.
[Adaptive Direct Illumination Sampling](https://cgg.mff.cuni.cz/~jaroslav/papers/2016-directillum/2016-vevoda-directillum-abstract.pdf)
caches cuts on demand in world-grid cells, evaluates cluster probabilities at the actual receiver, and picks
one cluster and one light. Its queries can occur at different path vertices.

| Property | Screen-space regions | World-space regions |
|----------|----------------------|---------------------|
| Main reuse | Nearby visible pixels, usually with contiguous G-buffer access | Spatially nearby actual hits, potentially from different pixels or bounces |
| Natural advantage | Work scales with active screen queries; nearby threads can read common cut data | Partition does not depend on camera projection; can organize off-screen and indirect receivers |
| Main mismatch | Depth discontinuities, silhouettes, and differently oriented surfaces in one tile | A cell crossing walls, containing multiple surfaces, or spanning too large an area |
| Refinement | Group by depth, world-position extent, and normals, rather than pixel distance alone | Group actual surfaces and normals within active cells; do not necessarily cover empty volume |
| Updates and amortization | Straightforward per-frame rebuild; motion and disocclusion change membership | Potential reuse, requiring active-region management and current tree identity and lifetime |

The goal is to group receivers with similar importance distributions. Screen proximity does not guarantee
world proximity, and world proximity does not guarantee similar normals or visible emitters. Points on one
flat wall may share well, while nearby points around a corridor corner may need different refinement. With
visibility still deferred, consider at least position extent, normal extent, and relative distance to important
light clusters. Region-size-to-light-distance ratios can suggest near-field variation, but alone neither
guarantee distribution similarity nor establish a chosen threshold.

A compromise candidate is to form a few receiver groups inside each screen tile using world positions and
normals, with one cut per group. This retains screen scheduling and contiguous access while reducing mixtures
across silhouettes and surfaces. Grouping, membership indices, and additional cuts cost work; the benefit is
unmeasured. For indirect lighting, do not assume that a ray can use its originating pixel tile's cut: nearby
pixels can send rays to unrelated locations. Regrouping actual secondary hits entails hit storage, binning,
and subsequent scheduling. Sparse queries with few samples per region may not amortize that work.

### Sharing node lists and sharing probabilities are distinct candidates

| Shared data | Work remaining per receiver | Main tradeoff |
|-------------|-----------------------------|---------------|
| Cut node references and reusable node data | Score K clusters at the actual position and normal, normalize, then select a light in the chosen subtree | Can change coarse-cluster estimation error and expose common reads, but need not reduce weight evaluations |
| Cut plus a regional cluster PMF/CDF | Draw a cluster from the shared distribution, then use a defined conditional light sampler | Removes K per-query cluster scores; within-region mismatch can increase variance and contribution tails |

With shared probabilities, the actual sampling density is

$$
p(l,y\mid x,R)=P_R(C)P(l\mid C,x)p(y\mid l,x).
$$

Complete contribution support and the actual probabilities can preserve the Monte Carlo estimator's
unbiasedness; they do not guarantee low variance. Continuing geometric traversal within a chosen cluster
improves only its conditional allocation and cannot repair a severely underestimated regional cluster PMF.
Do not share the randomly selected cluster itself or copy representative-point lighting across the region.
Shared structure does not require shared random numbers. This round does not adopt interleaved reconstruction
that assigns fixed cluster subsets to pixels and recovers lighting through neighborhood filtering.

Cut construction should consider more than which cluster is brightest. Geometric variation over the region,
uncertainty in coarse-node estimates, and refinement cost also affect allocation. One representative point is
cheap but can miss the needs of small receiver surfaces. Several representatives or regional bounds cover
more variation while adding preparation cost or excessive refinement. These are construction strategies to
compare, not inherited original-Lightcuts illumination error guarantees. Keep coarse clusters when the budget
is exhausted; do not discard the remaining emitters.

As a counting example, a full four-way tree with 4,096 leaves evaluates 24 child weights across six levels for
one root-to-leaf sample. Sharing its 16-node depth-two cut while scoring it per query requires 16 cluster
weights plus 16 weights across the remaining four levels: 32 total, before preparation. Sharing the cluster
PMF instead leaves 16 geometric weights, plus CDF lookup, subtree reads, and amortized cut preparation. This is
not a measurement of our irregular tree. Common node reads, caching, parallel scoring, and dependency-chain
length can change GPU time; evaluation counts do not replace timing.

Persistent regional storage need not copy full 32B nodes. With K=16 and a 4B node index plus 4B CDF value per
entry, each region needs 128B. At 720p, 8×8 tiles give 14,400 regions and about 1.76 MiB; 16×16 tiles give 3,600
regions and about 0.44 MiB. Index-only cuts halve those amounts. Fully populated 32³/64³ world-cell grids need
4/32 MiB respectively. These are entry payloads only, excluding region descriptors, counts, membership maps,
alignment, and multiple buffers. Active-region storage avoids empty-volume waste but still needs indexing
and allocation.

Two current source constraints matter:

- [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl) uses 16×16 workgroups
  and obtains visible-surface receivers from the G-buffer through
  [LightSampling](../../../shaders/techniques/debug/LightSampling.glsl). This supplies a screen-oriented entry
  point, not proof that 16×16 is the best sharing size. A separate preparation stage can produce cuts for the
  sampler; same-workgroup shared memory is another execution choice with synchronization and occupancy costs.
- [LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl) clears and rebuilds the tree,
  allocating local node ranges atomically. Stable world-cell identity therefore does not imply stable cut-node
  indices. Updating cuts against the current tree is the direct option; reuse across frames requires emitter
  membership identity, tree-update, and invalidation contracts before reusing old indices.

The recommendation remains conditional: dense visible-surface queries favor investigating geometrically
grouped screen cuts; arbitrary bounce points favor active world/surface regions. If a region mixes many
dissimilar receivers, or queries cannot amortize preparation, per-query light-tree traversal remains an
essential control. Judge shared cuts and shared PMFs separately, comparing equal-sample quality and total
cost including region preparation. This discussion adds no cut variant to the scheduled three-weight
experiment and establishes no new runtime acceptance result.

### Better cluster importance without changing the emitter mass

First isolate emission mass from geometric weighting. For the current diffuse scope, the desired unoccluded
cluster score approximates a sum of emitter contributions, including their angular extent and receiver cosine.
The present center/radius score can assign a large weight to empty portions of a bound or poorly represent a
cluster crossing the receiver horizon. This affects cut probabilities as well as ordinary tree branching.

Two different levels of improvement are worth distinguishing: a cheap correction to near-field distance
weighting, and a richer representation of where the emitting energy actually lies. For the latter, global nodes
could aggregate flux-weighted position and spread:

$$
\Phi_C=\sum_l\Phi_l,\qquad
\mu_C=\frac{\sum_l\Phi_l\mu_l}{\Phi_C},\qquad
s_C^2=\frac{\sum_l\Phi_l(s_l^2+\|\mu_l-\mu_C\|^2)}{\Phi_C}.
$$

Include each cube's finite extent in s_l². These moments are shared globally and can distinguish a thin energetic
group from a mostly empty AABB. They still collapse multimodal or elongated arrangements; using them to estimate
illumination requires a model, such as an SG approximation, with verified support and numerical behavior. They
are not the blocker-distance moments used by the previous visibility field. More stored statistics alone do not
establish a better estimator or an acceptable node size.

[Real-Time Stochastic Lightcuts](https://www.cemyuksel.com/research/stochasticlightcuts/realtime_stochastic_lightcuts-preprint.pdf)
offers a near-field heuristic that averages two separately normalized branch distributions based on minimum
and maximum cluster distances. It is not an average of the raw weights. A controlled test could hold emission
mass fixed and compare this with the original geometric score. Its published binary rule needs an explicit
multi-child extension and zero-distance handling before use in our four-way sampler.

The same work shares cut selection between pixels, but its lightcut estimator uses multiple representative
lights. [Adaptive Tree Splitting](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf) likewise creates
multiple light samples. Keep these separate from the single-light regional-cut precedent above.

[The 2024 SG light-tree paper](https://yusuketokuyoshi.com/papers/2024/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting.pdf)
directly addresses inaccurate node importance with one light per traversal. Omnidirectional emitters do not
make the idea irrelevant: their spatial distribution still creates a directional incident-radiance lobe at a
receiver. The emission-direction model could simplify, while the spatial model remains useful. However, the
paper's strongest results concern glossy materials; diffuse-dominant results are more modest. It omits shadow
visibility and reports mathematical-function and register overhead. Its uncompressed cluster summary is ten
floats, so “more compact” than its own reference is not evidence of being smaller than our complete 32B node.
Treat a diffuse-only adaptation as a candidate, not a high-confidence upgrade, and consult the
[published errata](https://gpuopen.com/download/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting_(errata).pdf).

### Cost comparison against the current node and traversal

These are source-level storage and operation estimates, not GPU timings. The current experimental node is
two vec4 values, 32B including AABB, mass, and packed link. Its geometric weight uses center distance, a radius,
receiver cosine, and a distance denominator. Sampling already reads each evaluated child once.

| Candidate | Node data with the current AABB retained | Preparation | Work for each evaluated child or sibling set |
|-----------|------------------------------------------|-------------|---------------------------------------------|
| Near/far-distance probability mixture | Same 32B; no new statistics | No extra node reduction | Compute squared distances to the closest and farthest AABB points; retain two weight sums and normalize the two distributions before mixing. Same visited children and tree depth. |
| Flux-weighted center/spread with a simple spherical weight | Add a vec4 for center xyz and scalar spread: straightforward layout 48B | Extra weighted-moment reduction, still O(node count) | A simpler geometric model need not evaluate SG integrals. Exact ALU cost depends on the selected formula and cannot yet be ranked against the distance mixture. |
| Spatial moments with diffuse SG integration | Same spatial summary as the spherical model; no need to add glossy BRDF machinery for this experiment | Same moment reduction | Compute incident-lobe parameters and approximate its clamped-cosine integral, including exponential functions, an error-function approximation, rational expressions, and numerical limit handling. More intermediate values remain live. |

For AABB center c, half-size h, and a=abs(x-c), the two squared distances are

$$
d_{\min}^2=\|\max(a-h,0)\|^2,\qquad
d_{\max}^2=\|a+h\|^2.
$$

They require no extra distance square roots. Keeping the current cosine estimate isolates the distance change;
the additional work is abs/max/add/dot operations, a second distance reciprocal, and a second sibling
normalization. Four-way selection can retain two four-element weight vectors and their totals, then combine
the normalized probabilities. The actual register allocation is compiler-dependent. Zero minimum distances
need the explicitly defined multi-child rule discussed above. This is extra arithmetic on already fetched
nodes, not another traversal or lookahead.

Appending center/spread increases the straightforward node's nominal storage and full-node payload by 50%,
not necessarily memory transactions or pass time by 50%. It is not a fundamental SG size requirement. Under
the omnidirectional approximation, replacing the AABB with center, radius, spread, flux, and link could fit a
different 32B layout; that would require updating the bound-dependent consumers and validating the replacement.
No such layout is implemented or selected. The uncompressed full-paper SG summary also includes emission
direction parameters that this approximation may omit.

Spatial moments and SG integration must therefore be evaluated separately. The former mainly changes node
representation and reduction; the latter adds query arithmetic, special-function work, and register pressure.
The paper's diffuse/glossy rendering timings do not predict our diffuse-only sampler or isolate its traversal
cost. The near/far mixture is the smallest implementation change to compare first; a concrete spherical or SG
formula is required before estimating a meaningful runtime multiplier.

### Visibility: learn receiver-region / cluster behavior

Deferred at the user's request. The following retains the previously discussed direction.

The relevant cache key is a receiving region plus a selected cut cluster. It represents illumination sampled
from that cluster, rather than a universal blocker-distance distribution for the entire receiving brick.
For our candidate, begin with unoccluded cluster geometry and investigate a learned visibility/contribution
correction. Continue evaluating position and normal effects at the query; cached darkening across a wall must
not be treated as a certificate that every point in the region is occluded.

For a fixed conditional light sampler r_l=P(l|C,x), define Y_C=Z_l/r_l. The final estimate is Y_C/P(C|x,R), so

$$
P^*(C\mid x)\propto\sqrt{E[Y_C^2]}.
$$

Thus a visibility hit fraction alone is not generally the best cluster-selection weight. The conditional PMF
and cube PDF matter when collecting contribution statistics; averaging raw hit brightness does not estimate
the cluster's contribution. Statistics collected only when a cluster is chosen also reflect the receiver
positions favored by that proposal, not automatically a uniform average over the region.

[Bayesian Online Regression for Adaptive Direct Illumination Sampling (2018)](https://cgg.mff.cuni.cz/~jaroslav/papers/2018-bayesianlighting/2018-vevoda-bayesianlighting-paper.pdf)
directly learns region/cluster statistics and derives the second-moment objective. It separates occlusion from
visible contribution, models distance variation, and uses an unoccluded prior when data are sparse. Its CPU
implementation amortizes distribution construction over 16 samples and explicitly flags the cost without
splitting. This is a model reference, not evidence of affordable one-sample GPU execution here.

For this project, first validate frozen statistics from representative actual receivers, then consider online
updates. Preserve nonzero proposal support, retain the final true visibility ray, and consume a published
statistics snapshot when sampling. A sample must not retroactively change its own probability. Initially this
does not require a neural model, control variates, or the paper's full Bayesian fitting machinery.

Splitting light clusters cannot repair a receiver region that combines two rooms. Conversely, refining receiver
regions cannot distinguish visible and hidden lights inside an unresolved cluster. These are separate spatial
resolution budgets; increasing SH order alone repairs neither lost distinction.

### H-PLOC: provide clusters and account for cache stability

On hold at the user's request. The global cross-brick scope remains applicable if this work is reopened.

H-PLOC changes topology and build scheduling, with merged-AABB area as its criterion. Its
[ray-tracing evidence](https://gpuopen.com/download/HPLOC.pdf) cannot rank it above the receiver-aware alternatives
for this sampling workload. If retained in experiments, preserve the agreed global cross-brick scope and
separate construction, conversion, and sampling costs.

Its relevant hypothesis here is that tighter cross-brick clusters make the same importance model or fixed-size
regional cut more accurate. A light-sampling construction objective would instead care about contribution
variation across receivers and the remaining conditional variance. Merely multiplying SAH by flux does not
prove that objective is optimized. Receiver-conditioned cuts can express local grouping resolution while the
global hierarchy stays receiver-independent.

Topology and learned statistics also couple through identity. A rebuilt node index can denote a different light
set; retaining its old regional statistics would attach observations to the wrong cluster. Stable-topology refit
can preserve membership, whereas topology changes require cut regeneration and deliberate invalidation or
mapping. No remapping mechanism is selected here. A faster builder that constantly destroys learned context
may lose overall efficiency, so evaluate rebuild cadence and statistics lifetime alongside geometric quality.

## Directions not prioritized in the next comparison

| Direction | Why it does not resolve the immediate question |
|-----------|-----------------------------------------------|
| More SH coefficients for the existing brick visibility field | They cannot recover shading-origin information lost by mixing positions across walls. |
| Dense per-light tables or neural visibility | Dense tables violate the selected estimation granularity. [Neural Visibility Cache](https://jcgt.org/published/0014/02/01/paper-lowres.pdf) supports learned proposal visibility, but does not establish that online neural training is affordable here; simple cluster statistics remain a different candidate. |
| Full screen-space ReSTIR or denoising | Relevant later, but would change the initial-sampling experiment and confound its diagnosis. |
| Merging adjacent emissive faces into larger lights | Could reduce N, but changes emitter identity, conditional sampling, and textured-emission handling. It exceeds the current cube-list comparison. |
| STBN or low-discrepancy random-number remapping | A separate noise-distribution experiment; it does not repair an incorrect importance estimate or support. |

## Proposed experiments and decision rules

The user has confirmed the three weight variants below for inclusion in the next experiment candidates.
This records the comparison scope, not a default change, validated benefit, or instruction to run experiments
in this discussion. The prompt now includes audit repair, current four-way reconstruction, half-space/cosine
A/B ablations, classified rejections, and necessary fixed-receiver references. Shared-cut and MIS implementation
remain later candidates; visibility guidance and H-PLOC remain deferred.

### Weight candidates for the next experiments

The [experiment prompt](light-sampling-weight-experiment-prompt.md) packages the verified wt4 entry points,
remaining formula work, comparison controls, and delivery criteria for the experiment session.

| Candidate | Comparison scope | Details to settle before implementation |
|-----------|------------------|-----------------------------------------|
| Near/far-distance probability mixture | Retain the current 32B AABB node; compute and normalize near/far weights separately, then mix the probabilities | Four-child normalization and zero-minimum-distance behavior |
| Flux-weighted center and spatial variance with a simple spherical weight | Add shared global spatial moments; use a geometric spherical estimate without SG integration. Retaining the AABB gives the straightforward 48B layout described above | Concrete spherical-weight formula, finite cube extent, near-field behavior, and support |
| Full diffuse SG weight | Use the spatial summary to construct an incident SG and evaluate its diffuse clamped-cosine integral | Diffuse-only formula, numerical limits, support, and the paper's errata |

Compare all three with the original geometric weight. Hold the emitter set, emission mass, Morton four-way
topology, cube sampling, and actual contribution definition fixed; keep geometric lookahead and visibility
guidance disabled. Do not combine the weight comparison with a regional cut or ReGIR change.

For attribution, the spherical and SG candidates should use the same spatial moments and node layout, so their
difference measures the weighting model and its query cost. A redesigned compact 32B spherical layout is a
separate optimization, not part of this first comparison. Record the actual node bytes and all construction
costs alongside sampling cost and quality; the common acceptance criteria below apply.

The broader diagnostic comparisons remain:

| Order | Comparison | Question answered |
|-------|------------|-------------------|
| 1 | On selected real receivers, compare the current tree PMF with normalized sums of the exact same per-leaf score; then with per-cube unoccluded contribution estimates | Is the error primarily aggregate non-additivity, the leaf score, or something not modeled by unoccluded weights? Full scans are diagnostic, not proposed per-pixel production work. |
| 2 | Fixed topology and emission mass; original versus all three weight candidates listed above | Does each cluster importance model improve enough to pay for its storage, construction, and query cost? |
| 3 | Fixed weights and topology; per-query tree traversal versus a bounded regional cut | Does exposing appropriate clusters before selection justify partition storage and query scoring? |
| Deferred: visibility | Fixed cut and conditional sampler; frozen region/cluster visibility-contribution statistics versus unoccluded weights | Is remaining variance driven by occlusion, and does the receiving-region resolution preserve useful information? |
| Deferred: H-PLOC | Same sampling policy; Morton versus global H-PLOC, including cut rebuilding and statistics invalidation | Does improved global grouping help after accounting for its complete downstream cost? |

One-sample BRDF/light MIS is an orthogonal comparison once reverse probabilities can be evaluated. ReGIR may
later consume a chosen tree/cluster proposal; standalone small-candidate RIS is not a prerequisite experiment.

For a small diagnostic receiver set, conditional per-light samples with true visibility can also estimate
E[Z_l²]. Comparing that target with the unoccluded references separates visibility/conditional-sampling limits
from hierarchy error. Do not infer that distinction from visible-hit rate alone.

Use the following decision rules rather than treating the tree as a commitment:

- If exact leaf-score sums are much better than the hierarchy, the present aggregate estimator is the main
  suspect. Better cluster weights and regional refinement address it; this does not prove trees fail.
- If unoccluded per-cube selection is already good and topology changes are minor, H-PLOC has limited room to
  improve sampling quality. Cost or shared work becomes the more relevant comparison.
- If visibility-aware references dominate all unoccluded proposals, changing tree topology or unoccluded
  resampling alone will not remove the main source of variance. Revisit receiver-conditioned visibility.
- If a regional cut wins at equal total cost, replace repeated upper-tree decisions with that cut. Keep the
  chosen conditional sampler inside each cluster; neither tree removal nor RIS follows automatically.
- If active-cell counts or dynamic updates destroy amortization, keep a per-query hierarchy as a serious
  contender. Representative counts from five scenes are insufficient to assume regional caches always fit.

Acceptance needs both 1-spp and 32-frame outputs, separate dark-region and near-emitter errors, contribution
tails, and low-frequency spatial error against a sufficiently converged reference for representative crops.
Keep raw mean-energy and support/PDF checks separate from perceptual metrics. Report nonfinite or incomplete
samples explicitly. Compare equal sample counts first and then equal time, charging all light extraction,
table/reservoir/tree preparation, sampling, and visibility work that the method actually requires. Future
measurements should use matched scene/resolution settings, normally 720p, rather than mixing the old suites.

## Integration boundary

The production [GI initial sampler](../../../shaders/techniques/gi/InitialSample.glsl) currently passes a directional
PDF and mixes directional proposals. The isolated experiment has its own estimator. A RIS contribution weight
cannot simply be written into the production PDF field. Integration needs an explicit weighted-candidate
contract and compatible MIS/reuse handling; it follows the initial-sampling comparison, not the reverse.

Any shared pool also needs stable emitter identity and a defined publication lifetime. Atomic brick allocation
order is not identity; a refreshed light list must not reinterpret a cached index as another cube. Initially
rebuild from the same-frame list and publish before queries. Temporal reuse would require deliberate mapping
or invalidation. This is an ownership requirement, not a reason to introduce a general cache framework now.

This round updates documentation and adds an offline review script and evidence record. No runtime shader,
setting, emitter representation, or sampling default was changed; no old runtime code was removed and no new
GPU captures or timings were made.


## Weight experiment formula contract (2026-10-31)

Implementation preparation, not measured acceptance. Model IDs: 0 original, 1 half-space A, 2 cosine B,
3 near/far probability mixture, 4 moment sphere, 5 diffuse SG. Leaf emission mass and the target contribution stay fixed.
A/B retain the specified formulas; classify a box as behind only when h+e is below a conservative floating-point
margin 1e-6*(dot(abs(normal),abs(delta))+e+1). An all-zero child sum returns a zero sample without retry; audit includes its probability.
For model 3, use the original cosine bound and two separately normalized mass*cosine/distance-squared distributions,
then average their probabilities. Zero minimum-distance children share the near distribution in proportion to mass*cosine;
all other children receive zero near probability. The far distribution remains positive.
Models 4/5 append mean.xyz and trace spatial variance to the AABB node (48B; models 0..3 remain 32B).
Leaves use uniform area measure over all six unit-cube faces: mean=low+0.5, variance=5/12.
Merge by emission mass using the parallel-variance formula in this record. Model 4 substitutes the weighted mean
and effective radius sqrt(2*variance) into the original cosine-bound/distance model, retaining its 0.25 distance floor.
Model 5 uses the paper's omnidirectional limit: k=distance^2/variance, axis=(mean-x)/distance,
weight proportional to mass/variance times the cosine-weighted SG integral B/(2*pi).
Use the Section 4 endpoint interpolation and the corrected t(k) rational fit from the errata; Taylor endpoint
integrals handle k<0.05. At zero distance the SG is isotropic. An explicit lower bound
B/(2*pi)>=1e-6/(1+k) preserves numerical support at sharpness/underflow extremes; this is part of the proposal,
not a modification of evaluated radiance. Actual child probabilities and their full path product are used by the estimator.
Independent hemisphere quadrature must quantify the SG approximation error before GPU sampling acceptance.
The audit uses 8x8 unique bin-center pixels at each actual resolution; 64 records plus 16 uint diagnostic counters.
Counters 0..9 are exclusive terminal outcomes; counter 10 separately counts nonfinite final RGB values.

The first target-Iris moment audit failed in basilica (node 9210: stored variance 3.5045831, child-derived 3.4910553). The follow-up copies empty merges exactly, uses both positive mass fractions directly, and anchors the mean at the heavier child. This avoids amplifying reciprocal/subtraction error by absolute world-position squared; audit tolerances stay unchanged. GPU revalidation is required; the failed capture remains evidence, not an accepted candidate.

The half-space margin also bounds the absolute products inside the normal/delta dot product, not only its possibly cancelled result. A float32 counterexample had true front extent +4.364748e-6 but computed -2.503395e-6; the earlier result-based margin could falsely reject it. The corrected product-sum margin retains support. A/B captures and timings use the corrected guard; other model code is unchanged.
