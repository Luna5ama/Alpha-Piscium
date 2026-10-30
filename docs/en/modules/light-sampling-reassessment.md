# Light Sampling Architecture Reassessment

Language: English | [简体中文](../../sc/modules/light-sampling-reassessment.md)

Research record, 2026-10-30. This extends the [design record](light-sampling-design.md), following the request to
reconsider light BVHs and alternative sampling methods from first principles. It records research and proposed
comparisons, not an implemented replacement or authorization to start experiments.

## Assessment

The evidence does not rule out light BVHs. It does question the cost and quality of making every shading point
follow one irreversible path through approximate aggregate weights. Three choices are independent: how lights
are grouped, how receiver-dependent importance is estimated, and how that work is shared between queries.
H-PLOC principally changes the first; it cannot settle the other two.

The next comparison should include a simple spatial probability table and, if multiple cheap candidates are
allowed, fresh RIS before a more involved ReGIR implementation. A one-ray BRDF/light MIS baseline is valuable
alongside either approach. Keep the accepted Morton four-way/cube baseline for comparison. Global cross-brick
H-PLOC remains a candidate; this research recommends reconsidering its priority, not revoking its agreed scope.

The per-query budget for multiple cheap candidates is not yet confirmed. Sections below distinguish a strict
single-candidate budget from multiple candidates followed by one actual visibility ray. Neither implies running
screen-space ReSTIR, temporal reuse, or a denoiser in the isolated initial-sampling experiment.

## What the existing evidence establishes

The experiment branch was inspected at `5c022ba1`, including the geometry record at `4920b6d9`, spatial grouping
record at `181878b9`, and wide-tree record at `3e7f66b9`. These records and experimental shaders belong to
`codex/light-sampling`; they are not integrated into the current branch. The original discussion is
`01a18648-f66e-78e0-8d01-cd906ca45429`.

| Condition or observation | Consequence for this reassessment |
|--------------------------|-----------------------------------|
| Pure Iris, software voxel tracing, unit-cube emitters, direct diffuse isolation | Hardware ray-tracing paper timings and glossy-heavy quality wins are not predictions for this path. |
| Five wide-tree scenes contain 3,921 / 2,111 / 2,194 / 908 / 877 emitters | Also consider simple methods for thousands of lights; this sample is not a maximum scene-size guarantee. |
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

Source inspection also matters. The experiment multiplies actual branch probabilities, but has no general
reverse PMF query for an arbitrary emitter. A material ID does not uniquely identify an emitter instance.
After a light is selected, a different first-hit emitter returns zero; the BRDF baseline instead receives
emission from its first hit. This is valid selected-emitter visibility testing, but it explains one difference
between the proposals. Reusing that different emitter with the selected emitter's PDF would be incorrect.

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

## Alternatives under these conditions

| Route | What changes | Final visibility rays per query | Main limitation |
|-------|--------------|---------------------------------|-----------------|
| One-sample BRDF/light MIS | Combine complementary proposals with the correct mixture density | One | Needs reverse light PMF and conditional density evaluation; does not make either proposal cheap. |
| Spatial CDF/alias table | Precompute receiver-region light probabilities, then sample a flat distribution | One | Region approximation, table memory, and update cost; no automatic visibility knowledge. |
| Fresh per-query RIS | Score several cheap light candidates at the actual receiver, then select one | One | Missing an important light in the candidate pool cannot be repaired by resampling. |
| ReGIR-style candidate grid | Share candidate generation in world space and resample at the actual receiver | One | Shared-sample correlation, finite coverage, update ownership, and estimator-weight accounting. |
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

### Spatial tables: include the simpler non-BVH baseline

A receiver cell owns a distribution over lights; it does not own those emitters. Precompute positive approximate
weights and a CDF or alias table, then use its actual normalized probabilities for one light sample. Inaccurate
cell weights affect variance, not the mean, provided support and PDF accounting remain correct. PBRT v3's
[SpatialLightDistribution](https://github.com/mmp/pbrt-v3/blob/master/src/core/lightdistrib.cpp) demonstrates this
principle with receiver-voxel distributions and a positive floor. Its CPU construction strategy is not a GPU
implementation plan.

With C active cells and N lights, a simple full scan costs O(CN) score evaluations per rebuild. As a scale
illustration, C=1,024 and N=4,096 means about 4.2 million scores and 16 MiB for one FP32 weight table alone,
before extra CDF/alias/scratch storage. This is arithmetic, not a timing prediction. A much smaller local list
can use a global full-support component:

$$
q(l\mid x)=(1-\epsilon)q_{\mathrm{cell}(x)}(l)+\epsilon q_{\mathrm{global}}(l),
\qquad 0<\epsilon<1.
$$

Lights outside the local list remain reachable. Evaluate both terms for the selected light; a fixed spatial
cutoff without the global component would lose illumination. List selection and its PMF lookup cost must also
be counted. Start without visibility history. The cell size is a sampling decision, not automatically the
existing 16³ extraction brick; a large cell can mix incompatible positions or normals just as before.

This is worth testing when many shading queries reuse a modest number of cells. It loses appeal when the
number of active cells, light count, or rebuild frequency makes O(CN) work and storage too large. A full table
is a diagnostic baseline, not a requirement to retain dense storage in production.

### Fresh RIS: multiple cheap scores, one expensive visibility test

Draw M light IDs from a known full-support source q0, evaluate an unoccluded receiver-dependent score a_l, and
use weighted reservoir selection with w_i=a_i/q0(i). Only the selected light needs cube sampling and true
visibility. For these independent initial candidates, use

$$
\widehat L_{\mathrm{RIS}}=
Z_{l^*}\frac{\sum_{i=1}^{M}w_i}{M a_{l^*}}.
$$

The normalized reservoir selection weight is not the selected light's marginal PDF. Keeping the estimator
weight is essential; see [RIS](https://doi.org/10.2312/EGWR/EGSR05/139-146) and the later
[GRIS treatment of weighted and correlated samples](https://research.nvidia.com/labs/rtr/publication/lin2022generalized/).
An approximate score is acceptable if it remains positive wherever the true contribution can be nonzero.

The proposed diagnostic is M=1/4/8, changing only candidate count first. M=1 reduces to the source sampler and
checks the weight convention. Uniform or global-power candidates are simple controls, but with thousands of
lights they can miss a nearby dominant light even at M=8. A local spatial source with a global component is
therefore a separate candidate dimension. Running M complete light-tree traversals would mix better selection
with M times the traversal work and is not the intended cheap alternative.

Resampling does not discover visibility before the final ray. If the main error is occlusion rather than the
unoccluded contribution estimate, increasing M without visibility information may give little benefit.

### ReGIR: share candidate work when independent RIS is useful but costly

[ReGIR](https://cwyman.org/papers/rtg2-manyLightReGIR.pdf) prepares multiple light reservoirs per world-space cell,
then reweights candidates for the actual shading point. The grid stores candidate samples rather than assigning
each emitter to one spatial bin. This differs from the current per-brick emitter subtrees. The
[RTXDI integration guide](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/Integration.md) separates this initial
candidate machinery from screen-space ReSTIR; the concept does not require adopting that SDK.

Our proposed first version would refresh candidates without temporal reuse. Measure the amortization using
C cells, S slots, K candidates per slot, and M candidates per shading query: approximately C*S*K+Q*M score
evaluations for Q queries, plus storage and sampling overhead. A BVH may remain useful to build those pools
without being traversed by every pixel.

Do not copy the original chapter's approximate PDF treatment and claim an unbiased implementation: its results
section explicitly reports slight bias. Use a derived weighted-resampling estimator, preserve coverage, and
check convergence. Shared finite pools can produce correlated noise even with correct weights. A cell-center
proposal also cannot distinguish opposite sides of a wall without additional receiver information. These are
reasons to compare the simpler table and fresh RIS first, not reasons to assume ReGIR is unsuitable.

### Retaining the tree: compare relevant improvements, not only builders

[Real-Time Stochastic Lightcuts](https://www.cemyuksel.com/research/stochasticlightcuts/realtime_stochastic_lightcuts-preprint.pdf)
offers a near-field heuristic that averages two separately normalized branch distributions based on minimum
and maximum cluster distances. It is not an average of the raw weights. A controlled test could hold emission
mass fixed and compare this with the original geometric score. Its published binary rule needs an explicit
multi-child extension and zero-distance handling before use in our four-way sampler.

The same work shares cut selection between pixels, while sampling independently inside cut nodes. That is a
way to amortize refinement, distinct from repeating geometric lookahead per pixel. However, the published
lightcut estimator uses multiple representative lights. [Adaptive Tree Splitting](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf)
also creates multiple light samples. Choosing only one sample from a shared cut, with a full cut-to-leaf PMF or
RIS weight, would be our adaptation and needs its own validation. It is not an already demonstrated one-ray
replacement in either paper.

[The 2024 SG light-tree paper](https://yusuketokuyoshi.com/papers/2024/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting.pdf)
directly addresses inaccurate node importance with one light per traversal. Omnidirectional emitters do not
make the idea irrelevant: their spatial distribution still creates a directional incident-radiance lobe at a
receiver. The emission-direction model could simplify, while the spatial model remains useful. However, the
paper's strongest results concern glossy materials; diffuse-dominant results are more modest. It omits shadow
visibility and reports mathematical-function and register overhead. Its uncompressed cluster summary is ten
floats, so “more compact” than its own reference is not evidence of being smaller than our complete 32B node.
Treat a diffuse-only adaptation as a candidate, not a high-confidence upgrade, and consult the
[published errata](https://gpuopen.com/download/Hierarchical_Light_Sampling_with_Accurate_Spherical_Gaussian_Lighting_(errata).pdf).

H-PLOC changes topology and build scheduling, with merged-AABB area as its criterion. Its
[ray-tracing evidence](https://gpuopen.com/download/HPLOC.pdf) cannot rank it above the receiver-aware alternatives
for this sampling workload. If retained in experiments, preserve the agreed global cross-brick scope and
separate construction, conversion, and sampling costs.

## Directions not prioritized in the next comparison

| Direction | Why it does not resolve the immediate question |
|-----------|-----------------------------------------------|
| More SH coefficients for the existing brick visibility field | They cannot recover shading-origin information lost by mixing positions across walls. |
| Visibility-aware spatial tables or neural visibility | Potentially useful, but add a receiver/light visibility model, update work, and support protection. [Neural Visibility Cache](https://jcgt.org/published/0014/02/01/paper-lowres.pdf) is evidence for learning proposal visibility, not proof that online training is affordable in this Iris path. |
| Full screen-space ReSTIR or denoising | Relevant later, but would change the initial-sampling experiment and confound its diagnosis. |
| Merging adjacent emissive faces into larger lights | Could reduce N, but changes emitter identity, conditional sampling, and textured-emission handling. It exceeds the current cube-list comparison. |
| STBN or low-discrepancy random-number remapping | A separate noise-distribution experiment; it does not repair an incorrect importance estimate or support. |

## Proposed experiments and decision rules

These are recommendations for the experiment discussion, not work started here. No candidate is accepted by
this document. Keep the current cube sampler and actual contribution definition fixed unless a comparison
explicitly isolates them.

| Order | Comparison | Question answered |
|-------|------------|-------------------|
| 1 | On selected real receivers, compare the current tree PMF with normalized sums of the exact same per-leaf score; then with per-cube unoccluded contribution estimates | Is the error primarily aggregate non-additivity, the leaf score, or something not modeled by unoccluded weights? Full scans are diagnostic, not proposed per-pixel production work. |
| 2 | Independent BRDF, current light tree, and their one-sample MIS mixture | Can complementary coverage address the near-light versus dark-region tradeoff with one ray? |
| 3 | Spatial table against the current tree; fresh RIS at M=1/4/8 if multiple candidates are allowed | Is direct receiver-based scoring better than serial aggregate decisions, and what does it cost? |
| 4 | ReGIR-style sharing only if fresh RIS shows useful quality but too much query work | Does sharing amortize the work without objectionable correlation or cell mismatch? |
| 5 | Controlled near-field weight change, SG adaptation, or H-PLOC based on the diagnosed error | Which specific remaining approximation is worth improving? These are separate candidates, not a combined patch. |

For a small diagnostic receiver set, conditional per-light samples with true visibility can also estimate
E[Z_l²]. Comparing that target with the unoccluded references separates visibility/conditional-sampling limits
from hierarchy error. Do not infer that distinction from visible-hit rate alone.

Use the following decision rules rather than treating the tree as a commitment:

- If exact leaf-score sums are much better than the hierarchy, the present aggregate estimator is the main
  suspect. Shared tables, RIS, better weights, or refinement all remain viable; this does not prove trees fail.
- If unoccluded per-cube selection is already good and topology changes are minor, H-PLOC has limited room to
  improve sampling quality. Cost or shared work becomes the more relevant comparison.
- If visibility-aware references dominate all unoccluded proposals, changing tree topology or unoccluded
  resampling alone will not remove the main source of variance. Revisit receiver-conditioned visibility.
- If a flat/shared method wins at equal total cost across the relevant scenes, remove per-query BVH traversal
  from that path. A tree can still serve preprocessing; no requirement says it must remain the final sampler.
- If active-cell counts or dynamic updates destroy amortization, keep a per-query hierarchy as a serious
  contender. Representative counts from five scenes are insufficient to assume shared tables always fit.

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

This round changes documentation only. No runtime shader, setting, emitter representation, or sampling default
was changed; no old runtime code was removed and no new GPU measurements were made.
