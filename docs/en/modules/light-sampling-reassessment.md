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

The user's follow-up rejects per-region, per-light tables as too fine and expensive, favoring tree clusters as
the estimation unit. Fixed-budget global-candidate RIS is not a suitable replacement at large light counts;
ReGIR is a complementary sharing/resampling direction that can use a tree-based proposal. The earlier ranking
mixed these different levels. This revision removes the dense-table and standalone-RIS experiment priorities.

The focus is now cluster importance, receiver-region cuts and visibility, and the global clustering that
supports them. Keep the accepted Morton four-way/cube baseline. Global cross-brick H-PLOC remains a candidate,
not an accepted quality improvement. MIS is a separate proposal-combination choice. Full per-leaf scans remain
small diagnostic references, not a second production architecture alongside spatial tables. No experiment or
temporal-reuse implementation is authorized by this discussion.

## What the existing evidence establishes

The experiment branch was inspected at `5c022ba1`, including the geometry record at `4920b6d9`, spatial grouping
record at `181878b9`, and wide-tree record at `3e7f66b9`. These records and experimental shaders belong to
`codex/light-sampling`; they are not integrated into the current branch. The original discussion is
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

### Visibility: learn receiver-region / cluster behavior

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

These are recommendations for the experiment discussion, not work started here. No candidate is accepted by
this document. Keep the current cube sampler and actual contribution definition fixed unless a comparison
explicitly isolates them.

| Order | Comparison | Question answered |
|-------|------------|-------------------|
| 1 | On selected real receivers, compare the current tree PMF with normalized sums of the exact same per-leaf score; then with per-cube unoccluded contribution estimates | Is the error primarily aggregate non-additivity, the leaf score, or something not modeled by unoccluded weights? Full scans are diagnostic, not proposed per-pixel production work. |
| 2 | Fixed topology and emission mass; original versus a near-field weight candidate, then a separate SG candidate if justified | Does the cluster importance estimate improve enough to pay for its cost? |
| 3 | Fixed weights and topology; per-query tree traversal versus a bounded regional cut | Does exposing appropriate clusters before selection justify partition storage and query scoring? |
| 4 | Fixed cut and conditional sampler; frozen region/cluster visibility-contribution statistics versus unoccluded weights | Is remaining variance driven by occlusion, and does the receiving-region resolution preserve useful information? |
| 5 | Same sampling policy; Morton versus global H-PLOC, including cut rebuilding and statistics invalidation | Does improved global grouping help after accounting for its complete downstream cost? |

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

This round changes documentation only. No runtime shader, setting, emitter representation, or sampling default
was changed; no old runtime code was removed and no new GPU measurements were made.
