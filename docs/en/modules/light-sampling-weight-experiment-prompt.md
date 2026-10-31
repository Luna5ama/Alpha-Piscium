# Light Sampling Diagnostic and Weight Experiment Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-weight-experiment-prompt.md)

This is a task prompt for the experiment session. Writing this document does not start or dispatch the task.
When the user starts it, perform the implementation and validation below, including the formula preparation
that remains necessary; do not stop at a plan or an unmeasured shader change.

## Dedicated-scene rerun correction (2026-10-31)

The user requested rerunning this comparison on `light-sampling-1` through `light-sampling-5`, all natively 1920×1080.
The previous legacy-scene run remains a [supplemental record](../../../experiments/light-sampling/20261031-weights-results.json);
do not relabel it or treat it as dedicated-scene evidence. Reuse the committed weights, audit and original-mass implementation.
Unchanged numerical/layout checks remain applicable, but recapture quality, references and timing for all seven methods.
Use a separate run directory and tracked result. Do not use the old room2 fixed-light texture for points in these scenes.
Pyxis Terra medium Fast still owns runtime capture and image comparison; the parent prepares commits, reviews evidence
and updates the paired records.

## Result and working location

First repair the audit and reassess the current four-way tree, completing independent half-space/cosine
ablations and rejection-reason classification. Then implement and compare three weight candidates against
the original geometric weight:

1. Near/far-distance probability mixing.
2. Flux-weighted center and spatial variance with a simple spherical weight.
3. Full diffuse SG weighting using the same spatial summary as candidate 2.

Work in `I:/code/mcshaders/Alpha-Piscium-4` (`wt4`), branch `2.0/light-sampling`. The source snapshot checked when
updating this prompt and its evidence was `b8295409`. Inspect the current branch, HEAD, and dirty
state before working; preserve other work and use the current relevant source. This hash is not a reset target.
The historical experiment worktree is no longer the working location, and this task does not modify `2.0/dev`.

Deliver attributable implementations, independent probability checks, target-Iris evidence, raw image
comparisons, complete build and sampling timings, and a recommendation for each candidate. Inclusion in this
experiment does not select a new default. Keep the original baseline default until separately accepted.

| Previous recommendation | Scope of this batch |
|-------------------------|---------------------|
| Out-of-bounds / competing audit writes | Repair and validate 720p/1080p addressing, then capture the current four-way tree; old 1080p quality conclusions still require corresponding recaptures |
| Half-space / cosine estimation | Run the two independent ablations below first, reusing matched scene/quality/timing protocols without crossing them with the three weights |
| Three weight candidates | Implement all three against the original weight; do not silently fold diagnostic variants into the shared baseline |
| Rejection reasons and residual occlusion questions | Add classified counters; if necessary, collect per-light visible contribution/second-moment references at a few real receivers |
| Where a cut should refine | Report depth/branch distribution discrepancies; shared-cut implementation and benefit comparisons remain later work |
| One-sample MIS, visibility guidance, H-PLOC | Retain as later work, outside this implementation batch; ReGIR remains complementary |

## Required starting material

Read the [root rules](../../../AGENTS.md), [shader rules](../../../shaders/AGENTS.md),
[tooling rules](../../../scripts/AGENTS.md), and [documentation rules](../../AGENTS.md). The authoritative scope is
the [design record](light-sampling-design.md), [evidence review and next work](light-sampling-reassessment.md#narrowing-the-next-work), and the
[accepted weight candidates](light-sampling-reassessment.md#weight-candidates-for-the-next-experiments).
Use the formulas, costs, references, and limitations in that reassessment rather than reopening unrelated designs.

| Source | Responsibility and first inspection |
|--------|-------------------------------------|
| [LightSampling](../../../shaders/techniques/debug/LightSampling.glsl) | Node layout, geometric weights, packed child links, and actual path-PMF multiplication |
| [LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl) | Emitter extraction, leaf mass, brick-local construction, and global reduction |
| [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl) and [LightSamplingCube](../../../shaders/techniques/debug/LightSamplingCube.glsl) | Selected-emitter contribution, retained conditional sampler, actual visibility, and audit output |
| [Options](../../../scripts/options.main.kts), [program registration](../../../scripts/programs.main.kts), and [resource declarations](../../../scripts/shaders.properties) | Experiment selection, generated entry points, stage dependencies, and SSBO allocation |
| [Wide-tree analysis](../../../scripts/light_sampling_wide_analysis.py), [geometry analysis](../../../scripts/light_sampling_geometry.py), and [runtime artifact helpers](../../../scripts/light_sampling_runtime.py) | Starting points for host-side audits, raw captures, and analysis; inspect their historical assumptions before reuse |
| [Geometry record](../../../experiments/light-sampling/20261029-geometry.json) and [wide-tree record](../../../experiments/light-sampling/20261030-wide-tree.json) | Existing scenes, configuration, capture protocol, provenance, and limitations; these are historical evidence, not current runtime validation |
| [Review record](../../../experiments/light-sampling/20261031-evidence-review.json) and [offline reconstruction](../../../scripts/light_sampling_evidence_review.py) | Back-hemisphere probability, depth discrepancies, unoccluded cube references, and counterexamples; the reconstruction assumes the old binary tree and needs adaptation for the current four-way tree |

Known traps require direct handling:

- `SETTING_DEBUG_LIGHT_WEIGHT` currently changes both the geometric formula and leaf emission mass through
  `ls_emissionMass`. Enabling that old switch does not implement this experiment. All candidates must retain
  the same baseline leaf mass. Give the new variants a coherent experimental selection path, updating callers
  and removing any superseded conflicting path according to repository rules.
- The geometry/runtime tree readers assume binary children and eight 32-bit words per node. The wide-tree
  auditor handles packed child counts but still assumes 32B nodes and the original formula; its run paths and
  revision map are also historical. Adapt the affected decoder and audit path to the actual layout and variant.
  Preserve historical records; do not relabel old captures as new validation.
- The fixed 64-slot audit has 144 writers at 1080p, causing competing valid-slot writes and out-of-bounds
  access. The [original-evidence review](light-sampling-reassessment.md#review-of-original-scene-evidence-2026-10-31)
  checked the historical source and configuration. Before new captures, correct pixel selection and indexing
  so writers are unique and within capacity at the actual resolution, updating host-side point decoding.
  An old 1080p PDF audit pass is not correctness evidence; preferring 720p must not hide this defect.

## Preliminary diagnostics and independent ablations

Repair audit addressing first, checking unique in-bounds slots at the actual resolution. Include one 1080p
audit check without requiring a rerun of the old full four-case arity/lookahead matrix. The main quality
rerun matrix uses the dedicated presets at native 1080p. Old 1080p results not recaptured remain pending revalidation; a new addressing
check cannot retrospectively validate their image quality. Reconstruct actual selection distributions,
back-hemisphere probabilities, leaf-formula and unoccluded-cube references, and depth/branch discrepancies
for the current four-way baseline. Do not inherit binary percentages or inspect only distribution medians.
Reproduce old counterexamples by world position/normal or choose equivalent near-light receivers rather
than relying on old pixel-slot numbers after the audit grid changes.

Both diagnostic variants retain 32B nodes, original mass and distance denominator, Morton four-way topology,
cube sampling, and RNG policy:

| Variant | Isolated change and control |
|---------|-----------------------------|
| A: exact half-space rejection | Against original weights, assign zero only to nodes whose entire AABB lies behind the receiving plane; retain the remaining cosine calculation |
| B: AABB cosine estimate | Against A, replace only cosine estimation with the projected-extent approximation from the existing alternate-weight branch, excluding its different mass and distance terms |

Let `h = dot(normal, delta)`, `e = dot(abs(normal), halfSize)`, and `d2 = dot(delta, delta)`.
A's geometric condition is `h + e <= 0`; handle rounding conservatively so near-tangent positive contribution
does not lose support. B retains A's zero test; otherwise its clipped height is
`h >= e ? h : (h + e)² / (4e)`, divided by `sqrt(max(d2, 0.25))` and clamped to the original `[0.001, 1]`
range. Keep the original distance denominator `max(d2, radius², 0.25)`. This isolates an existing cosine
approximation; it is neither an exact cube integral nor an accepted new default.

The current selection loop assumes positive total child weight. Introducing exact zero weights requires
handling paths whose children all have zero weight. Return an explicit zero-contribution outcome without
retrying or dividing by zero. Include that outcome in host probability audits, checking
`sum(q_leaf) + q_zero = 1` instead of requiring leaf probabilities alone to sum to one for every receiver.
Positive-contribution leaves must retain positive proposal probability. Report entirely zero-contribution
reference points separately without normalizing an empty target. Here `q_zero` means only the selection
stage's null outcome, not an extra probability for occlusion or other zero contributions after selecting a light.
Check all-back-side, mixed-cluster, receiver-plane-crossing, and near-tangent cases.

Add mutually exclusive terminal counters along actual return paths: no lights/zero selection outcome,
cube-direction generation failure, invalid cosine or target intersection, exhausted trace, no hit,
non-emissive first hit, proxy interior without a surface entry, emissive first hit not matching the target,
and successful target hit. Within successful hits, distinguish zero and nonzero actual texture contribution;
retain nonfinite counts. State denominators and reconcile terminal counts with receiver counts. Do not
label every target mismatch as occlusion. Disable added diagnostic writes and reference probes for timing.

If classified counts and the unoccluded reference still cannot explain room2's dark region or near-light
tails, collect per-light conditional samples at a few fixed real receivers. Compare actually visible
contribution means and second moments with the unoccluded reference, including sampling uncertainty.
Expand only the points/samples needed to distinguish remaining causes. This diagnosis does not implement
visibility guidance or full-image per-light scans.

## Formulas and data contracts before implementation

The three model families are selected for comparison; some implementation mathematics is still open. Before
shader integration, write a compact specification for each candidate in the research record: the actual
formula, parameter meanings, finite-domain behavior, support, and the probabilities used by the estimator.
Resolve ordinary choices within these model families and record the rationale. If a required resolution would
change the candidate's meaning or the fixed contribution contract, return that decision to the design discussion.

| Candidate | Required preparation |
|-----------|----------------------|
| Near/far mixture | Define the four-child extension of the paper's separately normalized near/far distributions, including one or several zero minimum distances. Retain the existing cosine treatment to isolate the distance change. Mix probabilities, not raw weights. |
| Spherical weight | Specify one concrete geometric formula using flux-weighted center and scalar spatial spread. Define the unit cube's finite leaf spread and its measure, aggregation, distance treatment, and support. This candidate does not evaluate an SG integral. |
| Diffuse SG | Use the same leaf moments, reductions, and node layout as the spherical candidate. Specify the omnidirectional-emission adaptation and the diffuse clamped-cosine integral, consulting the paper and errata linked in the reassessment. Handle small/large sharpness, near-field limits, and numerical underflow without silently losing contribution support. |

Retaining the AABB and appending one vec4 gives the straightforward 48B layout for candidates 2 and 3; candidate
1 and the original baseline remain 32B. Keep the spherical and SG representation identical for attribution.
A replacement 32B spherical layout is outside this first comparison. Treat node stride, SSBO capacity, clears,
leaf initialization, every reduction level, consumers, and host capture decoding as one contract. In particular,
the existing `bufferObject.12` capacities in the resource declarations depend on the node size.

Use small independent host-side cases to check normalization and support before the live run: receivers inside
one or overlapping child bounds, near a cube and the horizon, very distant clusters, unequal masses, and finite
cube extent. For SG integration, use independent numerical integration or suitable higher-precision references
at representative parameter values; duplicating the shader expression alone is not an accuracy check.

## Fixed comparison conditions

- Keep the same emitter positions, material identities, mass values, Morton grouping, and four-way topology.
  Preserve projected-solid-angle cube sampling and the actual textured contribution evaluation.
- Use the existing isolated one-sample Lambertian direct-lighting path. No sky/sun contribution, radiance cache,
  path guiding, ReSTIR, spatial reuse, TAA, or denoising. Do not change RNG policy to improve a candidate's images.
- Keep geometric lookahead and visibility guidance disabled. Do not add tree cuts, H-PLOC, RIS/ReGIR, BRDF/light
  MIS, or new emitter geometry to this matrix. Visibility and H-PLOC remain deferred.
- Compare original weight plus all three candidates, separately completing preliminary A/B ablations.
  Repair only the audit in the main matrix's original sampling baseline; do not fold in A/B or automatically
  test combinations. Keep an independent cosine-BRDF reference for interpreting images and mean energy,
  rather than substituting it for the original-tree control.
- Performance comparisons use arity 4. Preserve supported arity 2/8 behavior and check affected compilation and
  capacity branches; do not expand the performance study into an arity sweep.

## Implementation and runtime path

| Order | Work | Required outcome |
|-------|------|------------------|
| 1 | Audit repair, four-way reconstruction, and classified counters | Correct 720p/1080p addressing, reproducible current-baseline diagnostics, and rejection causes separated from aggregate hit rate |
| 2 | A/B ablations and their required compilation, captures, and timing | Separate attribution for half-space rejection and cosine changes, with correct zero outcomes and actual PDFs |
| 3 | Three-weight formula and host-reference preparation | Defined models and meaningful checks, with remaining decisions explicit |
| 4 | Builder/resources/sampler integration and compilation | Each candidate reaches raw direct-lighting output; layouts/PDFs and maintained/generated sources agree, including target-Iris compilation |
| 5 | Matched quality and timing comparisons | Raw artifacts and scene-specific results for all three weights, including failures/regressions; fixed-receiver visible-contribution references when needed |

Register settings through the maintained options source and run the normal generator from the scripts
directory when required. Do not edit generated wrappers or bindings directly. Keep GLSL indexing and dispatch
declarations within the shader rules. The existing [static syntax harness](../../../scripts/light_sampling_wide_syntax.py)
uses stubs and the old output directory; adapt it if needed and use it only when IDE diagnostics are unavailable
or external validation is requested. It cannot establish target-Iris or GPU correctness.

Use PowerShell 7 without a profile and prefer the latest available GraalVM JDK. Run time-insensitive compilation
and long background preparation at lowest process priority; keep benchmark conditions comparable. If the global
Gradle home is broken, stop and report it rather than switching to another Gradle home.

Use Vibris for live compilation, capture, and measurements. Verify that the loaded shader pack actually comes
from wt4 and record its resolved revision or source hash, effective settings, scene/preset hash, and resolution.
Derive affected pass names from current registration and loaded programs, not historical wrapper numbers.
Do not restart Minecraft or its launcher. If safe recovery cannot restore the runtime, preserve completed work
and report the missing runtime evidence; do not replace it with replay evidence or a claimed pass.

## Evidence and acceptance

Use the five dedicated presets: `light-sampling-1`, `light-sampling-2`, `light-sampling-3`, `light-sampling-4`, and `light-sampling-5`.
Verify their availability and actual resolution. Use matched native 1920×1080 conditions across methods; do not reuse
old wide-tree milliseconds as a control. If a preset is missing, report that gap instead of silently
substituting a differently named scene.

Before accepting quality or timing samples, verify the emitter-set and mass hashes, bounds/child links, overflow
and exhaustion counts, node stride, finite values, and full PMF normalization including zero outcomes. Independently reconstruct the
actual selected-light PMF and conditional cube PDF. Check the same-frame tree and audit provenance. Original
binary readers or a successful compiler run cannot substitute for this check.

For the failure modes in the [evidence review](light-sampling-reassessment.md#review-of-original-scene-evidence-2026-10-31),
record, at a few real receivers, selection probability assigned to entirely back-side emitters, tree versus
flat normalization of the same leaf formula, and an unoccluded cube-integral reference with fixed mass.
Inspect near-light/horizon tails as well as medians. The leaf formula can itself fail, so getting closer to
its flat distribution is not the final quality objective. Complete A/B ablations separately as specified
above and report depth/branch diagnostics, without combining them with a regional-cut implementation.

Quality captures retain raw linear 1-spp frames and 32-frame means/variance with valid-sample counts. Use matched
warmup and capture windows, starting from the existing 256-frame quality warmup. Record dark regions, near-light
or bright-tail regions, and low-frequency spatial error separately. Preserve nonfinite and incomplete counts;
do not hide them by clipping or silently dropping pixels. For representative crops, use a sufficiently converged
reference and distinguish reference error from short-window variance. Keep exposure and masks consistent across
methods and preserve the underlying raw data.

For timing, disable audit atomics and use comparable forward/reverse pairs, initially 128 warmup and 128 measured
frames per case as in the existing protocol. Measure the initial-sampling pass, all required preparation/build
stages, affected siblings, and a stable workload sentinel. Include spatial-moment construction and the actual
node memory footprint. State the statistic used; a sum of stage averages is not a median. Initial-pass time
includes cube sampling and tracing, so do not describe it as isolated traversal time. Extra diagnostic tree
builds are not a necessary cost of the cosine-BRDF method; distinguish them from its required work.

Compare equal sample counts first and complete equal-time cost second. Repeat measurements when drift or
uncertainty prevents a conclusion, rather than broadening the matrix without cause. Fix numerical or layout
defects within scope and rerun the affected checks. A candidate with invalid support or PDFs is not eligible
for a quality/performance acceptance claim.

## Delivery and stopping point

Store a unique run's raw artifacts, receipts, analysis, and offline comparison page under the existing
[light-sampling build directory](../../../builds/light-sampling/). Retain native Scene, Sampler, and Accumulation
comparison dimensions and zoom/pan; reuse the available image-comparison workflow. Record provenance and
recoverable artifact paths in a tracked result under the [experiment records](../../../experiments/light-sampling/).
Do not overwrite historical runs or commit ignored raw dumps.

Update the bilingual design records with formulas, implementation state, per-scene findings, and limitations.
Commit scoped source, generated outputs, analysis, and tracked records on `2.0/light-sampling`, after
`git diff --check` and `git diff --cached --check`. Preserve unrelated work. This task does not publish, merge to
`2.0/dev`, start another session, or send external messages.

Finish with a table for the baseline, A/B ablations, and each of the three weights: actual node bytes, build
cost, initial-pass cost, dark/bright quality, numerical/PDF result, and recommendation. Also deliver audit-repair
evidence, current four-way rejection/depth diagnostics, and any necessary fixed-receiver references; if the
latter were not triggered, explain why the existing evidence was sufficient. Distinguish implementation, target compilation, measured
quality, and measured performance. If a candidate fails or runtime evidence is unavailable, retain the useful
results and state the unresolved obligation. Stop after this comparison and report; do not automatically extend
into tree cuts, visibility, H-PLOC, or a new default.
