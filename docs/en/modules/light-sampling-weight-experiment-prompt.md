# Light Sampling Weight Experiment Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-weight-experiment-prompt.md)

This is a task prompt for the experiment session. Writing this document does not start or dispatch the task.
When the user starts it, perform the implementation and validation below, including the formula preparation
that remains necessary; do not stop at a plan or an unmeasured shader change.

## Result and working location

Implement and compare the three accepted experiment candidates against the original geometric weight:

1. Near/far-distance probability mixing.
2. Flux-weighted center and spatial variance with a simple spherical weight.
3. Full diffuse SG weighting using the same spatial summary as candidate 2.

Work in `I:/code/mcshaders/Alpha-Piscium-4` (`wt4`), branch `2.0/light-sampling`. The source snapshot checked when
writing this prompt was `8a6bd338c871dd6c7f65d80b59a26470fb3bdbf5`. Inspect the current branch, HEAD, and dirty
state before working; preserve other work and use the current relevant source. This hash is not a reset target.
The historical experiment worktree is no longer the working location, and this task does not modify `2.0/dev`.

Deliver attributable implementations, independent probability checks, target-Iris evidence, raw image
comparisons, complete build and sampling timings, and a recommendation for each candidate. Inclusion in this
experiment does not select a new default. Keep the original baseline default until separately accepted.

## Required starting material

Read the [root rules](../../../AGENTS.md), [shader rules](../../../shaders/AGENTS.md),
[tooling rules](../../../scripts/AGENTS.md), and [documentation rules](../../AGENTS.md). The authoritative scope is
the [design record](light-sampling-design.md) and the
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

## First deliverable: explicit formulas and data contracts

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
- Compare original weight plus all three candidates. Keep an independent cosine-BRDF reference for interpreting
  the images and mean energy; do not use it as a substitute for the original-tree control.
- Performance comparisons use arity 4. Preserve supported arity 2/8 behavior and check affected compilation and
  capacity branches; do not expand the performance study into an arity sweep.

## Implementation and runtime path

| Order | Work | Required outcome |
|-------|------|------------------|
| 1 | Formula and host-reference preparation | Defined models and meaningful checks, with remaining decisions explicit |
| 2 | Builder, resources, and sampler integration | Every candidate reaches the existing raw direct-lighting output, with matching node layout and actual PDF |
| 3 | Generator and compilation checks | Source and generated outputs agree; each affected variant compiles, including target Iris |
| 4 | Runtime audits and captures | Verified source/configuration/scene, matching light set and mass, independently checked probabilities, and valid raw samples |
| 5 | Matched quality and timing comparisons | Raw artifacts and a scene-specific result for all three candidates, including failures and regressions |

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

Start from the five scene presets recorded in the geometry experiment: `flatroom-light-sampling-1-720p`,
`flatroom-light-sampling-2-720p`, `basilica-1-720p`, `gi-shadow-1-720p`, and `codex-fg-night-tunnel`.
Verify their availability and actual resolution. Use matched 720p conditions across methods; do not reuse the
old 1080p wide-tree milliseconds as a control. If a preset is missing, report that gap instead of silently
substituting a differently named scene.

Before accepting quality or timing samples, verify the emitter-set and mass hashes, bounds/child links, overflow
and exhaustion counts, node stride, finite values, and leaf PMF normalization. Independently reconstruct the
actual selected-light PMF and conditional cube PDF. Check the same-frame tree and audit provenance. Original
binary readers or a successful compiler run cannot substitute for this check.

For the failure modes in the [evidence review](light-sampling-reassessment.md#review-of-original-scene-evidence-2026-10-31),
record, at a few real receivers, selection probability assigned to entirely back-side emitters, tree versus
flat normalization of the same leaf formula, and an unoccluded cube-integral reference with fixed mass.
Inspect near-light/horizon tails as well as medians. The leaf formula can itself fail, so getting closer to
its flat distribution is not the final quality objective. These are diagnostics for this matrix, not automatic
additions of a separate half-space/cosine ablation or regional-cut candidate.

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

Finish with a table for baseline and each candidate: actual node bytes, build cost, initial-pass cost, dark/bright
quality, numerical/PDF result, and recommendation. Distinguish implementation, target compilation, measured
quality, and measured performance. If a candidate fails or runtime evidence is unavailable, retain the useful
results and state the unresolved obligation. Stop after this comparison and report; do not automatically extend
into tree cuts, visibility, H-PLOC, or a new default.
