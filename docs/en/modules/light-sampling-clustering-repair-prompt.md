# Light Sampling Clustering Repair Experiment Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-clustering-repair-prompt.md)

## Offline follow-up after review (2026-11-08)

This follow-up is complete and independently reviewed. The [design record](light-sampling-design.md#review-of-layer-isolation-and-controlled-composition) reports four-tree isolation and the reviewer's additional controlled composition. The specification below remains for provenance and does not restart earlier GPU stages; no new clustering baseline is accepted.

This diagnosis follows result commit `e426ff265` and supersedes the unexecuted GPU candidate expansion/dynamic construction in this prompt. The red-pillar stop remains effective. Use retained references offline only: no new GPU jobs, runtime/weight/default changes, or lookahead/K work.

The original World Prefix/Camera comparison changed only the cross-brick hierarchy; global SAH also rebuilt brick interiors. Its red-pillar regression therefore cannot yet be attributed entirely to the upper hierarchy. With fixed 12-bin flux light-SAH + Kr, identical deterministic four-way conversion, and A + Near/Far, compare four logical trees:

| Method | Cross-brick hierarchy | Brick interiors |
|---|---|---|
| Existing control | Original World Prefix | Original Morton subtrees |
| Upper-only change | Light-SAH over actual brick-root AABBs/flux | Preserve complete original subtrees |
| Interior-only change | Original World Prefix topology | Build light-SAH separately over each brick's emitters |
| Existing global candidate | Global light-SAH | No fixed brick-subtree boundaries |

Upper SAH primitives are variable-sized actual brick-root AABBs, not uniform points or unit cubes. Bin their true bounding-box centers, use complete areas/flux, and tie-break with stable world brick keys. The interior-only variant must retain upper semantic membership; report floating-point differences from new flux reductions separately. Extract native brick roots or maximal single-brick logical subtrees, verifying complete disjoint coverage and each brick's world emitter keys. **Indivisible bricks are a diagnostic control, not a decision to narrow the final global cross-brick scope.**

Reuse the [new-receiver compact reference](../../../experiments/light-sampling/20261108-clustering-repair-reference.npz), [original five-point reference](../../../experiments/light-sampling/20261108-cluster-summaries-reference.npz), and [new moving interior points](../../../experiments/light-sampling/20261108-nearfar-combination-reference.npz). Evaluate each complete input cohort separately and retain both windows; do not multiply old ls1 results with new whole-image ratios. Do not add another Kr sweep or clustering algorithm. Verify matching builders/formulas before reusing existing control results.

Report per-point conditional variance, positive-M2 support, null probability, and node/depth counts, emphasizing the new red pillar, central ceiling, ls5 vertical face, and moving interior points. Add one diagnostic at the red pillar: retain all descendant conditional proposals and replace only root probabilities with their conditional-M2 optimum. Child j has `A_j = sum_i(M2_i / p(i|j))`; root probabilities are proportional to `sqrt(A_j)`, and minimum variance is `(sum_j sqrt(A_j))² - ||sum_i mean_i||²`. This is neither a runtime oracle nor a new weight candidate. It measures how much root-only correction can explain; globally optimal per-light shares are not substitutes.

Reuse existing scripts/result entry and store this follow-up in a separate field. Deliver reproducible findings identifying upper, interior, or interaction effects and whether upper-only changes preserve existing improvements, without promoting them to a runtime fix. The discussion chat owns this round's bilingual design updates; the experiment chat owns scripts, result fields, and necessary compact inputs while preserving previous results. Use Idle for expensive computation and retain existing evidence agreements.

Continue the human-authorized two-chat arrangement and return route: experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429`, host=`local`; notify discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12` with commits, result fields, and limitations for review. No additional stage approval is required.

## Task and authorization

Prioritize clustering quality for the world-stable light tree, and verify current Near/Far cost and benefit. The user's latest instruction prioritizes clustering, defers one-level lookahead and K1/K2/K4 until clustering is repaired, and authorizes arranging the next work. Pause new combinations and selection among those summaries/lookahead candidates; retain existing code, images, and evidence. All clustering comparisons use Half-space A + Near/Far (model 6); do not change production defaults.

Experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` executes; discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12` independently reviews, host=`local`. The human previously designated this experiment chat and requested notification back to the discussion chat for review. Continue that authorization for scoped implementation, necessary Vibris experiments, dispatch, and completion/material-blocker return messages, keeping model and effort unchanged. Use wt4, `I:/code/mcshaders/Alpha-Piscium-4`, branch `2.0/light-sampling`; preparation started at `29a5f1be3`. Inspect newer commits and workspace changes rather than resetting.

## Sources and changed scope

Read the [current decision](light-sampling-design.md#current-priority-repair-clustering-first-2026-11-08), [matched-input grouping record](../../../experiments/light-sampling/20261108-grouping-control.json), and [viewer](../../../builds/light-sampling/20261108-grouping-control/viewer/comparison.html). Current whole-image World Prefix/Camera RGB variance ratios are 1.333/1.502 for ls1/ls5, despite local ls1 improvements. The starting point is an established grouping difference; historical render-distance changes do not explain this matched-input regression.

Frozen controls are `experiment/light-sampling-grouping-prefix` (`e9132607e`) and `experiment/light-sampling-grouping-camera` (`24e7ca13d`). The latter combines original camera grouping with the current evaluation path; it is not the abandoned world-modulo variant. Development entry points are [LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl), [LightSamplingLayout](../../../shaders/techniques/debug/LightSamplingLayout.glsl), [LightSampling](../../../shaders/techniques/debug/LightSampling.glsl), and [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl).

Reuse the [five-point conditional reference](../../../experiments/light-sampling/20261108-cluster-summaries-reference.npz), [Near/Far combination record](../../../experiments/light-sampling/20261108-nearfar-combination.json), and [new interior-point reference](../../../experiments/light-sampling/20261108-nearfar-combination-reference.npz) where applicable. The older ls1 five-point batch has different voxel identity from the new grouping comparison. Identical emitter keys do not justify reusing its M2 or multiplying result ratios; screen each complete input cohort separately.

## Candidate and retained semantics

Preserve emitter membership/flux, true texture/visibility evaluation, conditional cube sampling, the A + Near/Far formula, and actual PDF. Retain four-way final nodes and 32B storage; account separately for temporary build storage and capacity. Exclude orientation factors, near-zero rule changes, visibility guidance, per-pixel cuts, and lookahead from this grouping comparison.

Identical world emitter keys, material, and flux must yield the same semantic topology; leaf PMFs must also agree for a fixed world receiver and normal, ignoring physical addresses. Camera windows, world modulo, and atomic allocation order must not determine membership. **Fixed Morton-prefix parents are replaceable.** Bricks own extraction; the global candidate consumes individual emitters, allowing partial neighboring bricks to form a cluster. Do not silently reduce the task to rearranging indivisible brick roots. Sorting globally by Morton and rebuilding the same prefixes does not constitute new clustering.

Implement one deterministic global binned light-SAH candidate first. Bin emitter centers along all three axes and choose the lowest-cost nonempty binary partition using current flux `Phi` and AABB surface area `S`, including the full unit-cube bounds:

```text
C(axis, split) = Kr(axis) * (Phi_left * S_left + Phi_right * S_right)
Kr(axis) = max(parent_diagonal) / parent_diagonal[axis]
```

This simplifies the [PBRT light-BVH](https://pbr-book.org/4ed/Light_Sources/Light_Sampling) cost under uniform omnidirectional bounds, using existing mass as a flux proxy without adding direction summaries. Start with a fixed small bin count, such as 12, without a bin-count sweep. When no centroid axis can separate emitters, partition deterministically by complete world key. Resolve equal costs, coincident centers, reductions, and wide-child order deterministically. An offline ordinary geometric `N_left * S_left + N_right * S_right` SAH control may clarify the effect of power terms/spatial partitions; it does not automatically enter the GPU matrix.

Convert the temporary binary tree during construction to at most four contiguous children using a fixed rule; initially eliminate the next binary internal level per output node, stopping at leaves. Use the same widening rule for all candidates, avoiding receiver-driven expansion. Record final depth, nodes/empty slots, cross-brick clusters, and peak temporary memory. SAH/area are explanatory metrics only. Construction must not consume receivers, normals, per-light M2, or true visibility. No H-PLOC parallel core is specified this round, and ray-tracing results are not evidence of sampling benefit.

## Execution and experiment exits

1. **Screen whether grouping actually improves light selection offline.** Rebuild both controls and the candidate on verifiable matching references. Align per-light PMF, support failures, and null probability, and use true conditional RGB M2 to evaluate `sum(M2_i / p_i) - ||sum(mean_i)||²`. Zero probability with positive M2 fails; unobserved M2 does not establish permanent invisibility. Along ancestors of affected emitters, record where probability first diverges, cluster extent/flux, and allocation from each Near/Far half rather than reporting SAH alone. Preserve already improved points. Deliver a reproducible first screening result without building a general test platform.
2. **Cover the user's actual regions.** Include improved left red-pillar faces and the yellow-lit upper-middle ceiling in ls1, whole-image and the reviewed regressing vertical face in ls5, and the newer interior points in moving-4. Old side-26/top-28 are historical supplements. Before inspecting candidate outcomes, choose fixed interior ROIs by geometry/material, record world coordinates, normals, and display coordinates, and mark them in the viewer, avoiding block edges. Do not substitute the old right-side pillar point for the left red pillars. Capture true per-light M2 only for missing representative points in these regions when needed; mismatched old references are not substitutes. Follow existing window/uncertainty protocols.
3. **Integrate a justified candidate into the live runtime.** If screening improves quality while preserving support/stability, continue implementing and comparing one candidate without awaiting approval between stages. Prefer the actual dynamic-build-to-sampler path. A frozen-tree prototype may first isolate quality, but identify its upload/freeze mechanism and continue distinguishing incomplete dynamic construction. If offline candidates provide no benefit or reveal a severe counterexample, return data and the first failure instead of forcing a complex GPU builder or reopening K/lookahead work.

GPU construction may use multiple dispatches; temporary child indices, partitioning, and compaction are implementation choices. Avoid unbounded cross-workgroup spinning. Validate publication across dispatches, indirect counts, and capacity; overflow must not silently drop emitters. New resources require coherent generator, consumer, generated-output, and contract changes. Keep traceable experiment refs rather than adding production options for one comparison. Report only implemented capabilities; CPU construction/frozen uploads are not free.

## Runtime quality, stability, and Near/Far cost

The experiment chat owns shared Minecraft operation. Verify active jobs and loaded sources first; do not take over unreturned state or wait for unrelated workspaces to reach global idle. Reuse the available Vibris queue and established capture path without restarting Minecraft or its launcher. Keep the render distance read at the start (the user reports 32; do not sweep 16/24/32), scene loading, and tick conditions fixed.

Start quality comparisons with ls1, ls5, and moving-4: World Prefix + A + Near/Far, Camera Grouping + A + Near/Far, one new clustering candidate + A + Near/Far, and independent Cosine BRDF. Preserve native 1920×1080, G16, four-way nodes, cube sampling, direct-only evaluation, no reuse/RC/TAA/jitter/denoising, 256 warmup frames, a 1-spp image, and 32 consecutive frames for the mean. Retain actual PDFs and final visibility. Reuse captures only after matching actual emitter keys/flux, world voxel materials, receiver geometry, camera, and evaluation settings; counts or preset identity alone are insufficient. Recapture only affected groups when needed rather than forcing old counts. Verify actual Target Iris loads, effective settings, generated resources, and sampled PDFs.

Use common complete finite receivers across methods to report whole-image/fixed-ROI RGB variance, means, zero-sample coverage, and excluded counts, without brightness filtering. Darkness alone does not establish bias. State finite 32-frame/M2 scope; repeat narrow windows only for concrete anomalies. Similar image means do not prove correct PDFs, and selected winning regions cannot represent the image.

First reuse saved cross-boundary inputs from all four moving presets to check candidate topology/PMF invariance under changed camera addresses for identical world inputs. After dynamic integration, repeat crossings with those presets. Separate actual emitter/occluder changes from camera-only movement. Add one offline perturbation diagnostic using saved old/additional emitter cohorts, reporting changes in topology and PMF on common emitters without claiming insertion/removal stability. Frozen-tree screenshots do not prove dynamic motion repair.

Measure Near/Far separately as **current World Prefix model 1 versus model 6**, only ls1 and ls5, in adjacent ABBA order with 128 metric-warmup frames and 128 measured samples per entry, auditing disabled. Measure [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl), the full build-stage sum, related shared prepare passes, and a stable sentinel; report absolute ms, differences/ratios, and drift. Initial includes proposal/cube/trace work, not an isolated query benchmark. Historical A→A+Near/Far deltas of approximately +5.23%/−2.10%/+5.19%/+14.22% for ls1/2/4/5 are historical controls only.

Also verify current-input A versus A + Near/Far benefit, reusing fully matching quality/M2 evidence or adding only the missing A controls for these two scenes. Retain prior local Near/Far counterexamples without reopening summary-combination matrices. If dynamic new clustering exists, measure its full build + initial cost against World Prefix. Report CPU/upload/query costs and missing components separately for frozen prototypes; measuring only the fastest core kernel is insufficient.

## Delivery, ownership, and return notification

The experiment chat may change scoped builder sources, necessary resources/generators, diagnostic scripts, and bilingual result summaries. The discussion chat owns route changes, overall acceptance, and independent review. Return evidence before changing sampling support, weight formulas, final 32B representation, or world-stability semantics. Handle other local implementation choices autonomously. A rejected candidate is a valid experiment outcome, not a completed clustering fix.

Use this topic entry and retain compact tracked results plus necessary reproducible code. Place artifacts under `builds/light-sampling/20261108-clustering-repair/`, with a suggested index at `experiments/light-sampling/20261108-clustering-repair.json`. Add a separate viewer while preserving previous pages/provenance. Check raw OpenGL row orientation, lossless images, and real browser switching. Deliver input/source identities, whole-image/ROI results, M2/PMF, motion, and cost findings, explicitly separating static prototypes, dynamic implementation, untested conditions, and rejected candidates.

Use PowerShell 7 without a profile; run expensive non-benchmark work at Idle priority. If using Pyxis, verify the effective priority tier before expensive execution. Follow repository rules and run both diff checks before committing. Existing artifact-retention pauses and automatic-policy rejections remain effective; do not retry deletion or route around them with another tool. Retain new evidence until independent review and list paths, purposes, and cleanup conditions. Avoid unnecessary worktrees.

On completion or a material design counterexample, stop submitting new runtime work, verify and return actual source, scene, configuration, freeze state, and active jobs. Use `send_message_to_thread` to notify discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12`, host=`local`, with this task entry, commits/refs, results page, necessary originals, and remaining obligations, explicitly requesting independent review. Authorization derives from the human's existing two-way arrangement. A useful first offline result may be returned early while independent work continues.
