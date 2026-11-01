# Light Sampling Motion Stability Experiment Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-moving-stability-prompt.md)

This page preserves the 2026-10-31 modulo candidate and review. Its implementation task is superseded by the 2026-11-01 [complete world-coordinate grouping design](light-sampling-world-grouping-prompt.md); do not continue implementing its periodic keys or fixed-capacity restriction.

## Experiment submission (2026-10-31)

The minimal candidate is committed as `0d1629ae`, with frozen old source `6d69b145`. The [result index](../../../experiments/light-sampling/20261031-moving-stability.json) and [24-image viewer](../../../builds/light-sampling/20261031-moving-stability/comparison.html) preserve old A, new A and Cosine BRDF for all four presets, with 1 spp and 32 consecutive-frame means.
Quality and timing retain the existing controlled G=16 baseline. The shader default G=64, which the presets do not override, receives compilation coverage only; this is not G=64 quality validation.
The first three old-A captures with drifting inputs are preserved and excluded. After targeted recapture, each pose has matching old/new emitters and complete world-voxel inputs. Poses 2→3 also share actual emitters and occluders across the y-brick crossing: old semantic topology changes, with full-PMF TV distances about 0.789/0.900 at two fixed world points; new topology matches and all three points have zero TV.
Poses 1→2 and 3→4 have actual window emitter and voxel-input changes (1182→1136 and 1136→1083 emitters). Those fall outside the equal-input invariance guarantee, so this does not establish that every positional discontinuity is removed.
The [world-space diagnostic](../../../scripts/light_sampling_moving_analysis.py) restores voxel positions using actual G and reconstructed camera coordinates, rather than comparing raw slots or memory hashes across cameras. Three receivers match by world position, normal, material and albedo; all 24 checks have 32 complete samples.

| Preset suffix | New/old whole-image RGB sample variance | Quality assessment |
|---------------|----------------------------------------|--------------------|
| 1 | 3.376 | Significant regression |
| 2 | 3.839 | Significant regression |
| 3 | 0.0642 | Large improvement with local regressions |
| 4 | 1.900 | Significant regression |

Forty-five offline checks cover three G values, binary/four/eight-way branching, negative coordinates, empty/single-light cases and complete PMFs across boundaries. Eight real trees pass bounds, mass, sampled-PDF and conditional-emitter-PDF audits. Target Iris compilation covers default four-way, binary/eight-way 48B configurations on grids 32/64, and experiment-off.
The wrapping counterexample moves a 241-block-wide joint cluster from depth 1 to depth 5; stability does not establish quality acceptance. Representative ABBA timing records all 14 build kernels, initial, shared prepare and sky LUT; median build time is about 0.139–0.141 ms, without a credible speedup conclusion. Rare nonfinite RGB events are reported separately and excluded through a common complete-pixel mask.
The experiment submission validates the mechanism with significant quality tradeoffs and does not recommend unconditional acceptance of periodic grouping. Continuous motion and the first crossing frame remain untested. The 32-frame variance is not converged error or proof about brightness bias. The candidate and originals are preserved; discussion-side review follows.

## Independent review (2026-10-31)

Decision: equal-input stability passes, but the periodic-grouping candidate fails quality acceptance and is not promoted to the accepted baseline. The source implements the proposed minimal design faithfully; the quality cost belongs to that design rather than an experiment-agent deviation.
The discussion chat inspected the six-line shader diff and original moving-2/3 mean images, then read saved raw data with an [independent script](../../../builds/light-sampling/20261031-moving-stability/parent_review.py), without calling the experiment analysis functions or capturing new GPU samples.
The [independent results](../../../builds/light-sampling/20261031-moving-stability/parent-review.json) reconstruct the camera, world-voxel identities, semantic topology and full light-selection PMFs, and recompute all four variance ratios using the common complete-pixel mask across all three methods. All agree with the submission.

Poses 2→3 contain the same 129606 nonempty world voxels. At the first fixed receiver, old A's null-sample probability jumps from 0.14046 to 0.92887; new A remains at 0.14412 on both sides.
Here a null sample means traversal enters a branch whose later children all have zero weight, before any actual visibility test. It explains how regrouping wastes many samples, without proving brightness bias.
The three old-A PMF TV distances are 0.78911, 0.90046 and 0; all new-A distances are 0. Stability holds, but the noise penalties in moving-1/2/4 still reject the candidate.

Review covers only the existing G=16 static poses and 32-frame windows. G=64 quality, the first continuous-motion crossing frame, convergence and the cause of nonfinite samples remain unaccepted.
This review changes records only, without reverting or editing shaders. The candidate remains on the experiment branch, and original failed captures and the viewer are preserved. A later design must preserve both world-space stability and actual spatial locality; stable periodic slots alone are insufficient for acceptance.

## Authorization, starting point and delivery

In discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12`, turn `01a1947c-4f4f-7f21-bca6-465fe2a724af`, the user explicitly requested:

> 我准备了几个light-sampling-failure-moving preset，你自己简单设计一下修改方案然后让实验agent做实现，完事之后你让那个agent发消息回来提醒你做复查

This authorizes experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` to implement the minimal candidate below, perform targeted validation,
and proactively message the discussion chat when ready for review. The reported symptom is an abrupt regional noise or brightness change when crossing a position.
Implementation completion and review acceptance remain separate; this prompt is not evidence of improvement.

Use `I:/code/mcshaders/Alpha-Piscium-4` (wt4), branch `2.0/light-sampling`. The clean starting HEAD checked for this prompt was
`6d69b145ec7a1e70ed719886549242e4fd93542f`. Recheck current differences without resetting; freeze the pre-implementation shader source as baseline.
Follow the [repository rules](../../../AGENTS.md), [shader rules](../../../shaders/AGENTS.md), [scripts rules](../../../scripts/AGENTS.md),
and [current design](light-sampling-design.md). This scope supersedes the old weight batch for this task; do not restart the seven-method matrix.

## Verified mechanism and minimal candidate

[LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl) builds a subtree inside each 16³ brick and publishes its root at
`VOXEL_GRID_BRICKS + brickMorton`. This Morton index belongs to the camera-relative voxel grid; fixed outer Morton levels merge the brick roots.
Crossing a world 16-block boundary shifts the window origin and can change outer grouping even when the reachable emitter set is unchanged.
Leaf AABBs are already in world coordinates; changing their coordinate representation alone cannot fix this grouping dependency.

Change only the **outer light-tree slot receiving each brick root**, using low world-brick coordinate bits within the existing dense storage.

```glsl
ivec3 worldBrick = (cameraPositionInt >> 4) - ivec3(VOXEL_GRID_SIZE / 2)
    + ivec3(morton3D_30bDecode(brickMorton));
uvec3 slotCoord = uvec3(worldBrick) & uvec3(VOXEL_GRID_SIZE - 1);
uint lightBrickMorton = morton3D_30bEncode(slotCoord);
// Publish the existing brick subtree at VOXEL_GRID_BRICKS + lightBrickMorton.
```

Reuse existing `brickOrigin` or integer computations. For negative coordinates, use unsigned low-bit masking, not abs, clamping or a signed-remainder remapping.
Current `VOXEL_GRID_SIZE` values are 16, 32 and 64. G consecutive bricks on each axis map bijectively to G modulo slots, so the current G³ window has no slot collisions.
For the same world emitter set, keys and semantic parent membership stay unchanged as the camera brick moves; atomic allocation addresses need not be identical.
The discussion side checked 39 one-dimensional integer mappings across all three G values, negative/zero/large coordinates, and an example where old relative keys
change neighboring groups while the new keys do not. This is not GPU acceptance.

| Order | Entry | Responsibility in this change |
|-------|-------|-------------------------------|
| 1 | [Voxelization](../../../shaders/techniques/voxel/Voxelization.glsl), [VoxelAllocatorMP_Remap](../../../shaders/pass/begin/VoxelAllocatorMP_Remap.comp.glsl) | Preserve the camera-relative voxel window, allocation, material addressing and window clipping |
| 2 | [LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl) stages 0/1 | Keep clearing every outer leaf slot each frame; preserve brick-local subtrees and only change root publication slots |
| 3 | Outer reduce stages in the same build pass | Retain contiguous children, existing reduction levels, actual world AABBs and mass |
| 4 | [LightSampling](../../../shaders/techniques/debug/LightSampling.glsl), [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl) | Preserve Half-space A, four-way/32B nodes, cube sampling and actual PDFs; add diagnostics only where needed |

Do not modify `voxel_allocatedBricks`, voxel Morton keys or the shared voxel tree to match light-tree slots. Preserve binary/eight-way compatibility and the existing 48B candidate layouts.
Do not add permanent user switches, another node pool, general sorting, H-PLOC, cuts, visibility guiding, temporal smoothing or RNG changes to hide the discontinuity.
Edit the candidate source directly and compare against frozen revisions; do not retain the old builder as a parallel product path.

## Required counterexample and limits

This is a finite tree with periodic world-coordinate mapping, not an unbounded world Morton prefix tree. Wrapping can reduce spatial compactness:
for G=16 and an active world-brick x range `[1,16]`, bricks x=1 and x=16 at identical y/z map to slot x=1 and x=0. They merge at a low level despite being 240 blocks apart.
Collision freedom and temporal stability do not establish acceptable sampling quality. Inspect wrapped cluster spans, important-emitter PMFs and image quality.
If quality regresses materially, preserve the candidate and evidence and report back; do not claim success from stability alone or silently expand into a larger dense grid/general sorting rebuild.
Freezing the window or delaying rebuilds only defers the discontinuity and is not this repair.

Voxel-window entry/exit, distance-prioritized allocation under capacity pressure, and changes to shadow-submitted geometry may still change emitters or occluders.
This candidate does not promise invariant images when those inputs change. At a fixed world receiver with the same integrand and correct PDFs, changing only the proposal
should not change the converged expectation. A 32-frame appearance change is not automatically bias, and this theoretical property does not rule out real input loss.

## User failure scenes

The wt4 Vibris listing on 2026-10-31 verified these four native 1920×1080 presets. Preserve the user presets; do not substitute old ls1–ls5 or silently reduce resolution.

| Preset | SHA-256 |
|--------|---------|
| `light-sampling-failure-moving-1` | `3569c88f4694f5a0e294b6647099fcc3d2f097d97d5d9d3bcaf3c8f0b3059109` |
| `light-sampling-failure-moving-2` | `3096814726434fe24fb637bfe950f7eee02bf85d8a1392504521b201da97cdcf` |
| `light-sampling-failure-moving-3` | `25d9adaab2e85b2e8daf993b301c3438562a5556d09b2f79c20077638ab3034b` |
| `light-sampling-failure-moving-4` | `8133f617323d1587d3d79c472e2849911dfc9579962526e73b6cfc41ec671551` |

Read actual poses, dimensions and configurations; identify boundary pairs from coordinates rather than assuming numbered presets are consecutive motion frames.
Record integer camera positions, camera bricks, window origins and actual frame indices. Reproduce with old A, then compare candidate A; always include independent
**Cosine BRDF**, not Cosine B. The current Vibris action interface has no continuous camera-move action: comparisons of four static poses are not continuous-motion
video or observations of the first frame after crossing. Use an existing supported continuous-capture capability only after verifying its capability and timing;
do not build new control infrastructure for this task.

## Implementation and targeted acceptance

1. **Inputs and attribution.** Reuse reachable leaf identity and recursive topology from [weight analysis](../../../scripts/light_sampling_weight_analysis.py).
   Record world emitter membership/mass, occluder identity, allocation counts and overflow for each pose. Its existing `voxel_identity` uses camera-relative brick indices
   and hardcodes grid16; it cannot directly establish identity across cameras or grid sizes. Reconstruct integer world coordinates from actual G, camera brick and Morton
   decoding, then canonically sort world position plus full material data. Raw buffer addresses/byte hashes establish artifact integrity only. Preserve and separately
   attribute poses with different inputs; do not call them a same-input topology comparison.
2. **Mathematical and data checks.** Cover grid16/32/64, negative coordinates, 16-block crossings, modulo-G seams, empty/single-light cases and binary/four/eight-way trees.
   For one fixed emitter set under at least two window origins, require identical reachable leaves, mass, semantic topology and complete PMFs at fixed world receivers,
   with explicit floating-point tolerances. Independently audit descendant AABB containment, child ranges, normalization, support and sampled PDFs.
   A few random hit rates are insufficient. If all four runtime poses have different emitter sets, use one actual reachable set as frozen offline input to isolate grouping;
   do not label that offline check runtime same-input evidence.
3. **Runtime and images.** Verify loaded source/config/preset, Target Iris compilation and default A. Compare old A, new A and Cosine BRDF in all four scenes under
   direct-only settings and equal sample budgets, disabling existing history/reuse/denoising interference. Retain 1 spp, a mean over 32 consecutive frames, raw variance
   and nonfinite counts. Match world positions, normals and materials on common visible surfaces; identical screen pixels from different cameras are not identical receivers.
   Inspect the original affected region and seam guard regions. Add matched high-sample ROI references only where converged brightness or quality attribution remains unclear;
   do not repeat the full seven-method matrix. Report failure to reproduce, input drift and missing samples. Keep nonfinite events separate and use common complete pixels;
   never silently replace them with zero.
4. **Cost and regressions.** Verify unchanged node capacity/layout. Measure the complete build and initial passes plus a stable workload sentinel on a representative preset
   using adjacent forward/reverse comparisons. Do not substitute whole-frame time or call one noisy pair a speedup. Cover affected compilation configurations; emphasize
   default four-way runtime validation and add only relevant smoke cases instead of unrelated matrices.
5. **Delivery.** Commit the scoped implementation, necessary diagnostics, bilingual status and reproducible results. Preserve raw captures, sidecars, source hashes and
   canonical identity/PMF results, with an offline viewer organized by preset × method × 1 spp/32-frame mean. Separately assess reproduction, mechanism validation,
   visible improvement and unresolved issues; compilation alone is not completion.

Reuse applicable portions of [weight reference](../../../scripts/light_sampling_weight_reference.py), [near/far analysis](../../../scripts/light_sampling_nearfar_analysis.py)
and existing capture facilities. Keep a separate batch under the [light-sampling build directory](../../../builds/light-sampling/) and tracked summaries in the
[experiment directory](../../../experiments/light-sampling/). Preserve earlier batches; delete only matching remote manifests after local copies and hashes are verified.
Run long analysis/compilation at the lowest process priority while keeping timing conditions normal. Preserve the current Minecraft instance: no restart or runtime substitution.
Follow durable-job wait instructions without polling just to report progress. Run `git diff --check` and `git diff --cached --check` before committing.
Do not modify `2.0/dev` or force push.

## Proactive completion notification

After necessary validation and artifact delivery, call `send_message_to_thread` with target `01a18c0e-8678-7800-8ab8-04b69f072f12`, host `local`.
State the actual completion scope and include implementation commits, absolute viewer/result paths, all four preset outcomes, same-input topology/PMF results,
wrapping quality, input drift and nonfinite limitations. Explicitly request: “请现在复查本轮移动稳定性修改”. Report a genuine blocker too, without calling the task complete.
Return-message authorization comes from the user's original request quoted above; if needed, verify the corresponding user turn in the discussion chat instead of treating
an inter-agent message itself as authorization. The discussion side independently reviews the delivery; the experiment report does not replace acceptance.
