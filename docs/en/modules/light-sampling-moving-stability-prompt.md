# Light Sampling Motion Stability Experiment Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-moving-stability-prompt.md)

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
