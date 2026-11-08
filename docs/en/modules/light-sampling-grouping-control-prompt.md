# Light Sampling Same-Input Grouping Control Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-grouping-control-prompt.md)

## Task and authorization (2026-11-08)

Complete a same-input live quality comparison of original camera-relative and World Prefix grouping in ls1/ls5, separating grouping changes from historical emitter-population changes. The user recalls increasing render distance from perhaps 16 or 24 to the current 32 chunks, then authorized proceeding at our discretion. This is a user-reported causal candidate, not verified historical settings. Do not expand this round into a 16/24/32 distance sweep.

Retain experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` as executor and discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12` as independent reviewer, under the existing human authorization for experiments, necessary Vibris use, dispatch and bidirectional completion/substantive-blocker reports. Use host=`local` and existing model/effort. wt4 is `I:/code/mcshaders/Alpha-Piscium-4`, branch `2.0/light-sampling`, preparation HEAD `4e6b74baf`. Inspect later changes before starting; do not reset. The previous no-new-GPU instruction applied only to the preceding read-only investigation; this prompt authorizes this supplementary experiment.

## Starting evidence and question

Read the [historical-image investigation](light-sampling-design.md#historical-october-31-images-versus-current-results-2026-11-08) and [record](../../../experiments/light-sampling/20261108-historical-comparison.json). Old/current emitter counts are 3,921/10,235 in ls1 and 877/2,172 in ls5. Matching preset/scene hashes did not fix actual voxel inputs. Current same-input M2 favors prefix model 6 over the original camera tree at two ls1 points; a controlled whole-image comparison on current inputs remains missing.

Current capture source is `e9132607ec1540820fc0d7f53772579a0598e50f`; original camera grouping is available at `78823b540d1164190cd86566f783707ee63c2c8f`. Prepare two frozen experimental sources from the same current code, varying only the necessary outer layout/build and generated contract. Original grouping merges brick Morton keys inside the camera window; it is **not the abandoned periodic world-modulo candidate**. Comparing the entire historical commit against the current pack would not isolate grouping.

Both sources retain identical brick-local construction, emitter classification/mass, four-way 32B nodes, cube sampling, model-6 weights/PDF and true evaluation. Different outer capacities/addresses are necessary implementation differences; verify generator/resource contracts. Preserve experimental sources through separate local refs or existing isolated source preparation. Add no production setting for this one-off comparison and switch no default. The experiment side owns source edits; the discussion side only maintains the task documents this round.

## Runtime conditions and minimal matrix

Check the shared Minecraft operator, active jobs and restoration state before switching scene/pack. If another experiment still owns the runtime, continue offline preparation and report the concrete dependency instead of taking over. Retain necessary Greenfield tick-freeze/restoration conventions. Do not restart Minecraft or its launcher.

Read and record actual client render distance, available server view/simulation distances and scene-loading state. Keep the actual render distance at the start of this round unchanged. The user reports 32; record any live discrepancy instead of labeling it confirmed 32. Missing historical settings do not block the same-input comparison, and a new shorter-distance run must not be presented as restoring the historical emitter set.

| Scenes | Method | Output |
|---|---|---|
| `light-sampling-1`, `light-sampling-5` | Original camera grouping + Half-space A + Near/Far (model 6) | One 1-spp frame and a 32-frame mean each |
| Same | World Prefix + Half-space A + Near/Far (model 6) | Same |
| Same | Cosine BRDF-only | Same |

Six cases, twelve images. This grouping diagnosis does not replace the established World Prefix A main-experiment baseline. Retain the previously requested native 1920×1080, G16, four-way, cube, direct-only, TAA/denoising/RC/reuse off, 256 warmup frames, 32 consecutive frames in eight batches of four, and existing exposure/display transform. Reuse the [combination protocol](light-sampling-nearfar-combination-prompt.md#fixed-comparison-matrix) for capture/image acceptance. Add no performance or Ward/lookahead matrix.

Bracket each scene capture with world emitter keys/masses, complete world voxel materials, actual receiver geometry, camera and projection identities. Both trees must share emitter/voxel/camera inputs. Their topologies may differ and must each match the appropriate camera-relative/world-prefix reconstruction. Verify successful Target Iris loading, effective settings, resource layout and actual selection PDFs for both sources. Emitter counts alone are insufficient; old physical node addresses cannot substitute for world identities. Reject compilation failures, overflow, drifting inputs and incorrect sources.

Existing World Prefix and BRDF captures may be reused only after matching complete input identities, evaluation paths, sampling conditions and display transform to the new capture. Otherwise recapture the three-method group only for the affected scene. Do not force new runs to reach historical counts of 10,235/2,172. Reuse the established scripts and source preparation instead of rebuilding the capture infrastructure.

## Interpretation, delivery and callback

Use common complete finite receiver pixels per scene for RGB variance, mean and zero-direct-light coverage, reporting nonfinite/excluded counts without brightness filtering. Reuse applicable fixed wall/pillar/floor regions with recorded geometry identities; do not choose regions based on results. The primary comparison is model-6 World Prefix / original camera, with BRDF as an independent control. Separate finite 32-frame statistics from the scope of existing measured per-light M2. Dark pixels alone do not establish bias, and different 1-spp random sequences do not establish an implementation regression. Add narrow repeated windows or diagnostics only when a concrete anomaly requires them; no default new per-light capture.

Deliver a separate twelve-image grouping comparison linked from the current topic docs, preserving the provenance of the existing 132-image and historical pages. Verify lossless conversion, OpenGL row orientation and actual browser switching. Suggested tracked summary: `experiments/light-sampling/20261108-grouping-control.json`; artifacts: `builds/light-sampling/20261108-grouping-control/`. Report actual distance, emitter/voxel identities for both scenes, source differences, whole-image/local results and tested/untested scope. Do not upgrade the user's historical-distance recollection to established causality.

Use PowerShell 7 without profiles and Idle priority for time-insensitive nonbenchmark work. If expensive work is delegated through Pyxis, verify its effective priority tier. Follow repository/generated-file rules and check both diffs before commits. Existing raw cleanup remains paused; do not bypass automatic approval rejections. Retain necessary new raw data for independent review with explicit paths and cleanup ownership. Reuse wt4 unless isolation actually requires a temporary worktree.

Verify and hand back the actual runtime state at completion, with no new jobs being submitted. Use `send_message_to_thread` with host=`local` and discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12` to return this prompt, commits, source refs, viewer/index, input identities and results, explicitly requesting independent review. Report substantive blockers through the same route with remaining independent work and evidence. Preparation alone is not capture completion.
