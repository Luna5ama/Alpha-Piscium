# Four-hour GI and voxel experiment, November 22, 2026

The exclusive GPU and dedicated game window ran from 03:03:13 to 07:03:13 PST (11:03:13 to 15:03:13 UTC). The preparation run has a separate record. Board input 110 superseded the earlier clock. Frozen work item `gi-algorithm-optimization-4h-20261122@3ea4b5db2402f507b626d187e5858fe7331cc9d0`. Compare against `2.0/dev@5ba8718d4245f3d2cf4db627bfbe3258e18aaaea` on the isolated branch `codex/gi-algorithm-20261122`. The hybrid screen-space GI release uses its own checkout.

## Final paired result

Final shader revision `9475d8a0a225c7665ed551e46a0fe7fb8454e738`. Frame samples include the 29 active GI compute stages and the two terrain shadow scopes that perform voxelization. The latter also contain terrain raster work. The compute-only and expanded columns make this shared cost visible. A and B are the averages of the arm medians across both independent ABBA groups. The interval across all four adjacent A−B pairs describes the observed variation. All times are GPU milliseconds per frame.

| 720p preset | Compute A → B | Expanded A → B | Expanded savings | Expanded % |
| --- | ---: | ---: | ---: | ---: |
| basilica-1-720p | 5.293 → 4.751 | 7.370 → 6.827 | 0.535–0.546 | 7.36% |
| non-cube-1-720p | 5.297 → 4.848 | 9.556 → 9.091 | 0.445–0.490 | 4.87% |
| night-gi-1-720p | 4.799 → 4.382 | 9.087 → 8.668 | 0.409–0.437 | 4.62% |
| mirror-room-1-720p | 2.418 → 2.323 | 3.716 → 3.618 | 0.087–0.108 | 2.64% |

- `baf5a52a-52c7-47f3-9c65-d4d2071e6fb3`, 224 timed frames per arm.
- `9a64fec6-cd6d-415f-b4b1-0f565efdd171`, 256 timed frames per arm.

The managed priority log confirms Realtime=5 on 32/32 final arms. WDDM engine samples cover 31/32 arms. Peak external readings were chrome/3d 8.5%, System/copy 4.0%, dwm/3d 1.9%, ChatGPT/3d 1.2%. Arm-level observations, both terrain scopes, whole-frame sentinels and resolved settings are archived.

## Measurement

RTX 3080 Ti, 1280×720, Vibris PID 37844, managed task priority Realtime=5, idle=0, CPU Normal. Each arm loads the explicit revision and checks preset, live scene, shader snapshot and effective settings. For each sampled frame, add the active typed GPU program samples covering voxel maintenance and build, cache, initialization, visibility, ReSTIR and both denoising passes. Report the median of these frame sums. Default configuration executes 29 stages. The external 3D/copy/video readings, actual per-arm priority, whole-frame times, shadow sentinel, durations and exact membership are in `timing-and-provenance.zip`. The default settings fingerprint starts `4c082fd98204`. Individual stage medians do not sum exactly to the chain median.

Four-scene ABBA `a9ae4f16-52bf-497c-b19b-64330196c7d7` sampled 224 frames per arm after 144 warmup. The independently loaded second group `2d1eff40-5781-4d68-8a50-8f51c9b74455` sampled 256 after 160, with changed scene order. The first group used `d8a22a5b7`, the second `c0688edba`. The latter changes the spatial-reuse-off branch. Values below are milliseconds per frame.

| Default 720p | First A / B / B / A | Second A / B / B / A | Adjacent saving, all four pairs |
| --- | --- | --- | --- |
| Basilica | 5.288 / 4.796 / 4.791 / 5.282 | 5.297 / 4.799 / 4.801 / 5.290 | 0.489–0.498 |
| Non-cube | 5.290 / 4.907 / 4.907 / 5.292 | 5.300 / 4.910 / 4.912 / 5.297 | 0.383–0.390 |
| Night GI | 4.786 / 4.395 / 4.407 / 4.776 | 4.822 / 4.449 / 4.448 / 4.819 | 0.370–0.391 |
| Mirror room | 2.433 / 2.389 / 2.394 / 2.433 | 2.430 / 2.386 / 2.391 / 2.422 | 0.031–0.044 |

In the second group, shadow terrain solid varied 0.002 ms within non-cube and basilica and 0.006 ms within night and mirror. Chrome reached 4.6% 3D, with smaller DWM, ChatGPT and copy work. Long-arm engine samples are sparse (1–3 readings inside each measured draw interval), so driver telemetry provides context and paired program timers determine the result. The frozen baseline already contains the previous spatial-BRDF improvement `11a9a9755`; its savings are excluded from this comparison.

## Changes retained in the measured branch

- `d43df1790` walks previous-frame occupancy before recycling the physical material pool. It clears material quads that could contain data and removes the normal full-pool clear, saving 0.040–0.063 ms in the second default group. The build timing stays within 0.001 ms.
- `2052cf863` reuses a traced surface when the cache owner matches. A different owner keeps the original surface lookup. The material-value oracle compared 7,210,671 matching surfaces with zero changes to albedo, roughness, F0 or dielectric and observed 1,017 different owners exercising the retained lookup.
- `7c41db142` writes the two UNORM10 normals, UNORM8 material and original float depth into an existing RGBA32UI atlas slot during history fix. Both blur stages read one packed record per tap. The centre and the original depth border behavior are preserved. The history-fix cost rises by 0.010–0.025 ms. In basilica, the first and post blurs save 0.267 and 0.096 ms. Two independent isolated mirror ABBA groups (`c6a43172`, `8f329a83`) measured adjacent regressions of 0.0164/0.0154 and 0.0210/0.0205 ms, within the requested 0.050 ms limit.
- `d8a22a5b7` regenerates two model textures after pruning coplanar same-facing rectangles fully covered by a later quad with the same encoded centre, normal and alpha lookup. Companion Shadesmith commit `e794945` on `codex/gi-covered-quads-20261122` has 54 passing tests including 14 geometry cases. All 815,424 LUT words were checked. 355 of 3,713 model lists shrink, referenced quads fall from 61,183 to 59,528, and 53,727 LUT cells visit fewer quads. Block properties and the first two material LUTs remain byte-identical. Full asset generation used the compatible `2dc72797a` generator base.
- `c0688edba` removes the temporal pass's shading result when spatial reuse is off. The paired shading pass already produces the full result. C23 falls from about 0.251 to 0.237 ms in the measured off branch.

## Correctness and image evidence

- A doubled GPU path compared the old and compact model on exactly the same 5,285,649 primary rays and 2,226,905 bounded visibility segments. Changed hits, material, normal, exhaustion and occlusion were all zero. The original budget exhausted on 787 primary rays. An 8,192-step reference was available for exhaustion and changed rays. The shipped traversal budget and cell-crossing rules are unchanged.
- Exhaustive GPU scans after the sparse clear covered grid 16, 32 and 64, including saturated pools of 256 and 1,024 slots and radiance cache off. Both grid-64 scans inspected 67,108,864 physical words. Movement crossed brick boundaries and exposed previously hidden regions, with 17 integer camera-position changes and two brick changes. Residual material words were zero.
- The same-frame denoiser instrument runs the old and new filters on the same input, compares the ordered tap addresses, accepted counts and FP16 output per diffuse and specular lobe. Across 10 pass/scene/setting comparisons there were 9,211,426 active pixel-pass cases and 138,034,484 sample addresses. All address sequences and counts agreed. Relative RGB error is the Euclidean difference divided by the baseline RGB norm floored at 1e-6. P99 = 0, maximum = 0, fraction above 1% = 0% for both lobes. Eight default comparisons account for 102,439,824 addresses. Two instrument A/A checks also returned zero.
- The earlier five-commit stack used independent normal-output A/B/A scene reloads, with static, nine moving frames and returned-camera images in each scene. Camera displacement was 1.5 blocks and 14 degrees. The full image sets and comparison scripts are in the archive. At the middle turn, A/B versus A/A mean absolute RGB difference as a percentage of full scale was 0.036/0.702 for basilica, 1.129/1.217 for non-cube, 0.745/0.948 for night, and 0.011/0.006 for mirror. Stochastic/reprojection variation is visible in both sequences at the turn. No persistent new leak or outline was observed in these views. Independent captures do not reproduce identical random histories.

## Counterexamples and scope

- Half-rate surface updates saved 0.153 ms in non-cube and 0.323 ms in night but cost 0.051 ms in basilica. Night specular mean dropped 11.32% against A/A changing 1.09%, and diffuse flicker rose 8.94% against 0.65%. An earlier interleave raised brightness by 28–44%. Both were discarded.
- A history swap that appeared to save 0.58–0.69 ms aliased the atlas. Dedicated history storage removed that apparent win and used about 44 MiB for four streams. Compaction, packet dispatch, build passes and whole-screen/cache changes were timed with their previous and next passes. Splitting the initial direction pass moved 0.02–0.03 ms to its successor. Direct physical scans cost another 0.18–0.35 ms. Sharing spatial visibility geometry saved 0.009–0.015 ms in C30 but added 0.019–0.027 ms in C29.
- Exact tap variants that share kernel loops or prune kernel tails were slower. Moving axis-alignment handling outside the model loop added roughly 1.5 ms to non-cube initial tracing. An empty-entry traversal changed one of 7,109,485 primary rays without satisfying the original-exhaustion/high-budget criterion. All remain experiment patches.
- Precision and timings apply to this driver, device, presets, and branches exercised. Real Voxy and the separate hybrid GI release are outside this frozen experiment. GPU pass timers, workload counters and same-ray comparisons resolved the decisions here.

## Configuration checks

Separate baseline/candidate pairs exercised the following branches. These are short screens for configuration coverage.

| Settings | Scene | Stages | A → B (ms) |
| --- | --- | ---: | ---: |
| Cache off, grid 16 / pool 256 | non-cube-1-720p | 19 | 4.282 → 3.796 |
| Guiding off, grid 32 / pool 1024 | basilica-1-720p | 26 | 4.763 → 4.326 |
| Spatial denoising off | mirror-room-1-720p | 27 | 2.193 → 2.138 |
| Spatial denoising off | night-gi-1-720p | 27 | 4.054 → 3.967 |
| Decorrelation, 14 neighbors, 12 / 4 blur taps | non-cube-1-720p | 31 | 6.132 → 5.716 |
| Reference GI mode 2 | night-gi-1-720p | 30 | 4.904 → 4.300 |

## Linear GI before display mapping

For the earlier stack, the 400×230 centre region of each lobe was sampled across 128 frames in eight batches for each independent A/B/A run. The compared images show accumulated linear radiance with separate diffuse and specular atlas regions. A/B and A/A ratios below are expressed as percentage change from the first baseline. Per-pixel results, standard errors, counts and raw buffers are archived. All 24 measured lobe/arm captures had 92,000 valid pixels per batch and no nonfinite samples or incomplete batches.

| Scene | Lobe | Mean luminance A/B / A/A | Flicker A/B / A/A | Normalized RGB RMS A/B / A/A |
| --- | --- | ---: | ---: | ---: |
| basilica-1-720p | diffuse | -0.78% / -1.09% | -0.31% / -2.52% | +6.68% / +6.56% |
| basilica-1-720p | specular | -0.18% / -1.69% | +0.22% / -1.73% | +20.79% / +24.20% |
| mirror-room-1-720p | diffuse | -0.39% / -0.04% | +0.60% / +0.29% | +2.43% / +2.64% |
| mirror-room-1-720p | specular | -0.31% / -0.18% | -0.62% / -0.45% | +3.48% / +3.29% |
| night-gi-1-720p | diffuse | +0.42% / -2.02% | -1.43% / -1.67% | +3.76% / +1.90% |
| night-gi-1-720p | specular | -0.99% / -0.30% | +0.23% / -0.55% | +11.68% / +9.83% |
| non-cube-1-720p | diffuse | -0.22% / -1.55% | -0.02% / -1.11% | +5.26% / +5.87% |
| non-cube-1-720p | specular | -2.79% / -1.46% | -2.82% / -1.27% | +15.14% / +15.50% |

Night diffuse A/B temporal variation and non-cube specular brightness differ from their A/A samples. The latter changes by −2.79% relative to the first baseline and −1.34% relative to the second. These independent stochastic comparisons bound the scene evidence; the deterministic material, hit, occlusion and tap tests above compare identical inputs. Additional hardware, resolutions and long camera paths remain unmeasured.


## Final implementation and recovery

The tree builder in `b76f0b77e` scans material data only for leaves present in the previous root or marked during current shadow voxelization. An allocated brick reuses its occupancy word as a 32-bit mask of newly touched leaf pairs. The builder writes the final 0/1 flag for the next allocation. Unallocated bricks keep the raster occupancy flag. During raster, the previous root is read-only. There are no new SSBOs or dispatches.

In `3587c8df-d373-4aa2-8153-6e4505789012`, this change alone, relative to the five-commit stack, saved 0.054/0.059 ms in non-cube, 0.054/0.048 ms in basilica, 0.097/0.045 ms in night and 0.057/0.052 ms in mirror including both terrain shadow scopes. The night forward pair includes a 0.030 ms terrain-shadow shift. The independent in-place root prototype also saved time in all eight forward/reverse pairs, but wrote the old root during raster. The chosen version keeps reads and writes in separate buffers. An unconditional per-brick atomic shifted work to raster. Encoding voxel eligibility in the material-flag LUT gave a small scene-dependent raster saving in screening and remains an experimental generator change.

An independent shadow geometry writer marked each occupied brick in separate scratch storage. The builder compared new occupancy and opaque bits against a full scan of current physical materials. Eight snapshots across four scenes checked 1,504,576 leaves (567,377 nonempty) and 23,509 allocated bricks. All leaf bits, opaque bits, allocated and unallocated occupancy checks agreed. A sticky failure flag also remained zero through the camera moves, including two brick crossings each in non-cube and basilica. The 256-slot cache-off and 1,024-slot configurations took the same tests. Group `636df249` and the preceding full-material oracle cover an independent root implementation.

A compressed traversal that stopped skipping after 160 checks improved initial tracing by 0.072–0.126 ms in screening. Among 7,109,287 matched rays, 504 results or continuation states changed. The 8,192-step original explains 274 as previously exhausted rays now reaching the correct hit, leaving 230 unexplained. It was discarded. Replaying long rays to recover the old budget added roughly 2.1–2.3 ms. Reconstructing the exact final subcell erased the saving. Separating temporal splat from voxel trace made the inner trace faster but added a 0.062 ms dispatch, leaving the full chain slower.

`9475d8a0a` records zero temporal confidence for pixels without a reusable primary. C21 can skip a reservoir read, and C23 gathers one confidence channel in place of four texture gathers. The current-frame spatial pass retains its original temporal reservoir. In `4d905f0a-3f45-4e2a-832c-26cf1f8bf6d7`, the incremental change saved time in all eight adjacent comparisons, from 0.005 to 0.023 ms. The same-input GPU oracle compared 6,962,464 nonzero primary records and 29,245,668 gathered confidence values. Invalid associated reservoirs and confidence mismatches were both zero.



### Timing tails and workload controls

GPU driver 617.42, 12 GiB. Expanded P95 is calculated on the same per-frame sums as P50. Each range below covers the four adjacent baseline-minus-candidate differences in the two independent groups. The independent terrain sentinel column is the largest absolute A/B difference.

| Scene | Expanded P95 saving (ms) | Terrain sentinel max change (ms) | Outer composite P50 saving (ms) |
| --- | ---: | ---: | ---: |
| non-cube-1-720p | 0.386 to 0.521 | 0.0010 | 0.191 to 0.324 |
| basilica-1-720p | 0.537 to 0.562 | 0.0010 | 0.379 to 0.416 |
| night-gi-1-720p | 0.367 to 0.506 | 0.0010 | 0.250 to 0.371 |
| mirror-room-1-720p | 0.082 to 0.102 | 0.0010 | -0.071 to 0.101 |

The frozen baseline and final sources were visually inspected at rest and during camera motion. The high-contrast mirror turn shows a larger inter-run difference than its two baseline captures. At the middle turn the difference is 0.203% versus 0.006% of display RGB full scale, rising to 0.262% versus 0.052% at the last turn frame. Reflected edges and silhouettes stay aligned at the sampled poses. These observations do not establish identical time histories.

The whole-frame wall-time sentinel shifted -4.35 to +5.88 ms across adjacent pairs. It is affected by frame pacing and external activity. All sixteen expanded GPU comparisons improved, and the independent terrain GPU sentinel moved at most 0.0011 ms. The whole-frame series should not be used to infer a matching FPS change.


### Final normal output and independent histories

The final revision was reloaded in all four presets and sampled through the same nine-frame camera motion as the earlier baseline controls. The static display-RGB figures below use a denominator of max(baseline RGB norm, 0.01), after mapping channel values to [0,1]. Quantiles and percentages include the entire frame. These independently seeded paths include sampling and history variation. The same-frame FP16 rounding comparison is in the correctness section.

| Preset | P99 A/B / A/A | Maximum A/B / A/A | Above 1% A/B / A/A | Mid-turn mean absolute A/B / A/A | Returned A/B / A/A |
| --- | ---: | ---: | ---: | ---: | ---: |
| basilica | 23.53% / 22.26% | 367.74% / 352.29% | 87.15% / 81.35% | 0.79% / 0.70% | 0.87% / 0.75% |
| non-cube | 28.07% / 24.17% | 279.02% / 279.45% | 93.53% / 88.71% | 1.03% / 1.22% | 0.89% / 1.06% |
| night-gi | 41.57% / 30.10% | 350.48% / 403.22% | 92.18% / 89.62% | 1.05% / 0.95% | 0.83% / 0.87% |
| mirror-room | 14.29% / 15.32% | 183.45% / 121.87% | 83.06% / 78.87% | 0.20% / 0.01% | 0.69% / 0.54% |

The final-frame comparison JPEG presents the original camera pose and middle turn. Full-resolution captures, intermediate motion frames and all baseline control images are in the registered archives.

An additional A/B/A used 512 frames in 16 batches on the two earlier outliers. Percentage changes are B relative to A1 / A2, followed by A2 relative to A1.

- night-gi-1-720p long-diffuse, mean luminance -0.13% / -0.23%, A/A +0.10%. Flicker +0.64% / +0.31%, A/A +0.34%.
- non-cube-1-720p long-specular, mean luminance -0.92% / -1.21%, A/A +0.29%. Flicker -0.59% / -1.33%, A/A +0.76%.

- Final configuration load `8e92547e-d787-4109-ac23-040449752eea` on mirror-room-1-720p, requested {"SETTING_GI_SPATIAL_REUSE": "false", "SETTING_GI_TEMPORAL_BACKUP_SAMPLE": "false"}, resolved settings `624bd94d25cac1bf6cd9ea4b8a400c1cd2dcbce1efc4272e9a443c9586609a79`.
- Final configuration load `d70736f0-0819-4a28-9e69-b5c22703ac4b` on night-gi-1-720p, requested {"SETTING_DENOISER_SPATIAL": "false", "SETTING_GI_USE_REFERENCE": "2"}, resolved settings `0a68a41ee37f58723d3fac96466b9fcd82f6932cdab0c3e9ed59375eb1f1608d`.

Restored baseline and default settings in mirror-room-1-720p with successful action `a4e09567-ebc5-44b2-8d81-68824843596d`. PID 37844 was externally read back at Idle=0, CPU Normal. Core reported an empty queue and could accept work. At the final audit all 215 experiment workflows were terminal. 394 temporary source copies (19,513,998,873 bytes cumulatively) were consumed and removed. Core retains its own receipts and metadata.

A shared-memory compact leaf list was separately screened in `432aacf6-5c6e-4822-b214-5b99f3808487`: non-cube-1-720p tree build 0.142 → 0.145 ms, full compute chain 4.861 → 4.862 ms; basilica-1-720p tree build 0.117 → 0.118 ms, full compute chain 4.652 → 4.652 ms. This prototype was not included in the final shader.



For each 512-frame run, the 16 batch means were grouped into four successive 128-frame block means. Below are the minimum and maximum block means relative to the mean of A1. Blocks share temporal history. Their ranges describe observed within-run variation.

- night-gi-1-720p long-diffuse, A1 -1.70% to +1.53%, B -2.46% to +2.32%, A2 -1.74% to +2.03%.
- non-cube-1-720p long-specular, A1 -2.74% to +3.63%, B -4.53% to +3.92%, A2 -3.37% to +3.74%.


<!-- vela:evidence {"run":"eacfb7ff-0598-4f9e-9818-e970cdd19849","path":"timing-and-provenance.zip"} -->
<!-- vela:evidence {"run":"eacfb7ff-0598-4f9e-9818-e970cdd19849","path":"quality-and-search.zip"} -->
<!-- vela:evidence {"run":"eacfb7ff-0598-4f9e-9818-e970cdd19849","path":"visual-comparison.jpg"} -->
<!-- vela:evidence {"run":"26cb27a2-a2e9-432d-b322-dcc382ac09f9","path":"final-images-and-linear.zip"} -->

## Sealed evidence

- timing-and-provenance.zip, 62283893 bytes, SHA-256 `51d1f3749da1c7fc39f70bcfe9146035f5699dcdcdae4d051681dcc6641d3570`.
- quality-and-search.zip, 937231219 bytes, SHA-256 `f7d16754ac3c5bbcc2859f3aecd105fd4e017da686c493db8c5b0f670c321cb6`.
- final-images-and-linear.zip, 119592515 bytes, SHA-256 `85343872e621f596604be609b467383b088774a1ef609f3619d116c8167009a3`.
- visual-comparison.jpg, SHA-256 `cc979a52bc728e1c499e1291450e0c86d309a586eadb073c189d5c5df1c5ff5b`.

## Local 2.0 hybrid GI integration, 2026-11-25

The local `2.0/dev` integrates the measured stack at `5dff1e8f` with the published hybrid GI and old Iris bindings at `d7a05fe4`. The published ZIP built from `77cb79df` is the comparison baseline. The SPBR21 material experiment stays on its own branch. This integration and its acceptance records are local.

For compatibility, load the merged ZIP in the existing official Iris 1.10.4 mirror-room setup with RC on and path guiding both on and off. Read the bound buffers and GI output. Repeat the official Iris 1.10.7 Jungle view with Voxy enabled and check native LOD depths, finite diffuse and specular GI, and the active pack. Use the existing [binding](../rendering/iris-binding-acceptance.md) and [hybrid GI](../rendering/hybrid-gi-acceptance.md) receipts for the camera, readback and reference. Check near-screen endpoints, voxel hits and both history frames where the integration changes their treatment.

For performance, compare the exact merged shader contents against the published hybrid contents in the original 720p scenes. Use adjacent forward and reverse pairs, stable settings, the affected GI and terrain-shadow sums, and the same quality and drift checks as the four-hour experiment. Include a spatial-reuse-off check of the single shading path. Record actual loaded revisions, measured differences and any quality trade-off. Use complete groups after existing reservations and preserve evidence in the existing run system. The earlier per-scene savings above apply to their original baseline.
