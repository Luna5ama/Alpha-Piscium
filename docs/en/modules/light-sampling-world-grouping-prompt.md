# Light Sampling World-Space Grouping Design and Experiment Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-world-grouping-prompt.md)

## Authorization, status and starting point (2026-11-01)

In discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12`, turn `01a1961b-a945-7762-9500-333b877eeb35`, the user explicitly requested:

> 那就按世界坐标分组来改，写好方案让实验agent实现，再用那几个moving preset复测，完事让它通知你复查。

Experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` owns implementation, targeted retesting, commits and completion notification; the discussion chat retains independent review responsibility.
This page supersedes the implementation instructions in the [previous modulo prompt](light-sampling-moving-stability-prompt.md), including its old restriction against increasing node capacity. Preserve previous results and failed originals.
Use `I:/code/mcshaders/Alpha-Piscium-4` (wt4), branch `2.0/light-sampling`. At preparation, the worktree was clean at `fae8c413533d88e988db0b11b31552da1a1f9c48`.
Recheck differences before execution without resetting. Preserve that commit's bracketed input checks and offline derivative caching.
The starting shader contained periodic grouping rejected by quality review; it is not the accepted baseline. The main control is frozen camera-relative grouping at `6d69b145ec7a1e70ed719886549242e4fd93542f`.

The full world-prefix implementation and this experiment are recorded, with quality regressions and G64 input limitations; see the execution results below. Follow the [repository](../../../AGENTS.md), [shader](../../../shaders/AGENTS.md) and [scripts](../../../scripts/AGENTS.md) rules and [current design](light-sampling-design.md).

## Required contract

- Bricks remain world-aligned 16³ regions. Brick-local light trees and voxel addressing, clipping and allocation retain their existing responsibilities.
- **Every cross-brick parent cluster follows a fixed hierarchy of complete world coordinates.** Window coordinates determine which nodes need storage and their array addresses, never parent-child membership.
- Keep default four-way branching, Half-space A (model 1), 32B nodes, emitter sampling and actual PDF multiplication. Retain binary/eight-way and existing 48B model compatibility.
- The same emitter set and attributes in different containing windows produce the same ordered nonempty topology and fixed-receiver PMF, including the null outcome. Addresses may differ.
- Actual emitter-set or attribute changes can legitimately change quality and probabilities. No invariance is promised across input changes caused by clipping.
- Delete periodic keys and the old direct slot-based reduction path. Compare frozen source revisions instead of adding a construction-mode option, parallel product paths or a hidden fallback.

Do not change weights or enable near/far, visibility, lookahead, H-PLOC, regional cuts or ReSTIR. Do not freeze the window or hide discontinuities through temporal smoothing.

## Selected model: complete world prefixes with bounded layered storage

Use world-aligned reduction by level. The regular grid allows a first version without global sorting, hash tables, a general sparse-tree framework or GPU 64-bit integers.
Those mechanisms are unnecessary for this slice; world coordinates still cannot be used directly as array indices.

Let `a` be arity, `k=log2(a)` and `g=log2(G)`. Level 0 contains brick roots. Each higher level consumes k Morton bits, starting with x, y, z in repeating low-bit order.
Level l consumes `b=k*l` bits, with per-axis shifts:

```text
s(l) = (floor((b+2)/3), floor((b+1)/3), floor(b/3))
worldBrick = floor(worldBlock / 16)
u = worldBrick + 2^27
cell(l) = u >> s(l)          # Shift each axis separately.
```

Current signed 32-bit block addressing yields signed 28-bit brick coordinates. A fixed bias bijectively maps them into 28 unsigned bits per axis, preserving high bits and negative coordinates.
The logical world Morton key has 84 bits, but implementation needs only three 32-bit components and per-level shifts. Do not send full world coordinates through [morton3D_30bEncode](../../../shaders/util/Morton.glsl), which retains only 10 bits per axis.
The bias is a world constant, never camera-dependent; shifts do not exceed 28. This integer grouping contract does not extend existing floating-point world-position precision guarantees.

A parent identity is `(level, cell(level))`. Membership and child order use complete world bits. Subtract each layer's covered minimum only when calculating storage addresses.
Continue merging actual descendant AABBs and mass; do not substitute the geometric bounds of mostly empty world cells for actual light bounds.

## Implementable array layout

Let the current window's minimum world brick have biased coordinates `uMin`, and `uMax=uMin+G-1`. At each level:

```text
lo(l) = uMin >> s(l)
hi(l) = uMax >> s(l)
dim(l) = hi(l) - lo(l) + 1
index(l,c) = r.x + dim(l).x * (r.y + dim(l).y * r.z), r=c-lo(l)
```

Choose `L=ceil(3*g/k)`. Each axis then spans at most two world cells, leaving at most eight live records at the upper boundary.
Reserve one array segment for each level 1..L: **each parent cell owns a consecutive block of a child slots**. Index parent cells using the local linear index above; order child slots by the k complete world bits consumed at this level.
Siblings are therefore contiguous without changing parent identity when the storage origin moves. Do not modulo world identities again.

If `offset(l)` starts the child-slot segment for level-l parents, a level-(l-1) node is stored at:

```text
offset(l) + a * index(l, cell(l)) + childOrdinal(l)
```

`childOrdinal(l)` collects global Morton bits `[k*(l-1), k*l)` from low to high, yielding `[0,a)`.
Consumed axes can differ between levels; four-way grouping cannot be replaced with shifting x and y once at every level. Store level-L results separately in at most eight consecutive temporary records.

Reserve the first 32 outer nodes: root stays at 1; level-L temporary records can occupy 2..9; upper merging can write child records at 10..23.
One additional workgroup merges those at most eight records along the remaining complete world prefixes. Sort each group's children by ordinal; **do not assemble groups of four from temporary-array order**.
A single lane may perform this small reduction. Place dynamically indexed arrays in shared memory, following shader rules.
Continue to a common ancestor or the world root (84/k levels in total); do not force all level-L records into one four-way root.

At every level, zero nonempty children produce an empty node; one nonempty child copies its complete node to collapse a unary chain; at least two create a branch.
Lower layers may retain a fixed slots including empty children; upper layers pack nonempty children consecutively. Eight inputs produce at most 14 upper child records after unary collapse, within the 32 reserved slots.
Unary collapse prevents large coordinates and zero crossings from adding many probability-one traversal steps to every shading sample.

### Capacity and lifetime

Each axis has coverage bound `D(l,j)=ceil((G-1)/2^s(l,j))+1`, with `N(l)=product(D(l))`.
Reserve segments using those bounds and address them using actual dimensions. Capacity bounds must never change world parent identities.

```text
outerCapacity = 32 + a * sum(N(l), l=1..L)
offset(1) = 32
offset(l+1) = offset(l) + a*N(l)
localBase = outerCapacity
SSBO bytes = 32 + (outerCapacity + 262144) * nodeStride
```

| G | Binary outerCapacity | Four-way outerCapacity | Eight-way outerCapacity | Previous outer reservation |
|---|---------------------:|-----------------------:|------------------------:|---------------------------:|
| 16 | 10570 | 7620 | 7144 | 8192 |
| 32 | 74464 | 51732 | 46448 | 65536 |
| 64 | 558854 | 380432 | 333944 | 524288 |

These are derived layout counts, not performance measurements. Allocate for selected G/arity, retaining the existing 262144-node brick-local budget and overflow reporting. Do not increase emitter capacity to hide lost lights.
Both 32B and 48B strides must agree between [SSBO configuration](../../../scripts/shaders.properties), [GLSL layout](../../../shaders/techniques/debug/LightSampling.glsl) and construction.
Clear the entire reserved outer region each frame so edge padding and previous-frame nodes cannot leak. Continue reading only currently reachable brick-local nodes.
Use existing dispatch ordering and visibility rules between stages, without cross-workgroup spin waits. Upper completion must run before sampling.

## Implementation entry points and delivery order

| Order | Source / consumer | Required change |
|-------|-------------------|-----------------|
| 1 | [LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl), [LightSampling](../../../shaders/techniques/debug/LightSampling.glsl) | Add world-prefix/layer-address calculation; change brick-root publication, clears and outer reduction while preserving local construction and sampling formulas |
| 2 | [programs.main.kts](../../../scripts/programs.main.kts), [shaders.properties](../../../scripts/shaders.properties) | Register lower levels and final small upper reduction; resize buffers; remove old reduction wiring and regenerate wrappers/properties |
| 3 | [Moving analysis](../../../scripts/light_sampling_moving_analysis.py), [weight reference](../../../scripts/light_sampling_weight_reference.py) | Update real-tree decoding/reconstruction, especially the assumption that brick roots occupy `[G^3,2*G^3)`; explicitly distinguish historical modulo and new complete-world-prefix algorithms |
| 4 | [Capture](../../../scripts/light_sampling_weight_capture.py), [presets](../../../scripts/light_sampling_weight_presets.py) | Retain source/configuration and before/after input verification and original preservation; do not validate four-way trees with obsolete binary-only diagnostics |
| 5 | [Experiment records](../../../experiments/light-sampling/), [build artifacts](../../../builds/light-sampling/) | Separate batch `20261101-world-grouping`; bilingual status, originals, viewer, complete costs and completion notification |

The implementer may choose names and helper placement. Keep direct module-local data flow without a generic construction framework or new configuration surface.
Run `kotlin options.main.kts` from scripts to regenerate outputs for maintained sources. Run Shadesmith only for required bootstrap or atlas changes.
Use the latest available GraalVM JDK and PowerShell 7 without a profile. Compilation and offline analysis run at low priority; benchmarks retain normal priority.

## Preparation evidence and remaining obligations

The discussion-side [CPU layout prototype](../../../builds/light-sampling/20261101-world-grouping-design/layout_reference.py) and [results](../../../builds/light-sampling/20261101-world-grouping-design/layout-reference.json) compare 4374 layouts:
162 fixed-input cohorts across 27 window positions each, covering G=16/32/64, binary/four/eight-way, empty/single/multiple lights, negatives, zero, grid boundaries, coordinates exceeding the 10-bit Morton range and large coordinates.
The independent oracle uses full 84-bit Python integer keys, without reusing layout addressing. In four-way uncompressed levels, world brick x=1 and x=16 first merge at level 7; periodic truncation merges them at level 1.
These checks establish integer grouping, boundary addressing and capacity only. Real GLSL/GPU behavior, quality and cost still require independent verification.

Required acceptance:

1. An independent oracle validates world grouping, empty/unary collapse and complete ordered topology from the implementation. Cover window edges, old modulo seams, negative/zero crossings, G32 four-way depth, empty/single lights, contiguous child ranges and capacity.
2. Each emitter is reachable exactly once; AABBs contain descendants, mass is conserved, and there are no overflows or dropped lights. Independently audit sampled and conditional-emitter PDFs, complete PMFs and null probability. Compare semantics across equal-input moving windows, not buffer hashes.
3. Clearing and moving regions in/out leave no stale nodes; zero-light, single-light and history-reset cases work. Report float mass tolerances; atomic address changes are not regrouping.
4. Target Iris compilation/minimal runtime checks cover G16/32/64, binary/four/eight-way, 32B/48B and experiment-off using the smallest set of configurations covering the branches. Full-image quality uses default four-way A only.

## Retesting the four moving presets

All four were reread through Vibris on 2026-11-01: SHA values match the previous batch and native resolution is 1920×1080. Use the user's original presets without changing cameras or preset files.

| Preset | SHA-256 |
|--------|---------|
| `light-sampling-failure-moving-1` | `3569c88f4694f5a0e294b6647099fcc3d2f097d97d5d9d3bcaf3c8f0b3059109` |
| `light-sampling-failure-moving-2` | `3096814726434fe24fb637bfe950f7eee02bf85d8a1392504521b201da97cdcf` |
| `light-sampling-failure-moving-3` | `25d9adaab2e85b2e8daf993b301c3438562a5556d09b2f79c20077638ab3034b` |
| `light-sampling-failure-moving-4` | `8133f617323d1587d3d79c472e2849911dfc9579962526e73b6cfc41ec671551` |

- Compare **G=16 for continuity with the historical control** and **G=64 for the actual default** separately. Each G includes all four scenes with original camera-relative A, new world-prefix A and independent Cosine BRDF. G32 needs layout/smoke coverage only.
- Preserve modulo results as prior failure evidence without adding a fourth-method full-image matrix. Rebuild modulo offline only if same-input attribution requires it. Never pool different batches, grids or inputs into one statistical cohort.
- Quality retains these four specified presets' native resolution for continuity with the original images. Keep direct initial-only, RC/reuse/ReSTIR/denoising/TAA/visibility/lookahead off, and identical settings, sample budget, warmup and exposure.
- Save 1 spp, 32 consecutive-frame means, raw variance, valid counts, nonfinite and incomplete statistics. Use a common complete-pixel mask without silently zeroing data or cropping away regressions.
- Verify actual effective settings, source revision, scene/preset hashes and successful action receipts. Preserve camera/tree/full world-voxel identity before and after captures. Reconstruct world identity with actual G and camera, not relative slot numbers.
- Adapt capture tooling to the layout while preserving `fae8c4135` bracketed input and derivative-cache checks. Invalidate cache entries across revisions, configurations or input changes. Never overwrite originals or pair drifting inputs.
- Report whole-image and existing guard-ROI results per scene. Match world receiver position, normal, material and albedo, reporting cross-pose complete PMFs, null probability and same-surface statistics.
  Old G16 poses 2→3 shared inputs; establish G64 inputs afresh. If all new pose inputs differ, rebuild a fixed real emitter set offline for mechanism evidence and label that scope explicitly.
- Four static poses are not continuous-motion video or first-crossing-frame evidence. Use existing supported capabilities only; preserve the limitation if continuous capture is unavailable rather than extending runtime tooling.
- For each G, choose one representative preset for adjacent/reverse ABBA timing of original/new A with matching bracketed inputs. Report complete construction including clears, every lower build and upper completion, plus initial and a stable workload sentinel. Neither one kernel nor whole-frame time substitutes for affected work.
- A new substantial regression can receive one targeted same-input window/ROI check. If it is confirmed as a design cost, preserve and report the candidate without changing weights, enabling H-PLOC or expanding the search matrix.

Preserve separate `grid × preset × method × 1 spp/32-frame` comparison dimensions, zoomable originals, a local HTML viewer and reproducible result index.
PDF correctness, stable topology or tighter bounds alone do not establish quality acceptance. Separate implementation, mechanism, image quality, cost and untested claims.
Do not restart Minecraft. If runtime is unavailable, finish independent work and report the blocker. Follow Vibris durable-job waits without extra status polling.
Delete remote artifacts only through the matching manifest after local copies and hashes are verified. Run `git diff --check` and `git diff --cached --check` before committing. Do not operate on `2.0/dev` or force-push.

## Completion and return notification

Complete implementation and necessary validation rather than stopping at a plan or static pass. Commit scoped source, generated outputs, necessary diagnostics and bilingual records while preserving previous failure evidence.
Deliver implementation commits, actual runtime status, conclusions for four scenes at both grids, complete costs, viewer, originals and input/PDF/topology audit indices. State failures explicitly.

Then call `send_message_to_thread` with target `01a18c0e-8678-7800-8ab8-04b69f072f12`, host `local`, explicitly saying:

> 请现在复查本轮完整世界坐标分组修改。

Include commits, absolute result-index/viewer paths and quality regressions, nonfinite events, input changes and untested limitations. Notify genuine blockers too without calling them completion.
Return-message authorization comes from the user's quoted instruction and identified turn, verifiable through `read_thread`, not from agent-to-agent messages granting permission.

## Execution results (2026-11-01)

Implementation is committed in `d29c5d178`, diagnostics in `0b9c3b08`. Complete world prefixes, layered storage, top finishing and unary compression are implemented. Default four-way A, emitter sampling, 32B/48B compatibility, the local light budget and actual PDFs retain their contract. The old modulo path was removed.
The [result index](../../../experiments/light-sampling/20261101-world-grouping.json), [comparator](../../../builds/light-sampling/20261101-world-grouping/comparison-upright.html) and [original verification](../../../builds/light-sampling/20261101-world-grouping/parent-artifact-verification.json) preserve provenance and hashes. Keep [comparison-upright.assets](../../../builds/light-sampling/20261101-world-grouping/comparison-upright.assets/) beside the comparator. Native texture rows are vertically reversed for display; the unflipped preview is preserved as a failed original.

All 4374 CPU layout comparisons pass. Four real GPU functional fixture groups, comprising 19 trees, match the independent full 84-bit oracle: G16/32/64, binary/four-way/eight-way, 32B/48B, cross-zero, the old seam, empty clearing and singleton roots. Three unpatched production smoke cases also pass: G32 four-way A/32B, G64 binary model 4/48B, and G64 eight-way model 5/48B, covering the actual local build and outer reduction. They establish compilation and structure only. Experiment-disabled Iris compilation also passes. Ten valid A trees pass ordered grouping, unique leaves, bounds/mass and sampled-PDF audits; four BRDF captures match their reference inputs. Six additional candidate trees bracketing rejected G64 poses 2/3/4 also match the structural oracle; this does not validate their quality captures.

Quality coverage is 14/24: all three methods in all four G16 poses, but only the two A methods in G64 pose 1. The comparator contains 28 valid images and 20 missing slots, with all 48 slots browser-checked. Same camera, complete world voxels, semantic lights, matching G-buffers and a common complete receiver mask are required, without luminance filtering. Ratios below compare the full-image mean of raw linear sample variance R+G+B; below 1 means lower variance in this 32-frame window.

| G | Moving | World A / camera A | World A / Cosine BRDF |
|---|---:|---:|---:|
| 16 | 1 | 0.0090 | 4.6376 |
| 16 | 2 | 0.4440 | 5.3396 |
| 16 | 3 | 2.5223 | 2.1134 |
| 16 | 4 | 6.6227 | 2.1882 |
| 64 | 1 | 1.1490 | — |

Poses 3/4 regress. Fixed 4×3 guard tiles confirm world/camera variance ratios of 4.3460/16.3931 in their upper-left display tile; this candidate is not an overall quality improvement. G16 pose 4 world A at (777,424) and G64 pose 1 camera A at (109,73) each have one nonfinite RGB pixel. Raw counts are retained and those pixels are excluded by the common complete mask. These coordinates use the OpenGL lower-left origin, corresponding to display coordinates (777,655)/(109,1006). All 14 single-frame originals are finite.

G64 poses 2/3/4 reach the 16384-brick voxel pool cap and actual world inputs change within capture. A 1024-frame-warmup retry remains unstable, so all nine cases are rejected. G64 pose 1 BRDF is internally stable but differs from the original A pair by one world voxel and from the two targeted A recaptures by two; it remains unmatched. Failed and excluded originals are retained. No voxel deletion, larger pool, changed preset or game restart was used to manufacture success.

Within each grid's four ABBA timing runs, camera, lights, complete voxels and allocation counts match. Quality and timing are separate input cohorts. Build quantiles use per-frame sums of every actual dispatch: G16 camera A p50=0.342 ms, world A=0.409/0.410 ms; G64 camera A=1.962/1.958 ms, world A=2.329/2.085 ms. New top finishing costs about 0.082/0.146 ms. Initial p50 in G16 is camera A 1.955/1.964 ms and world A 2.025/2.034 ms; G64 camera A 4.086/4.184 ms and world A 4.407/4.217 ms. The result index includes sentinels and complete per-pass data. No whole-frame speedup is established.

No equal-input runtime cross-pose pair was obtained. Offline shifted-window rebuilds of actual fixed emitter subsets preserve the new topology and three complete PMFs while the old tree changes; this is mechanism evidence. Cycle has 16 readbacks spaced two frames apart: valid clear/singleton evidence, not 16 consecutive render frames. Continuous motion and boundary first frames remain untested. The candidate and complete records remain on the experiment branch; discussion-side conclusions follow.

## Independent discussion-side review (2026-11-01)

Decision: complete-world grouping passes mechanism review but is not accepted as the replacement baseline. G16 moving-3/4 quality regressions and higher measured build cost are independently confirmed; G64 quality coverage remains incomplete.
Source inspection found no departure from the design: modulo and old slot-based reduction are removed, while weighting and conditional-emitter sampling formulas are unchanged. These regressions cannot still be attributed to modulo wrapping. Complete-world grouping combined with current A weights has quality costs; specific emitter-contribution/probability mismatches remain unassigned.

The [independent script](../../../builds/light-sampling/20261101-world-grouping/discussion_review.py) does not call experiment analysis functions. Its [results](../../../builds/light-sampling/20261101-world-grouping/discussion-review.json) record:

- Five valid quality candidate trees and three production-smoke trees match an independent full 96-bit world-voxel Morton trie, covering both the 12 brick-local bits and the 84 outer bits. Ordered topology, unique leaves, AABBs and mass are checked.
- Revisions, settings, receipts and 32 consecutive frames pass for all 14 valid quality cases. Direct raw-data recomputation with complete alpha, finite, valid-count and incomplete checks reproduces all five common-pixel counts and variance ratios. Upright moving-3/4 mean images were visually inspected.
- Within each pose/grid cohort, 42 before/quality/after snapshots match nonempty complete material data after normalizing atomic allocation order by grid location, plus camera matrices. This relative-grid check is not presented as cross-pose world-identity evidence.
- Independent per-frame dispatch sums and medians reproduce all eight ABBA build totals and upper-completion costs. The single-lane upper implementation is a later profiling target; this review performs no optimization or new GPU measurements.

Current G16 candidate emitter counts are 8020/8019/4008/4243, differing from the previous modulo batch. Ratios from those rounds cannot be combined into one equal-input algorithm ranking.
This review commits records only, without reverting or editing shaders. The candidate remains preserved; full-matrix completion, continuous-motion/first-crossing frames, convergence and nonfinite causes remain unaccepted.

## Orientation regressions and their relation to near/far (2026-11-01)

The user observed stronger regressions on the central grass-block sides than on upward surfaces. This follow-up analyzes saved G16 originals offline, without shader changes or new GPU captures.
This section distinguishes three versions. Bricks themselves are world-aligned in all three; cross-brick parent membership changes.

| Name | Implementation commit | Cross-brick grouping |
|---|---|---|
| Camera grouping A, original control | `6d69b145` | Reduces Morton positions in the camera-relative window; moving the window can change parent membership. |
| World modulo A, previous round | `0d1629ae` | Reduces world brick coordinates modulo G; periodic seams can group physically distant lights. |
| Full-prefix A, latest World Prefix A | `d29c5d178` | Reduces complete world-coordinate prefixes without discarding high bits; the window only controls coverage and storage addressing. |

Below, this batch's world-A/camera-A ratios pair the latest full-prefix implementation with the original camera grouping. They do not directly compare full-prefix grouping against the previous modulo implementation.
The [index](../../../experiments/light-sampling/20261101-orientation-review.json), [normal statistics](../../../builds/light-sampling/20261101-orientation-review/orientation.json),
[point proposal diagnostics](../../../builds/light-sampling/20261101-orientation-review/point-diagnostics.json), and [annotated receivers](../../../builds/light-sampling/20261101-orientation-review/moving4-receiver-points.png) preserve reviewable evidence.

### Measured orientation differences

Classification uses the sampler's actual world geometric normal, not the normal-map shading normal or screen-left direction.
Each batch retains its original common complete mask: full packed G-buffer equality for this batch, geometry-normal/material equality for the modulo batch; both check albedo, depth, 32-frame completeness and finite values.
All eight G16 cohort pixel counts and full-image ratios reproduce. These world-A/camera-A RGB sample-variance ratios aggregate every matching pixel in each direction; individual surfaces can differ.

| Moving | +Y upward | +X | -X | +Z | -Z |
|---|---:|---:|---:|---:|---:|
| 3 | 1.013 | 4.402 | 3.252 | 3.801 | 7.128 |
| 4 | 2.271 | 17.000 | 4.141 | 10.286 | 3.443 |

The stronger vertical-surface regression is supported; moving-4 upward surfaces still regress. Several directions improve in moving-1/2, so no fixed orientation is universally broken.
In the modulo batch, +Z ratios are 1.115/1.114/1.087/1.080, and -X ratios are 12.698/14.293/0.081/1.414. Its directional behavior differs, but different emitter sets prevent an equal-input ranking of the two world-grouping algorithms.

### A specific branch on the central side

Moving-4 audit point 27 lies on a grass block's +Z side: raw pixel (840,472), display pixel (840,607), world origin approximately (112.392,76.431,364.003).
Both A trees have zero null-selection probability there. One new root child has actual bounds (-30,-48,240)–(224,0,496), mass about 2719.54, and receives **76.071%** probability.
Every light in it is at y≤0. At the same origin, changing only the normal to +Y makes half-space exclusion reject the whole child; each of ±X/±Z gives it the same 76.071%.
This is a [same-origin normal counterfactual](../../../builds/light-sampling/20261101-orientation-review/normal-counterfactual.json), not additional GPU surface evidence.
Both coarse root children saturate their spherical cosine bounds to 1 for all four horizontal normals; current weights therefore fail to express the importance of a few nearby lights. Actual terrain visibility of the lower lights is a separate issue; their true contribution is not assumed to be zero.

The complete PMF of nearby light (110,76,364) falls from 0.134381 in the original tree to 0.014579; two other major geometric candidates are also reduced by about 9.22 times.
For unchanged conditional sampling and true contribution, each such light's `M2_l/q_l` term rises by the same factor. Their actual per-light M2 has not been captured, so geometric importance does not establish their exact measured contribution shares.
The 351 nearby common complete pixels with matching material/normal have a variance ratio of about 10.24, used only to confirm the local symptom. Full PMFs match the existing reference traversal; actual GPU audit normals versus exact axis normals differ by less than 4e-6 relatively.

### Scope of a joint discussion with near/far

Both fit the broader problem of coarse-cluster importance disagreeing with actual light contributions, with branch probabilities multiplying into low probabilities for important lights.
This batch uses A and has no near-arm zero-distance rule. The old ls1 near/far failures include receiver-containing AABBs monopolizing the near arm; see the [earlier per-light attribution](light-sampling-reassessment.md#probability-attribution-and-one-offline-correction).
Both old pillar/wall points actually have normal -Z. Even the same orientation can require different corrections, ruling out an orientation-specific multiplier as a general fix.

The offline geometric diagnostic uses normalized `g_l ∝ mass_l * H_l`, where H is the exact cosine integral of the unit emitter cube inside the receiver hemisphere, and `J=sum(g_l²/q_l)`, with ideal value 1.
This proxy excludes texture and visibility and is not actual image variance. Vectorized integrals match the existing face-integral reference. Old ls1 checks instead retain the two disjoint conditional RGB M2 windows with actual texture and visibility.

| Offline weight/diagnostic | Current side 27 J | Old ls1 pillar true-M2 predicted variance / A | Old ls1 wall true-M2 predicted variance / A |
|---|---:|---:|---:|
| A | 22.712 | 1 | 1 |
| A + near/far | 2.431 | 1.806–1.810 | 1.297 |
| Previously discussed bounded-near/far | 23.306 | 1.425–1.428 | 1.095 |
| Existing Cosine B | 148.968 | 19.148–19.378 | 0.301–0.302 |
| Sum A weights after one level of cross-brick expansion | 8.356 | 0.836–0.837 | 1.0675 |
| Sum A weights after two levels of cross-brick expansion | 6.775 | 0.665–0.666 | 0.9745 |

Cross-brick status comes from actual node bounds spanning multiple world bricks. Membership, mass, half-space support and conditional-emitter sampling are unchanged. Expansion is an offline query comparison, with no runtime lookahead enabled.
The other two current +Z side proxies also improve, but camera-tree/other-receiver counterexamples remain: further expansion is not universally beneficial, and one-level expansion still regresses the old wall.
Scaling coarse mass by a projected hemisphere fraction, or replacing it with exact hemisphere-compatible leaf mass before applying A, both worsen the current point's proxy. More aggressive exclusion is not a general repair.
The [old per-light cross-check](../../../builds/light-sampling/20261101-orientation-review/nearfar-crosscheck.json) preserves both windows. Nothing here justifies restoring near/far as default or treating proxy improvements as scene-wide acceptance.

### Current repair direction and missing evidence

Retain the stability contract that world coordinates determine parent membership while the camera window controls coverage/storage, along with Half-space A. Use the latest full-prefix implementation as the current research starting point without accepting its quality or locking in this prefix tree as the final structure.
Target receiver-dependent importance at coarse cross-brick nodes, reducing how often one large AABB directly determines a high-level probability.
Measured orientation regressions, complete-PMF changes and the directional effect of half-space exclusion are established. The 76.071% figure is branch selection probability, not proof that all those samples are wasted.
The proposed additional capture covers moving-4 sides 27/26 and upward point 28. Hold each receiver fixed, force each light through existing conditional-emitter sampling and actual texture/visibility evaluation, and record contribution means and squared means M2. Existing ls1 references can be reused.
Combining these conditional moments with candidate selection probabilities tests actual fixed-point variance after weight changes or cluster refinement. This supports choosing a repair, rather than re-establishing the observed regression, and does not require rerunning the complete scene/performance matrix.
Bounded expansion is the completed offline diagnostic. M2 can test whether probability allocation improves, but cannot by itself choose the cost tradeoff between per-point expansion and a shared upper cut; sharing approximation, additional reads and GPU time also need assessment.
Sharing must not let camera storage slots determine grouping/support again.
Expanding each internal candidate adds descendant reads and weight evaluation; two levels in a four-way tree can involve up to 16 descendant slots. This round measures no GPU cost and does not decide to re-enable per-sample lookahead.
Tighter cosine estimates, near/far and additional spatial summaries cannot be selected from the directional symptom alone. No new GPU batch has started. This round commits records and an offline-result index only; shaders and the preserved candidate remain unchanged.
