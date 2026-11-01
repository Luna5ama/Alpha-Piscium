# Light Sampling World-Space Grouping Design and Experiment Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-world-grouping-prompt.md)

## Current task: moving-4 per-light M2 and five-point repair screening (2026-11-01)

### Authorization, objective and workspace

The human user in discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12`, turn `01a196cb-8b2c-7843-8c28-6b7a9a9036e3`, explicitly requested:

> Continue along this route. Prepare the prompt and have the experiment agent collect per-light M2 for moving-4's two sides and one upward surface, screen repairs together with the two earlier near/far failures, and notify you for review when finished.

Experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` now executes this task; the discussion chat performs independent review. The original human request also authorizes completion/substantive-blocker callbacks, verifiable with `read_thread`; callback authority does not come merely from an agent message.
Use `I:/code/mcshaders/Alpha-Piscium-4` (wt4), branch `2.0/light-sampling`. Historical cwd metadata does not override this routing. At preparation, HEAD is `ed4f5d192` and the checkout is clean; inspect subsequent changes before execution rather than resetting to an older commit.

Deliver actual conditional RGB means/M2 for the three specified receivers, a candidate table covering them and the two old ls1 points, per-light/branch attribution, and a repair recommendation. Follow the [selected main-design route](light-sampling-design.md#current-selection-and-backlog-2026-11-01): start from the latest full-world-prefix A and correct coarse cross-brick importance.
This is a fixed-point capture and offline screening task. Necessary fixture, mode-3 diagnostic, capture/analysis tooling and generated-binding changes are authorized. Do not change production sampling defaults, implement shared cuts, reopen H-PLOC/visibility, launch a whole-scene quality/performance matrix, or promote a screened candidate directly to default.
This section retains the task contract. Three-point capture, joint screening and independent review are complete: the conditional references and offline screening are accepted, but no complete repair was found; see the results below. Do not rerun the G16/G64 four-scene matrix.

### Three receivers and the previous reference

Use `light-sampling-failure-moving-4`, G=16, default four-way branching, 32B/Half-space A and existing cube conditional sampling. Retain native 1920×1080 to reproduce the specified pixels/surfaces, and obtain other effective settings from the successful saved capture.
Locations come from the [orientation diagnosis](../../../experiments/light-sampling/20261101-orientation-review.json), [complete receiver identities](../../../builds/light-sampling/20261101-orientation-review/orientation.json), and [annotated image](../../../builds/light-sampling/20261101-orientation-review/moving4-receiver-points.png). The source capture is [moving-4/world-a-recapture](../../../builds/light-sampling/20261101-world-grouping/captures/g16/moving-4/world-a-recapture/).

| ID | Raw/display pixel | World sampling origin, already offset | Geometric normal | Material / G-buffer RGB bytes |
|---|---|---|---|---|
| side-26 | (600,472) / (600,607) | (109.156212,76.138733,362.002991) | +Z | 4651 / (121,85,58) |
| side-27 | (840,472) / (840,607) | (112.391624,76.430634,364.002991) | +Z | 4651 / (89,61,41) |
| top-28 | (1080,472) / (1080,607) | (115.779617,76.003006,365.398621) | +Y | 4651 / (57,113,36) |

Use actual `ls_loadReceiver` origins, geometric normals, material and albedo. Do not add another 0.003 offset, treat display coordinates as raw texture coordinates, or substitute shading normals. Follow the existing diagnostic checks: position error <0.005, normal dot product >0.999, and matching material/reflectance identity. Do not silently substitute surfaces.
Reuse the old ls1 [3921-light/two-point fixture](../../../experiments/light-sampling/20261031-ls1-point-fixture.json), [two native conditional-moment windows](../../../builds/light-sampling/20261031-nearfar-iteration/candidate-runtime-evidence/ls1-point-windows/), and [recomputation script](../../../scripts/light_sampling_nearfar_reference.py). Its raw pixels are (1400,550) and (120,700), both -Z, corresponding to the established pillar/wall failures.
Verify originals/fixture identity and compatible conditional sampling, actual evaluation and color-space semantics before recomputing. Do not overwrite [light_sampling_reference.bin](../../../shaders/textures/light_sampling_reference.bin), which remains an actual dependency of the old analysis. Retain the old points' scene-specific conditional moments, first reproduce established A/near-far results on the old tree, then build a full-prefix tree offline from the same lights to evaluate the selected route. Conditional moments themselves do not depend on the selection tree. Map lights between topologies by world position/material and verified mass, not node-array indices. Never combine per-light arrays, PMFs or M2 from different scenes.

### Implementation and capture boundaries

Reuse [LightSamplingInitial mode 3](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl), [capture and bracketed input checks](../../../scripts/light_sampling_weight_capture.py), [world-identity reconstruction](../../../scripts/light_sampling_moving_analysis.py), and the [old conditional-moment reference](../../../scripts/light_sampling_nearfar_reference.py). Read applicable shader/scripts AGENTS; apply relevant opengl-glsl guidance if editing GLSL.
Mode 3 currently supports two points, two header texels per point followed by light records; the [maintained texture binding](../../../scripts/shaders.properties) still specifies 3925 texels. Prefer reusing this shader path with pairs (side-26, side-27) and (side-27, top-28), repeating side-27 to check consistency between groups. Use independent new fixture files and frozen source bindings, preserving each group's actual dimensions/source identity and the old ls1 file. A minimal three-point diagnostic adaptation is allowed if simpler, without new product settings, generic frameworks or compatibility-format branches.
Build fixtures only from root-reachable leaves of the current stable tree, preserving complete position/material/mass identity. Do not scan retained storage and count stale copies as lights. The historical count 4243 must not become a hardcoded invariant for new captures. Derive dimensions, light count and conditional samples per point/light/frame from actual metadata/fixture/audit instead of copying old constants 264/3921; unused tail pixels are not conditional samples.

First verify the actual shader pack, frozen revision, preset, effective settings, successful receipts, camera/receiver and complete world emitter/nonemissive-voxel identities. Reuse the established two-point path directly, but accept the batch only after a real first-group result confirms mode-3 receiver validity, binding dimensions, sample mapping and moment identity.
For each group collect two disjoint, consecutive 32-frame windows, each in 8×4-frame batches. Preserve mean, variance, valid count, nonfinite, incomplete/valid batches, frame numbers, raw audit, tree, G-buffer, camera and before/after complete world voxel snapshots. Retain sufficient warmup and check semantic inputs across both windows and both groups. Atomic allocation/buffer-address changes are not themselves world-input changes.
Prefer reproducing the historical moving-4 input. If lights/occluders differ but the three surfaces still match, establish a fresh stable cohort and recompute all of its trees/candidate PMFs. Report historical differences explicitly; do not reuse the old 76% or 13.44%→1.46% as new measurements. Preserve evidence and report back if input keeps changing within windows, a target surface fails to match, or the stated mismatch cannot be reproduced.
If the two groups cannot share semantic inputs, compute each group's M2/PMFs against its actual input and state that a common cohort is missing. Do not silently combine them into one three-point comparison. Targeted retries of the affected group are allowed, without expanding into a search for new scenes.

Mode 3 fixes `lightPdf=1` for each light while retaining actual texture, cube conditional sampling, hemisphere and voxel visibility. Keep linear-sRGB RGB output and alpha=`dot(RGB,RGB)`. Zero-contribution outcomes are valid samples, not candidates for filtering and renormalization.
Do not apply the ordinary image's alpha==1 receiver mask. Validity here comes from audit receiver-valid and separate completeness/nonfinite counters.
Report complete/missing/nonfinite sample counts per receiver/light and use the same valid conditional moments for every candidate. A light with no valid samples does not have zero M2; report support failures, trace exhaustion and input rejection explicitly.
First verify `E[||Z||²] = ||E[Z]||² + 31/32 * sum(sampleVarianceRGB)` on each conditional-sample pixel's 32-frame statistics, then aggregate by receiver/light. Infer each window separately and report their difference without calling it a rigorous confidence interval or convergence proof. Preserve per-light data for independent discussion-side recomputation.

### Five-point candidate screening

Keep the light set/conditional sampler fixed for each candidate and enumerate complete actual leaf probabilities `q_l` plus null probability. Do not renormalize `q_l` conditional on successful light selection.
Primary comparisons use the full-prefix tree for the new moving-4 input. Include original camera grouping A under identical input as an additional control, using matching saved originals or an already-checked reconstruction method, and label GPU trees versus offline reconstructions. Historical mismatched-input PMFs are not valid controls.
For the two old ls1 points, also report candidates on a same-light full-prefix tree while retaining the known old-tree A/near-far failure reproduction. Reuse [world-layout reconstruction](../../../scripts/light_sampling_world_layout.py) and its GPU comparison evidence; no old-scene recapture is required. Unverified reconstruction is a gap, and improvement only on the old tree does not establish success on the new route.
Use this fixed small set, with formulas from the reviewed [point_diagnostics.py](../../../builds/light-sampling/20261101-orientation-review/point_diagnostics.py) and [nearfar_crosscheck.py](../../../builds/light-sampling/20261101-orientation-review/nearfar_crosscheck.py). These are research references, not a requirement to turn them into a product framework:

| Candidate | Purpose and boundary |
|---|---|
| Half-space A | Baseline within each scene |
| Half-space A + near/far, model 6 | Historical failure control, not a presumed fix for the new problem |
| bounded-near/far | Near denominator `max(dmin²,halfDiagonal²,0.25)`; unchanged far arm, half mixture and half-space |
| Cosine B, model 2 | Directional counterexample for a tighter cosine estimate |
| One-level cross-brick expansion | Only for internal candidates whose actual bounds span world bricks, replace their weight with the sum of next-level A weights; unchanged leaf half-space criterion, mass and membership |
| Two-level cross-brick expansion | Continue up to two levels under the same rule and existing offline definition |

Do not scan mixture coefficients, hardcode corrections by orientation/these five points, or present offline oracles such as exact hemisphere-compatible mass as cheap production methods. Existing other ablations may be cited, but do not expand this candidate matrix.
For the three new points and two old points, each with two reference windows, compute `sum(M2_l/q_l) - ||sum(E[Z_l])||²`, clearly label tree versions, and report ratios to each scene's full-prefix A and original camera A. List old-tree candidate recomputations separately rather than labeling them full-prefix results. Do not compare absolute variances across different scenes.
Any light with measured positive M2 and `q_l=0` is a support failure, not a term to discard before calculating finite variance. Positive-contribution hit rate alone is insufficient. Rank major penalties by `M2_l/q_l`, report PMFs, conditional moments, relevant parent splits and probability changes, and check whether the actual contribution/second-moment share of the y≤0 root cluster supports the earlier geometric explanation.
If aggregating `sqrt(M2_l)` to localize branch mismatch, identify it as the finite-reference optimal leaf-probability diagnostic for this fixed conditional sampler. Separate irreducible conditional variance from light-selection mismatch; geometric proxy J is not actual M2.
The outcome is an evidence-backed shortlist or a finding that no candidate accommodates all five points. Report both-window counterexamples, upward-surface protection, changes on the old ls1 points, and added query work; gains comparable to between-window differences remain uncertain. No winner is required for completion.
Static counts may describe expansion's added reads/computation; do not run a GPU performance matrix in this task. Better fixed-point variance does not establish full-image, continuous-motion or final-cost acceptance, nor can M2 alone choose a shared-cut implementation.

### Artifacts, execution ownership and callback

The experiment chat is the sole coordinator for wt4 edits and live Minecraft capture during this task. Retain existing executor/model arrangements rather than opening competing operators. After dispatch, the discussion chat does not concurrently edit shaders or control the runtime.
Follow [AGENTS](../../../AGENTS.md) and [documentation rules](../../AGENTS.md): PowerShell 7 without profile, lowest priority for compilation/offline long tasks that are not timed benchmarks, and latest available GraalVM. Report global Gradle-directory failures without substituting a temporary home. Regenerate bindings from maintained sources and perform necessary Target Iris compilation/runtime checks for actual diagnostic changes.
Do not restart Minecraft/the launcher. Exhaust safe non-restart recovery, then report a blocker if unusable. Use Vibris durable jobs/waits without extra polling just for status reports. Before returning control, verify no capture job remains active, leave mode 3, and report the actual retained pack/revision/preset/configuration and unsaved changes.
Put new originals/full logs in an independent ignored directory, for example `builds/light-sampling/20261101-moving4-point-m2/`, without overwriting historical batches. Preserve native data, required identities, manifests, derived per-light arrays and reproducible analysis scripts. Delete remote artifacts only by their matching manifest after local copies/hashes are verified.
Track a compact experiment index, necessary fixture/tool changes and bilingual result summary linked to this task. Do not place large originals or per-file hash inventories in documentation. Check language parity/links, `git diff --check` and `git diff --cached --check` before committing. Do not modify `2.0/dev` or force-push.

After completing capture, five-point screening, self-checks and commits, call `send_message_to_thread`, host=`local`, target=`01a18c0e-8678-7800-8ab8-04b69f072f12`, explicitly requesting:

> Please now review the moving-4 three-point per-light M2 and joint repair screening with the two old near/far points.

Include implementation/diagnostic commits, absolute task/result-index paths, originals/per-light arrays, actual input consistency, the two-window candidate table, major counterexamples/support failures/nonfinite events/unverified scope, and final runtime state. Report substantive blockers too, without claiming completion. The discussion chat makes the independent review decision; do not presume acceptance or stop after a plan/static checks alone.

The completed full-world-prefix implementation task and historical diagnosis follow for provenance and limitations.

## Three-point M2 and joint screening results (2026-11-01)

The [result index](../../../experiments/light-sampling/20261101-moving4-point-m2.json), [complete dual-baseline table](../../../builds/light-sampling/20261101-moving4-point-m2/screening-table.csv) and [per-light/branch report](../../../builds/light-sampling/20261101-moving4-point-m2/report.md) preserve this measurement. Source/binding commits are `1b9b62c5` and `f06dcbe3`; restoration is `c091fc40`; actual-receiver and reconstruction checks are in `dcfe59ae` and `6d7c0c3d`. Production defaults are unchanged. The legacy reference texture and 3925 binding are restored.

Two groups each contain two nonoverlapping native 32-frame windows, with 8×4 batches per window: four new windows and eight receiver-window observations, including the repeated side-27. Both groups share 4243 root-reachable emitters, camera, surfaces and complete world input; all three bracketing snapshots match. Each light has 244 draws/frame and 7808/window; 3016 unused tail pixels are excluded. Zero outcomes are retained, with no missing or nonfinite RGB samples. Per-pixel second-moment identity error stays below 2.97e-7. Two historical ls1 windows supply the two old points, yielding five primary points with two windows each and two extra side-27 observations. The old scene was not recaptured.

Independent nonemissive fingerprints agree: after removing the 4243 reachable emitter voxels, 480363 nonempty voxels remain, SHA `bd558540e6e1b96092e50164307587ef22fac4d3eb0cb01ceb6a8e8d02429502`. Ten material values differ from historical moving-4 while emitters match. This table uses newly measured M2 rather than the previous geometric proxy. The actual legacy sampling revision is `fa8ac641`; conditional sampling, visibility and linear-sRGB norm-squared semantics match current source. Twelve old-tree A/near-far/bounded results reproduce with maximum difference 3.47e-18.

Ratios below use each scene's **complete-prefix A=1**; each cell gives window 1/window 2. Cosine B is a tree weight, not Cosine BRDF. Complete unconditional leaf probabilities and null probabilities are retained, without successful-selection or positive-contribution renormalization.

| Point | A + near/far | Bounded near/far | Cosine B | One cross-brick level | Two cross-brick levels |
|---|---:|---:|---:|---:|---:|
| side-26 | 0.1128 / 0.1129 | 0.9277 / 0.9277 | 4.3869 / 4.3868 | 0.4428 / 0.4428 | 0.3253 / 0.3253 |
| side-27 | 0.0832 / 0.0832 | 1.0057 / 1.0061 | 7.7749 / 7.7844 | 0.3537 / 0.3536 | 0.2810 / 0.2810 |
| top-28 | 0.3940 / 0.3946 | 1.2754 / 1.2765 | 13.1006 / 13.0676 | 0.5316 / 0.5325 | 0.1837 / 0.1853 |
| ls1-pillar | 1.4891 / 1.4887 | 1.2747 / 1.2749 | 0.4571 / 0.4580 | 0.5667 / 0.5669 | 0.5198 / 0.5201 |
| ls1-wall | 0.7556 / 0.7560 | 0.8715 / 0.8710 | 0.3600 / 0.3602 | 0.5916 / 0.5919 | 0.5306 / 0.5310 |

**The research shortlist is one/two-level cross-brick expansion; no complete repair was found.** Both improve prefix A at all five points in both windows. However, two-level variance at the new sides remains about 2.40×/3.05× original camera A; one-level remains about 3.27×/3.84×. At the upward point, one-level is still about 1.35× camera A, while two-level reaches about 0.466×. No tested repair satisfies both baselines across all five points. Near/far hurts the old pillar; bounded near/far hurts the pillar and upward point; Cosine B strongly hurts the new three points. Near-unit changes comparable to window differences are uncertain, not selected as gains.

Actual per-light M2 narrows the geometry explanation. At both sides, the y≤0 root child receives about 76.07% probability with zero observed RGB mean/M2 share in both reference windows. This is finite-reference evidence, not proof of exactly zero transport. Side-27 one/two-level expansion still assigns about 73.59%/77.03% to this child, so neither repairs that allocation. Gains mostly arise inside the positive-height branch. A's largest split penalty occurs at bounds (0,0,320)–(127,79,384), with about 9.81% path probability. The leading penalty light is (110,76,364,material 10242), A leaf probability about 1.458%. Full paths/probability changes are retained; sqrt(M2) aggregation is only a finite-reference optimum diagnostic under fixed conditional sampling.

Static work distinguishes nonempty logical records from native child slots. Retain the prior counts as nonempty logical records: about 46.5/sample at the new sides for one level and 143.5–143.7 for two levels, and 45.0/157.8 upward. Including zero-mass slots from captured GPU childBase/childCount, additional one/two-level slot checks are 65.07/207.23 for side-26, 65.14/207.14 for side-27, and 58.50/207.59 for top-28. Nonempty A geometry counts exclude empty-slot mass==0 tests and early returns. The old ls1 prefix is only an offline logical rebuild, with no physical-slot cost claim. These naive counts give no credit for early parent-bound rejection, reuse or shared cuts, and are not memory transactions or GPU timing; PMFs, M2 and quality results are unchanged. This remains a point-quality shortlist. Production implementation, full images, continuous movement and GPU cost are unvalidated; shared-cut representation is undecided and lookahead defaults remain disabled.

The [conditional arrays](../../../builds/light-sampling/20261101-moving4-point-m2/moments/), [complete PMFs](../../../builds/light-sampling/20261101-moving4-point-m2/screening/), [branch identity checks](../../../builds/light-sampling/20261101-moving4-point-m2/branch-decomposition-validation.json) and [legacy recomputation](../../../builds/light-sampling/20261101-moving4-point-m2/legacy-native-validation.json) support independent recalculation. Use the [moment tool](../../../scripts/light_sampling_point_moments.py) and [screening tool](../../../scripts/light_sampling_point_screening.py). Origins/normals come directly from actual mode-3 audit; fixture expectations only accept identity and no offset is added twice. GPU trees and replays match per-node bounds/mass/ordered children; historical full-prefix trees are explicitly offline same-emitter rebuilds.

Runtime has exited mode 3, retaining frozen `c091fc40`, moving-4 preset, G16/four-way/model 1/cube, sampler 2, with no live capture jobs. Actual active source UUID is `d4bc57f6-3bf3-4e1e-8c68-4d143e4607ce`; the interface does not expose a human-readable pack name, so none is guessed. See the [runtime handoff](../../../builds/light-sampling/20261101-moving4-point-m2/runtime-exit/runtime-final.json) for actual settings/state.

### Independent review (2026-11-01)

Accept the native conditional references and offline screening conclusions, without accepting any candidate as a complete repair or default replacement. Review covers `a2405c170` and cost correction `b8dcd2dd9`. Two-level expansion still leaves the side points at about 2.40×/3.05× camera A, with the main probability waste in the low cluster intact; near/far raises the old pillar's variance on its same-light complete-prefix tree to about 1.49×. Improvement at the three new points therefore does not justify selecting near/far, and improvement over prefix A does not establish quality acceptance for the movement repair.

The [independent recomputation](../../../builds/light-sampling/20261101-moving4-point-m2/discussion_review.py) imports none of the submitted capture, moment, layout-reconstruction or screening implementations. It aggregates conditional moments from raw means/sample variances/completeness data, independently decodes native trees, constructs complete world prefixes and traverses full PMFs. Its [results](../../../builds/light-sampling/20261101-moving4-point-m2/discussion-review.json) match all 12 point-window rows and 168 candidate/tree/point-window combinations. Maximum relative PMF difference is about 1.0e-15; maximum relative moment-identity error is below 2.97e-7 for the new windows and 4.11e-7 including the historical windows. No observed contribution in finite windows still does not prove exactly zero transport.

The [input review](../../../builds/light-sampling/20261101-moving4-point-m2/discussion_inputs.py) and [results](../../../builds/light-sampling/20261101-moving4-point-m2/discussion-input-review.json) independently verify sizes/hashes for 164 native files, actual frozen sources/effective settings, receivers and before/between/after world input. Both new groups share complete nonemissive input; the ten nonemissive material differences from historical moving-4 are independently reproduced, without applying historical M2 to new input. The eight exit-state originals are preserved and verified under [runtime-exit/raw](../../../builds/light-sampling/20261101-moving4-point-m2/runtime-exit/raw/). Exit-state claims refer to saved cleanup records, not a fresh live-runtime check during review.

Review found that the original static read counts included only nonempty logical children. The experiment chat corrected native empty slots in `b8dcd2dd9`. The [independent cost script](../../../builds/light-sampling/20261101-moving4-point-m2/discussion_costs.py) and [results](../../../builds/light-sampling/20261101-moving4-point-m2/discussion-cost-review.json) confirm the physical-slot figures above and unchanged hashes for 42 quality arrays/tables/manifests. The reviewer requested the correction and rechecked the revision; shader behavior, PMFs and M2 were unchanged. Review launched no GPU captures and did not measure implemented expansion performance, full-image quality or continuous movement, or select a shared-cut implementation.

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

The selected route is to continue correcting coarse-cluster importance on the latest complete-world-prefix A implementation; see the [main design decision](light-sampling-design.md#current-selection-and-backlog-2026-11-01). Modulo grouping is no longer an active repair route. Additional per-light conditional moments select a correction within this route.
Retain the stability contract that world coordinates determine parent membership while the camera window controls coverage/storage, along with Half-space A. Use the latest full-prefix implementation as the current research starting point without accepting its quality or locking in this prefix tree as the final structure.
Target receiver-dependent importance at coarse cross-brick nodes, reducing how often one large AABB directly determines a high-level probability.
Measured orientation regressions, complete-PMF changes and the directional effect of half-space exclusion are established. The 76.071% figure is branch selection probability, not proof that all those samples are wasted.
The proposed additional capture covers moving-4 sides 27/26 and upward point 28. Hold each receiver fixed, force each light through existing conditional-emitter sampling and actual texture/visibility evaluation, and record contribution means and squared means M2. Existing ls1 references can be reused.
Combining these conditional moments with candidate selection probabilities tests actual fixed-point variance after weight changes or cluster refinement. This supports choosing a repair, rather than re-establishing the observed regression, and does not require rerunning the complete scene/performance matrix.
Bounded expansion is the completed offline diagnostic. M2 can test whether probability allocation improves, but cannot by itself choose the cost tradeoff between per-point expansion and a shared upper cut; sharing approximation, additional reads and GPU time also need assessment.
Sharing must not let camera storage slots determine grouping/support again.
Expanding each internal candidate adds descendant reads and weight evaluation; two levels in a four-way tree can involve up to 16 descendant slots. This round measures no GPU cost and does not decide to re-enable per-sample lookahead.
Tighter cosine estimates, near/far and additional spatial summaries cannot be selected from the directional symptom alone. The diagnosis above originally committed records/offline results without shader changes. The subsequently authorized per-light capture follows the current task at the top of this page.
