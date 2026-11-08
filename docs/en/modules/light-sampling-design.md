# Light Sampling Design

Language: English | [简体中文](../../sc/modules/light-sampling-design.md)

This records Alpha Piscium light-sampling decisions, candidates awaiting validation, and experimental evidence.
The initial design and CPU experiments of 2026-10-29 remain below. Their implementation status and proposed next
steps describe that earlier work; the current decisions below take precedence where they differ.

## Discussion and experiment responsibilities (2026-10-30)

- The current discussion maintains this document and its Chinese counterpart, recording confirmed decisions,
  rationale, hypotheses, and the questions subsequent experiments should answer.
- Experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` owns candidate implementation, runtime captures, timings,
  and image comparisons. Its code stays on `2.0/light-sampling`. A discussed candidate does not automatically
  authorize starting an experiment.
- Experimental results identify the source commit, configuration, scenes, and limitations before informing design
  updates. Record selection, implementation, and validated benefit separately.

Design, experiment sources, and tracked records are consolidated on `2.0/light-sampling`. The migration
preserves experiment commit `5c022ba1` and the later design updates from `2.0/dev`. The runtime entry points
are [LightSampling](../../../shaders/techniques/debug/LightSampling.glsl),
[LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl), and
[option registration](../../../scripts/options.main.kts).

## Current selection and backlog (2026-11-01)

| Item | Confirmed choice |
|------|------------------|
| Conditional emitter sampling | Retain projected-solid-angle cube sampling without repeating its adoption comparison |
| Node weights | Half-space A (32B) is implemented as default; near/far and A + near/far remain explicit experiments with unresolved local regressions |
| Spatial grouping | Select the latest complete world prefixes as the starting point for further motion-stability repair; current quality remains unaccepted, as detailed below. Gilbert was rejected and its implementation removed |
| Branching | Default to four-way, retaining binary and eight-way alternatives; all run without geometric lookahead |
| Lookahead | Return to the backlog; revisit only in an explicit quality versus traversal-cost comparison |
| Visibility guidance | Exclude from the current runtime baseline; defer further discussion at the user's request |
| H-PLOC | On hold at the user's request; retain the global cross-brick scope if reopened |

Continue motion repair with **complete-world-prefix A (World Prefix A, `d29c5d178`) plus importance correction for coarse cross-brick clusters**.
Complete world coordinates determine parents. For the same emitter set, moving the camera window changes coverage/addresses without changing parent membership or introducing modulo seams.
Stop iterating on world modulo A as the repair route; retain original camera grouping A only as a quality/cost control. This selection does not accept the current prefix candidate's quality or continuous-motion behavior.
Retain Half-space A, default four-way branching and existing conditional-emitter sampling. First address probability mismatch from directly weighting coarse cross-brick clusters, without orientation-specific multipliers or a default switch to near/far.
Additional conditional moments for moving-4's two sides and one upward receiver select a repair within this chosen route. Reuse the two old ls1 points as regression guards; further evidence is not a prerequisite for choosing the motion-repair route itself.
Bounded expansion remains a diagnostic/candidate comparison; shared-cut representation and cost are undecided. If weight correction cannot reconcile quality and query cost, reopen the upper representation while retaining stable world membership.
See [world-grouping diagnosis and repair direction](light-sampling-world-grouping-prompt.md#current-repair-direction-and-missing-evidence) for evidence. The user has subsequently authorized the experiment chat to capture moving-4's three-point per-light M2, reuse the old ls1 points for screening, and notify the discussion chat for review. The [current capture prompt](light-sampling-world-grouping-prompt.md#current-task-moving-4-per-light-m2-and-five-point-repair-screening-2026-11-01) defines scope, excluding a full-image/performance matrix or default changes.

The [three-point M2/five-point screening](../../../experiments/light-sampling/20261101-moving4-point-m2.json) is complete and has passed [independent review of the conditional references and offline screening](light-sampling-world-grouping-prompt.md#independent-review-2026-11-01). One/two-level cross-brick expansion improves prefix A at all five points in both windows, but the two sides remain worse than original camera A; neither reduces the y≤0 allocation with no observed contribution. Retain a research shortlist without default changes. Including native empty slots, additional one/two-level child-slot checks per sample are 65.07/207.23 and 65.14/207.14 at the sides, and 58.50/207.59 upward; the prior nonempty logical-record counts remain separately reported. Empty-slot mass tests and early returns are excluded from nonempty A geometry arithmetic. The offline ls1 prefix rebuild makes no physical-layout cost claim. These static slot counts exclude early rejection/reuse and are not memory transactions or GPU timing; GPU cost and shared-cut representation remain undecided.

Future light-sampling comparisons always include independent **Cosine BRDF** (cosine-weighted BRDF-only) as a basic control under the same scene, direct-only settings and sample budget. This is distinct from the Cosine B tree weight.
Display names are Half-space A, Near/Far Mix, Half-space A + Near/Far Mix, Cosine BRDF and Original; original capture IDs and revisions stay unchanged.
For this temporary juxtaposition, previous Cosine BRDF, Original and available ls3 captures are reused. At the user's subsequent request, one supplementary ls3 combined-method capture supplies its missing 1-spp and 32-frame-mean images. Inputs can differ across batches; this does not create a new controlled quality or performance ranking.
The [complete temporary viewer](../../../builds/light-sampling/20261031-temporary-baselines-complete/comparison.html) includes five scenes and 50 images with no missing combinations. Its [supplement record](../../../experiments/light-sampling/20261031-ls3-combination.json) preserves provenance and verification hashes. The [earlier 48-image record](../../../experiments/light-sampling/20261031-temporary-baselines.json) remains unchanged.

These choices apply to the experimental path; the direct-lighting experiment itself remains disabled by default.
The option generator now implements half-space A as default. Target Iris reports model 1 with default origin,
four-way branching, cube sampling enabled, and 32B nodes. The [default validation record](../../../experiments/light-sampling/20261031-halfspace-default.json)
retains source snapshots, artifact hashes and a 63-point PDF audit. The disabled experiment configuration separately passed compilation;
this does not retroactively change the methods used in earlier captures. Earlier default settings and lookahead-option removal are in `f9dc71bf`; the five-scene wide-tree experiment record is in
`3e7f66b9`. Four-way branching is the selected tradeoff, not the lowest-variance result: binary two-level lookahead
had lower variance in both reported luminance regions in that round. A 32-frame variance window is not error
acceptance against a high-sample reference.

The 2026-10-31 [original-evidence review](light-sampling-reassessment.md#review-of-original-scene-evidence-2026-10-31)
confirmed out-of-bounds and competing writes in the fixed 64-slot audit of the old 1080p quality captures.
That round's PDF/quality acceptance requires corrected captures; its timing captures had audit disabled.
That review did not change the defaults at the time. Validly addressed old 720p captures separately exposed aggregate
distortion and near-field back-hemisphere weighting problems.

## Follow-up repair discussion (2026-11-01)

The current comparison includes **richer coarse-cluster summaries and a budgeted adaptive cut**, retaining complete world prefixes as the stable index. On 2026-11-08 the user proposed one-level lookahead, spatial moments and multiple centers, adding fixed two/four-component summaries to the discussion. The flat per-light control establishes room to improve aggregation, while ranking these repairs requires matched comparisons. First check whether a bounded representation preserves quality, then choose how to share it.

The new [offline diagnostic](../../../builds/light-sampling/20261101-moving4-point-m2/discussion_flat_proposals.py) and [results](../../../builds/light-sampling/20261101-moving4-point-m2/discussion-flat-proposals.json) reuse the reviewed conditional RGB M2. Emitters, conditional sampling and actual evaluation stay fixed; each emitter's A weight is normalized once over the complete list. A second control uses mass times the unoccluded cosine integral of a uniformly emitting unit cube. Both are O(N) diagnostic constructions, not per-pixel runtime proposals. The table reports variance inferred from actual conditional moments relative to each scene's **original camera A=1**, spanning both windows; side-27 includes four windows across its two groups.

| Receiver | Flat per-light A | Flat cube geometry |
|---|---:|---:|
| moving-4 side-26 | 0.317 | 0.329 |
| moving-4 side-27 | 0.197–0.200 | 0.235–0.237 |
| moving-4 top-28 | 0.328–0.331 | 0.329–0.331 |
| ls1 pillar | 0.274 | 0.275–0.276 |
| ls1 wall | 0.138 | 0.145 |

Neither control assigns zero probability to any positive-M2 emitter in the twelve point-window observations. Cube geometry omits texture/visibility, uses exact axis normals within 1e-5 of the audited normals, and is checked against the existing face integral; its variance evaluation still uses M2 with actual textures/visibility. The same per-light A beating camera A at all five points establishes substantial room to repair coarse aggregation. It does not establish full-image acceptance or that a small cut can attain these results.

At side-26/27, the low cluster's probability drops from tree A's 76.07% to flat per-light A's 14.43%/8.40%; cube geometry assigns 17.01%/14.45%. This distinguishes the coarse-cluster excess from the remaining unoccluded geometric allocation. The cluster still contains 1570/1545 lights with positive geometric integrals but no observed positive M2. Texture and visibility effects have not been separated, so the whole cluster cannot simply be assigned zero probability. Existing unoccluded repair potential can be pursued before reopening visibility guiding.

For a fixed receiver, if every cluster weight equals the sum of its descendant per-light weights, conditional probabilities along the tree telescope to the flat per-light distribution. Re-estimating from aggregate mass and a large AABB does not preserve this property. Radius/cosine saturation in current prefix A and zero-near-distance monopolization in the old near/far path have different triggers; their shared issue is loss of internal contribution distribution. The same per-light A improving all five points supports investigating this layer jointly without orientation-specific patches.

The proposed query first constructs a complete, disjoint cut, refining clusters with large geometric spans, mixed near/far content or receiver-hemisphere crossings. It then normalizes across cut nodes, selects one cluster and uses existing conditional sampling to choose one emitter; the actual PDF multiplies cluster and within-cluster selection probabilities. Refining only nodes with high current A weights, or stopping solely because one-level weights barely change, can miss underestimated nearby lights. Refinement priorities remain to be designed. Offline budgets of 16/32/64 entries can probe the five retained M2 points while counting construction visits and query work, without new GPU capture. These budgets are candidates, and finite M2 is not a runtime oracle.

This borrows the error-driven refinement structure of [Lightcuts](https://www.graphics.cornell.edu/~bjw/lightcuts.pdf); its approximate-illumination error bound is not a variance guarantee for this single-light sampler. If a small cut approaches the flat control, shared cut **structure** can amortize construction while receivers evaluate weights using their own positions/normals. Prefer considering fixed world-space partitions for this movement-stability objective: a fixed world point's cut/PMF should not jump solely because the camera moves. Screen-tile sharing requires separate checks of partition changes. Sharing amortizes cost but cannot repair an overly coarse cut.

This round reuses saved originals for diagnosis and updates discussion records only. No shader, defaults or conditional-emitter sampling changed, and no new implementation/GPU experiment was delegated. The cut remains a recommendation; entry budget, refinement rule and sharing scope are undecided.

### Lookahead, spatial moments and multiple-center summaries (2026-11-08)

All three approaches approximate the sum of per-light weights within bounded cost. They differ in whether queries read finer existing nodes or construction prepares a richer distribution summary.

| Candidate | Representation | Cost and existing evidence |
|---|---|---|
| One-level lookahead and summed child weights | Preserves spatial differences between existing children | Each internal candidate in a four-way tree evaluates at most four children. Existing one/two-level cross-brick controls improve all five points, but both sides remain worse than camera A and the main low-cluster mismatch persists. See the [native-slot review](light-sampling-world-grouping-prompt.md#independent-review-2026-11-01). |
| Single-center spatial moments | Mean locates energy, scalar variance describes spread, and full covariance also captures elongation direction | Current models 4/5 use a 48B node with mean and covariance trace. Full covariance has six independent elements, adding five floats when replacing scalar variance. Actual stride depends on packing. Statistics still need a receiver-dependent weight model. |
| Fixed two/four-component summary | Several small distributions with separate mass, center and scale can describe separated light groups | Each candidate evaluates a fixed number of components and sums them. Position, mass and scalar variance require five floats per component, or 20B of payload before alignment and support bounds. Additional storage and construction compression must be counted. |

The existing [moment and SG models](../../../shaders/techniques/debug/LightSampling.glsl) have been tested in the earlier five-scene experiment, with both gains and severe local regressions. Those [results](light-sampling-reassessment.md#dedicated-preset-rerun-1080p) apply to the specific models and inputs. Full covariance, higher moments and multiple-center summaries remain distinct candidates. The [SG light-tree paper](https://doi.org/10.1145/3680528.3687647) describes flux-weighted mean/variance merging and SG importance estimation used as a reference for the current single-center implementation.

Approximation reasoning suggests that low-order moments are most useful when the receiver is far from a cluster with a small angular extent. Near-field distance weights vary rapidly, and hemisphere boundaries introduce clipping, making a few moments less able to preserve isolated nearby lights or separated groups. Full covariance distinguishes spherical from elongated distributions, but finitely many moments do not uniquely recover a multimodal distribution. Success also depends on the formula converting those statistics into weights.

For the multiple-center candidate, each component would retain its own emission mass, weighted position and spread. Queries evaluate and sum component importance, allowing groups at different distances and directions to contribute separately. If components simply copy existing children and use identical weight functions, their importance equals one-level lookahead and precomputation only changes data access. Additional quality potential comes from different component grouping or weight models. Within each fixed parent cluster, construction could collect child summaries and compress them to two/four components, sharing this work across receivers.

Compression must preserve total mass and count each emitter's mass once. Component selection and merge order should depend on fixed world input, so camera addressing or atomic allocation order cannot change a summary for the same emitter set. Minimizing flux-weighted spatial error alone can still absorb a small group important to nearby receivers, so fitting quality needs full-PMF evaluation against the retained true M2. A component center behind a surface may represent lights in front, so zero-weight decisions require conservative support bounds. Summaries estimate actual tree-branch importance, and the final PDF multiplies the actual normalized branch probabilities.

The five-point, two-window screening in the [cluster-summary experiment prompt](light-sampling-cluster-summary-prompt.md) is complete and has passed the independent evidence review below; the current candidates are not accepted repairs. The old per-light arrays could not be recovered, so all three groups were recaptured at their original receiver pixels. Moving-4's 4,243 lights and full voxel hash match the historical input. The current ls1 tree has 10,235 lights, of which the old 3,921 are a strict subset. The first ls1 capture failed the old fixture's light-count guard; after rebinding all lights, all controls were regenerated for the new cohort. The repeated side-27 groups bring the total to 12 point-windows. The [result index](../../../experiments/light-sampling/20261108-cluster-summaries.json) records originals, per-window ratios, attribution and costs. The [recoverable reference pack](../../../experiments/light-sampling/20261108-cluster-summaries-reference.npz) holds both trees, physical slot counts, world light keys, per-light means/M2, receivers and provenance.

The table shows RGB variance relative to same-cohort world-prefix A, window 1/2. Flat per-light A diagnoses the effect of cluster approximation.

| Point | One-level lookahead | K1 | K2 | K4 | Camera A | Flat per-light A |
|---|---:|---:|---:|---:|---:|---:|
| side-26 | 0.4428 / 0.4429 | 102.94 / 102.93 | 0.1549 / 0.1550 | 0.2596 / 0.2597 | 0.1354 / 0.1355 | 0.0429 / 0.0430 |
| side-27 | 0.3537 / 0.3537 | 83.37 / 83.37 | 0.1190 / 0.1191 | 0.2362 / 0.2363 | 0.0923 / 0.0923 | 0.0183 / 0.0184 |
| top-28 | 0.5304 / 0.5307 | 3.216 / 3.215 | 1.5476 / 1.5473 | 10.125 / 10.120 | 0.3934 / 0.3937 | 0.1274 / 0.1279 |
| ls1 pillar | 0.6062 / 0.6056 | 0.8523 / 0.8515 | 0.5341 / 0.5338 | 0.2446 / 0.2445 | 0.9756 / 0.9796 | 0.1689 / 0.1681 |
| ls1 wall | 0.4328 / 0.4325 | 0.7041 / 0.7036 | 0.4305 / 0.4306 | 0.1983 / 0.1983 | 1.7702 / 1.7686 | 0.0903 / 0.0900 |

Fixed Ward compression preserves moments but can lose nearby lights. At side-27 the critical light accounts for 53.7% of its ancestor's per-light A sum, yet K1 gives that cluster about 0.00478% of the sum; the light falls from 3.99% probability with one-level lookahead to 0.0178%. K2 raises it to 10.82% and reduces probability on the low root child with no observed M2 from about 76.07% to 55.87%/56.11% on the two sides, but still loses to camera A there. K4 is strongest at ls1 but drops top-28's main penalty light from 34.58% with lookahead to 3.34%. Quality does not improve monotonically with component count. All 84 variance results across 12 point-windows have no positive-M2 support failure. Float32 summaries differ from the high-precision reference by at most 2.46×10⁻⁵ in full-PMF total variation; the observed regressions do not come from that precision difference.

| Method | Physical node slots/sample, mean range / max path | Summary kernels/sample, mean range / max path | Extra persistent / peak allocation |
|---|---:|---:|---:|
| Prefix A | 31.86–39.31 / 52 | 0 | 0 |
| One-level lookahead | 91.45–105.73 / 132 | 0 | 0 |
| K1 | 36.75–39.73 / 52 | 9.07–16.40 / 21 | 0.12 / 10.41 MiB |
| K2 | 36.13–38.34 / 52 | 18.62–31.95 / 42 | 0.32 / 20.90 MiB |
| K4 | 38.06–38.95 / 52 | 36.23–66.38 / 83 | 0.61 / 39.72 MiB |

Slot checks include empty slots. The index separately counts selected-node rereads, hemisphere checks, component headers, normalization and requested bytes. Additional space models directly indexed 16/44/84 B persistent records, a full-tree worklist and build scratch. K2 evaluates 32,857/72,578 candidate component pairs per build at moving-4/ls1, including work within bricks. These are static costs of a separate summary build; integration into the tree build would need measurement. This round's fixed Ward-plus-ball summaries cannot replace one-level lookahead across the five points; keep K2 as a diagnostic. The top-28 counterexample and added build work warrant stopping at static comparison. No candidate GPU timing or default change was made. Reproduce with the [summary screening script](../../../scripts/light_sampling_cluster_summaries.py) and the [capture and archive validation script](../../../scripts/light_sampling_cluster_capture.py).

### Independent cluster-summary review (2026-11-08)

Reviewed the three groups from candidate `5d9e9c355`: verified SHA and length for 192 native files, independently extracted all 12 point-windows from native statistic images, and matched the tracked reference package element by element. The maximum per-pixel second-moment identity relative error was 4.25×10⁻⁷. Direct native-tree parsing verified reachable nodes, light keys, mass and physical empty slots. Receiver-pixel material, origin/normal and existing color/depth constraints were also checked. Shader differences between the three frozen sources and the baseline are limited to fixture data and bindings, with no new importance implementation mixed into capture.

The [independent recomputation script](../../../builds/light-sampling/20261108-cluster-summaries/review/recheck.py) uses a separate scalar pair search retaining actual component membership to rebuild K1/2/4 for all three trees. Components, merge counts and candidate-pair counts match the submitted results. Re-evaluating weights from the formulas, traversing 72 complete tree distributions and computing all 84 variances including the flat control reproduced the report, with zero positive-M2 support failures. Another check recomputed node/component means and maximum paths for 60 query-cost cases and each layout's allocation arithmetic. The [review array summary](../../../builds/light-sampling/20261108-cluster-summaries/review/independent-review.json) and tracked index preserve scope and hashes.

Actual membership identifies the failure mechanism. At side-27, K1 merges the critical nearby emitter into a 17-light component at node 153. Its `dot(n,mu-origin)+sqrt(2*variance)` is −3.879, pushing the cosine term to 0.001. K2 retains the emitter in a four-light component with cosine 1. At top-28, K4 places the critical nearby emitter in a 76-light component at node 70, with center height 29.379, receiver height 76.003 and radius about 37.721. Its hemisphere-direction height plus radius is −8.903, again reducing cosine to 0.001; the corresponding K2 values are 18.304 and about 0.375. The original node AABB passes the half-space test, and the emitter set and positive-probability support remain intact. Distortion occurs in subsequent component importance. Moment conservation does not make a variance ball enclose isolated nearby emitters; more components and smaller variances can further underestimate them.

Accept the specified-point screening evidence, but none of the current summaries as a five-point repair. K2 approaches camera A on the sides but still has about 14%/29% higher variance, and about 55% higher variance than prefix A at top-28. This conclusion is specific to Ward merging and the ball kernel, not a rejection of every moment/multiple-center model. The ls1 comparison uses a new emitter cohort and cannot retroactively reproduce the old 3,921-light near/far failure batch. M2 remains a finite-sample estimate; unobserved contribution does not prove strict invisibility.

Peak extra allocation reserves scratch for all 269,764 native slots; the three groups contain only 7,490/7,490/18,119 reachable nodes. This is the chosen directly indexed layout's cost, not a memory lower bound for summaries. Compaction or fused-build costs remain unverified. Review ran no new GPU work and establishes neither equal-time image gains nor continuous-motion stability. Further design must address underestimation of important members near receivers and hemisphere boundaries.

## Weight experiment status (2026-10-31)

The latest [targeted iteration](../../../experiments/light-sampling/20261031-nearfar-iteration.json) compares A, original near/far and A + near/far in ls1/2/4/5,
accepting 12 raw 1-spp/32-frame quality captures, 24 timings and matched two-window references. The [comparator](../../../builds/light-sampling/20261031-nearfar-iteration/comparison-final.html) contains 24 primary images.
The default, candidate and point diagnostic are separately committed as `7de5c430`, `78823b54` and `fa8ac641`. Both user-selected ls1 faces still regress; the selected ls4 birch-leaf face improves, while the selected ls5 wall does not reproduce the earlier regression.
Per-light M2 identifies zero-near-distance branching and probability mismatch among contributing lights. Size-saturated near distance provides partial offline point relief, remains worse than A, and has no new runtime model.
See the [targeted findings](light-sampling-reassessment.md#half-space-default-and-targeted-near-far-iteration-2026-10-31) for regions, excluded input drift/incorrect frame counts, costs and nonfinite events.

The earlier five-scene screening uses the user-selected native 1920×1080 `light-sampling-1` through `light-sampling-5` presets.
The [tracked results](../../../experiments/light-sampling/20261031-weights-presets1080.json) retain 35 quality cases, 70 matching timings,
five two-window references and 70 images. The [dedicated-preset findings](light-sampling-reassessment.md#dedicated-preset-rerun-1080p)
separate crop error, full-image/fixed-ROI variance, input identity, full build/initial cost and limitations.
Original weights remained default when the rerun completed. The user subsequently reviewed the images, accepted half-space A
as default, and requested further research and iteration on near/far's local regressions. The experiment itself stays disabled
by default. Acceptance does not validate production GI, ReSTIR, or every numerical path; the recorded nonfinite events remain open.
The rerun's description of near/far as most consistent used original weights as its baseline. Against A, scene 2's reference-crop
MSE and fixed-floor RGB variance are respectively 1.811 and 1.872 times A. These ratios use existing raw statistics; lower
full-image variance does not resolve them. The user then identified three priority regressions: ls1's wall facing toward
the area behind the camera, ls4's left-side grass/leaves, and ls5's small wall at middle-left. Scene 2's floor remains
a supplemental comparison rather than replacing these three regions.
Original near/far model 3 does not apply A's half-space rejection. The model-6 controlled comparison and subsequent point validation are now complete and do not support automatically replacing A.
See the [completed targeted-iteration prompt](light-sampling-weight-experiment-prompt.md#completion-status-2026-10-31) for implementation entry points and original authorization. The [legacy 720p supplement](../../../experiments/light-sampling/20261031-weights-results.json)
stays separate; its dark-region regressions are not relabeled as current dedicated-scene results.
Observed input drift was handled with matching-cohort recaptures, without diagnosing its cause. Rare nonfinite RGB remains
unrepaired; statistics apply to common complete pixels. This run does not retroactively validate old 1080p arity/lookahead
controls, add cuts/MIS/visibility guiding/H-PLOC, or relabel the old fixed-point reference for these scenes.

## Motion stability candidate (2026-10-31)

The user supplied four `light-sampling-failure-moving-*` presets and authorized the experiment chat to implement a minimal repair, validate it and proactively request review from the discussion chat.
At that round's starting point, brick boundaries aligned to world 16-block cells, but outer light-tree grouping used Morton indices relative to the camera window and could change after a brick crossing.
The candidate only publishes light-tree brick roots into world-brick-coordinate modulo-G Morton slots, preserving the voxel window, brick-local subtrees, A weights, 32B nodes and actual PDFs.
Grouping should remain stable for the same emitter set; periodic wrapping can enlarge cluster AABBs and still requires quality checks. Input-set changes are outside that invariance guarantee.
The [motion stability prompt](light-sampling-moving-stability-prompt.md) records the derivation, counterexample, four preset identities, implementation scope and return notification.
The candidate is implemented in `0d1629ae`, with targeted experiments and independent discussion-side review complete: equal-input stability passes, but quality acceptance fails. In the real equal-input pose 2→3 pair, the old topology and two fixed-world-point PMFs change substantially; the new topology matches and all three diagnostic full PMFs have zero difference.
New/old common-complete-pixel RGB variance ratios are 3.376, 3.839, 0.0642 and 1.900 across the four presets, independently reproduced from raw data. At the first fixed point, the old tree's null-sample probability rises from 14.05% to 92.89% across the boundary; the new tree stays at 14.41%. This is selection waste, not proof of bias. The candidate remains on the experiment branch without promotion to the accepted baseline, and this review does not edit shaders. G=16 static poses do not validate G=64 quality or the first continuous-motion crossing frame.
See the [motion stability results](../../../experiments/light-sampling/20261031-moving-stability.json) and [viewer](../../../builds/light-sampling/20261031-moving-stability/comparison.html) for originals, wrapping counterexamples, PDF audits, compilation and representative costs. This does not reopen H-PLOC or expand into another algorithm experiment.

## Complete world-coordinate grouping implementation (2026-11-01)

The user authorized replacing periodic modulo grouping, rerunning all four moving presets and having the experiment chat notify the discussion chat for review on completion.
Complete world-brick prefixes define every parent cluster. Each layer stores only cells intersecting the current window; addresses subtract that layer's origin without changing parent-child identities.
Each parent owns consecutive child slots. At most eight upper records continue merging along full world prefixes with unary-chain collapse, retaining current 32B nodes, default four-way A and emitter sampling.
Capacity may vary with G/arity. World-coordinate keys and array indices are separate, without adding general sorting, a hash framework or a construction-mode option.
World prefixes are implemented in `d29c5d178`; 4374 CPU layouts and four GPU functional fixture groups pass, but quality does not pass overall. G16 world/camera A full-image RGB variance ratios are 0.0090/0.4440/2.5223/6.6227. G64 pose 1 A ratio is 1.1490 with unmatched BRDF; poses 2/3/4 are blocked by real input drift at the full voxel pool. Eight ABBA runs record complete build and initial costs without establishing a speedup. Quality coverage is 14/24 with 28 valid images and explicit missing slots. The [round result](../../../experiments/light-sampling/20261101-world-grouping.json) preserves provenance and limits.
Independent discussion-side review is complete: eight new trees match a full world-voxel prefix oracle; 42 same-pose input snapshots, five quality ratios and eight per-frame build timings pass independent checks. Complete-world grouping is accepted as a mechanism, but confirmed G16 poses 3/4 regressions and higher build cost reject promotion to the replacement baseline. The candidate remains preserved and review does not edit shaders. Current emitters differ from the modulo round, preventing direct cross-batch ranking; continuous motion, complete default-grid quality and nonfinite causes remain unaccepted.
The [world-grouping design and prompt](light-sampling-world-grouping-prompt.md) governs implementation, capacity, raw evidence, branch checks and return notification. Preserve previous modulo results as failure evidence.

## Architecture reassessment (2026-10-30)

The [architecture reassessment](light-sampling-reassessment.md) revisits whether per-query light-tree traversal
fits this workload, separating topology, receiver-dependent importance, and shared computation. Following the
user's correction, regional estimates target bounded sets of tree clusters rather than individual lights;
standalone small-candidate RIS is not a scalable replacement, and ReGIR is a complementary direction.
The record expands cluster weights, regional cuts, visibility statistics, and their coupling to topology updates.
Its diagnostic proposals do not accept a replacement, change the baseline above, or start experiments.
Global cross-brick clustering remains the agreed H-PLOC scope.
The current discussion focuses on weight costs and cut semantics; visibility and H-PLOC are deferred.

## Global H-PLOC clustering candidate

This candidate is currently on hold. The following preserves its scope and research context.

The confirmed scope allows global emitter clustering across 16³ brick boundaries. Bricks own emitter extraction,
not the final light-tree grouping. This decision is recorded in experiment commit `5c022ba1`; no H-PLOC builder
has been implemented.

The existing experiment already has one global light tree over the current voxel volume. The binary option
produces a global binary tree; the current default is four-way. Construction first creates a complete subtree
for each brick, then combines brick roots through a fixed Morton hierarchy. Ancestors spanning bricks already
exist, but a local cluster cannot first group only some emitters from two neighboring bricks while leaving
their remaining emitters outside. Here, global clustering removes that complete-brick-subtree constraint; it
does not introduce the first global root. H-PLOC candidate merges remain bounded by Morton ranges and local
neighborhoods.

The candidate for discussion is a globally Morton-ordered emitter list, H-PLOC binary construction, and conversion
to a compact four-way tree for the existing sampler. [Section 3.4 of the H-PLOC paper](https://gpuopen.com/download/HPLOC.pdf#page=8)
already provides wide-tree conversion: repeatedly open the internal child with the largest surface area until
the target child count is reached or all children are leaves. The paper's complete pipeline outputs a four-way
tree, but first constructs a binary tree and then runs a separate conversion kernel. Binary construction and
wide conversion can each use one kernel launch. This is neither native four-way clustering nor a single
dispatch for the whole preparation pipeline. Removing the intermediate binary topology requires further
algorithm changes, rather than setting a branching parameter to four.

The paper's ray-tracing results do not establish lower light-sampling variance here.
A proposed first comparison retains the current emitter set, cube sampling, and
node weights, with visibility guidance and lookahead disabled. This is a discussion starting point, not an
experiment assignment.

Questions to resolve next:

- Merge criterion: start with merged-AABB surface area or introduce a light-sampling objective. The latter also
  changes the topology objective and needs an explicit attribution argument.
- Binary-to-four-way conversion: the paper's largest-surface-area expansion rule is a first-version candidate;
  verify output layout and scheduling when converting temporary binary indices into contiguous children.
- GPU execution: verify global Morton ordering, scratch capacity, subgroup capabilities, and cross-workgroup
  publication. Do not assume the paper's execution mechanisms already exist in the GLSL path.
- Experiment criteria: record all preparation and construction costs, initial-sampling time, raw sample quality,
  and PDF/emitter-set consistency. Comparing only the core construction kernel is insufficient, and tighter
  grouping does not itself establish lower noise.

### Expected sampling-quality impact

There are no H-PLOC sampling-quality measurements yet. It is a topology candidate worth comparing, with a
hypothesized benefit. Current node weights approximate subtree contribution using aggregate emission mass,
the AABB center, bounding-sphere radius, and receiver normal. Clustering could reduce branch-probability error
by avoiding broad bounds across empty space or grouping nearby emitters across brick boundaries separately.
Near-field, sparse, or uneven emitter distributions are plausible beneficiaries, not guaranteed wins for every
receiver.

H-PLOC's area-based merge criterion does not use the receiver, normal, emission mass, or actual visibility.
Smaller area does not guarantee a better contribution estimate, nor does it identify lights separated by a wall.
Current leaves are equal-sized unit cubes and the baseline already uses Morton spatial grouping. The paper's
BVH-quality benefits for differently sized geometry cannot predict a noise-reduction percentage here.
With conditional emitter sampling and actual contribution definitions fixed, and support and PDFs correct,
topology changes affect sampling variance while preserving the target mean.

The existing five-scene spatial-grouping comparison (record commit `181878b9`) was itself mixed: without
lookahead, dark-region variance rose about 14% in scene 1 and fell about 7% in scene 2. This motivates studying
grouping but is not evidence of H-PLOC benefit. Judge the first comparison by sampling error/variance and
contribution tails with the same emitter set, four-way branching, and node weights; then include all build costs
in an equal-time comparison. Tree area, SAH, and visible-hit rate are explanatory diagnostics, not substitutes
for sampling-quality measurements or evidence that H-PLOC can replace lookahead.

## Scope and decisions

- Target a pure Iris shaderpack. Do not depend on modifying Iris or a companion mod to supply light geometry.
- Focus on NEE light selection and visibility guidance. Model geometry, per-quad storage, UVs, cutout evaluation,
  and exact sampling of a selected emitter's surface are deferred.
- Treat most Minecraft emissive blocks as omnidirectional for importance estimates: point or sphere approximations
  are sufficient initially; a cube is a more accurate later model. Emission cones, emission-direction vMF lobes,
  and position-normal cross-covariance are outside the first experiment.
- Receiver normals, BRDFs, and outgoing directions still affect importance. Omnidirectional emission does not
  remove these terms.
- Separate light selection from conditional sampling on the selected light. A cheap proposal approximation must
  not silently redefine the actual light contribution.
- Use visibility statistics to change branch probabilities first. Additional stochastic subtree pruning is deferred.
- Evaluate ordinary NEE before integrating ReSTIR, temporal reuse, or denoising.

## Data ownership and flow

The light BVH belongs to the emitter population. Visibility statistics belong to the receiver's spatial brick.
They are independent structures; a light-tree node does not own a copy of every receiving brick's statistics.

| Order | Stage | Purpose |
|-------|-------|---------|
| 1 | Light extraction and hierarchy construction | Build an emitter-only hierarchy; extraction details remain deferred |
| 2 | Visibility-statistics preparation | Sample valid shading locations in each active brick and collect directional first-blocker distances |
| 3 | Light-tree traversal | Combine flux, geometric/BRDF importance, and the brick's visibility estimate to choose a leaf |
| 4 | Actual light evaluation | Sample the chosen emitter as required, trace real visibility, and divide by the actual sampling probability |
| 5 | Statistics update | Initially collect and freeze statistics before evaluation; incremental updates are a later experiment |

Existing code anchors are [voxel storage](../../../shaders/techniques/voxel/Voxelization.glsl),
[voxel tracing](../../../shaders/techniques/voxel/VoxelTrace.glsl), the
[current GI estimator contracts](global-illumination.md#estimator-contracts), and
[program registration](../../../scripts/programs.main.kts). These are integration context, not an assertion that this
light sampler already exists. No new pass placement, resource binding, or user setting has been chosen.

## Light BVH

### Initial representation

Start with a compact binary spatial hierarchy containing emissive instances only. A Morton/LBVH construction is a
baseline candidate, not a claim of optimal sampling quality or GPU update cost. Leaves identify lights; internal
nodes aggregate spatial bounds and total flux. Exact packing, construction, refitting, and emitter identifiers
remain implementation decisions.

Do not directly reuse the voxel occupancy hierarchy as the light tree. Its occupancy role and 64-way branching do
not establish an efficient light-sampling layout. Wider branching reduces depth but increases child reads and
importance evaluations.

For an omnidirectional point-light approximation, a cluster's ideal unoccluded scalar importance is proportional to

$$
I_C(x)=\sum_{\ell\in C}
\frac{\Phi_\ell}{4\pi r_\ell^2}
f_r(x,\omega_\ell,\omega_o)\max(n\cdot\omega_\ell,0).
$$

A node stores an approximation to this sum, not its exact value. The scalar treatment of colored flux and the
near-field distance regularization are still to be selected. AABB-plus-flux is the initial simple candidate; spatial
Gaussian summaries remain an alternative if they beat spending the same cost on more detailed traversal.

### Query-dependent refinement

A distant compact cluster can use a cheap aggregate. A nearby or broad cluster, a cluster crossing the receiver
horizon, or one spanning a narrow BRDF lobe may need refinement before normalizing competing branch weights.
Refining only after choosing the wrong parent cannot recover probability already lost at that parent.

A visibility query must account for the cluster's angular footprint and distance range. Evaluating only its center
can suppress a visible descendant behind a doorway or confuse lights before and behind a wall. The initial
heuristic is weak visibility modulation for broad clusters and stronger modulation after they become compact.
The compactness rule and modulation strength are experimental parameters, not settled thresholds.

## Directional blocker-distance statistics

### Meaning and sampling population

For brick B, let D(x, omega) be the distance to the first blocker from position x in direction omega. The desired
quantity is a survival probability

$$
S_B(\omega,r)=P(D\ge r\mid B,\omega).
$$

This probability averages over the chosen origin population and angular filtering. Visibility at one exact origin
and direction is deterministic; the variance describes pooling, not randomness in the scene.

The first candidate uses representative actual NEE shading locations within each 16 x 16 x 16 brick. This is an
initial experimental choice: the user left center versus multiple-origin sampling open.

| Origin population | Meaning and limitation |
|-------------------|------------------------|
| Brick center | Center visibility only; the center can be inside solid geometry or across a wall from a query |
| Multiple valid positions | Visibility averaged over those positions; separate rooms can still be mixed |
| Sampled actual shading locations | Workload-relevant average; depends on the view/path workload and needs updates |

Increasing SH degree cannot recover origin information already discarded by sharing one record across a wall.
For an actual query, direction and distance are measured from its shading position, while the statistics still
represent the brick's pooled population.

### Bounded distance transform and two moments

Use a monotone transform with a positive scale s:

$$
T(D)=\frac{D}{D+s},\qquad T(\infty)=1.
$$

The scale is not a hard distance cutoff. Store directional first and second raw moments of T:

$$
m_1(\omega)=E[T],\qquad m_2(\omega)=E[T^2].
$$

Represent each moment field with low-order SH. Maximum degree 2 gives 9 coefficients per moment: 18 FP32 values,
or 72 bytes per brick, excluding counts, confidence, history, and update storage. Precision and SH degree remain
experimental choices. The transform compresses differences between far distances; the scale therefore affects
useful distance resolution.

At query distance r, reconstruct mu and the variance, then use a Cantelli/VSM-style estimate:

$$
t=T(r),\qquad \mu=m_1,\qquad \sigma^2=m_2-\mu^2,
$$

$$
\widehat S_B(\omega,r)=
\begin{cases}
1,&t\le\mu,\\
\frac{\sigma^2}{\sigma^2+(t-\mu)^2},&t>\mu.
\end{cases}
$$

For true moments of a defined distribution, the expression is an upper bound on its survival probability.
Finite sampling, truncated SH, and pooling over origins do not make it a guaranteed visibility bound for the
current shading point. Two moments do not uniquely determine a distance distribution, its tail, or escape
probability. This is a proposal heuristic; a positive tail is not a substitute for sampling support.

Reconstruction can produce invalid moments. For a variable in [0, 1], feasible moments satisfy

$$
0\le m_1\le1,\qquad m_1^2\le m_2\le m_1.
$$

Any projection into that feasible set is a numerical repair, not restoration of a conservative scene bound.
Zero variance permits a zero tail beyond the mean; the full sampler must still retain required support.

### Trace outcomes and training

A confirmed first hit is a distance observation. An unobstructed trace stopped at a light or finite limit R only
establishes D > R; it is right-censored data, not a hit at R and not proof of D = infinity. Step exhaustion is not
free-space evidence. Exiting the voxel grid only inherits an explicitly chosen boundary model; it does not prove
that the Minecraft world is empty beyond the grid.

For the first experiment, use dedicated uniform-sphere probes from sampled shading locations, build the statistics,
then freeze them and evaluate with separate query samples. Synthetic scenes should define their outside geometry
so that exact hits and true misses are distinguishable.

Existing NEE rays are sampled with a light-dependent angular distribution and often stop at a selected light.
Naively accumulating them into SH changes the intended statistics. Reusing those rays requires an appropriate
sampling-weight or fitting treatment and handling of censored observations; this is deferred.

Later online updates must keep the statistics used to generate a sample consistent with its recorded probability.
Previous-frame read state and separate current accumulation are a possible implementation, not a selected resource
layout. Geometry edits, newly active bricks, workload changes, and history invalidation remain later experiments.

## Sampling probabilities and estimator

Let q0 be the baseline tree distribution with support wherever a light can contribute. Let qv use visibility-modulated
node scores:

$$
w_C=I_C\,\widetilde v_C,\qquad
p_v(C\mid P,x)=\frac{w_C}{\sum_{j\in children(P)}w_j}.
$$

Here the visibility factor includes the chosen treatment of cluster extent and statistical confidence. A branch
with no usable statistics uses neutral visibility. If all child scores vanish, a normalized supported branch rule
must still be defined, using the baseline probabilities. Unknown visibility must not become a proof of zero
contribution.

Each tree's leaf PMF is the product of its branch probabilities. Preserve exploration by mixing the complete
distributions:

$$
q(\ell\mid x)=\alpha q_0(\ell\mid x)+(1-\alpha)q_v(\ell\mid x).
$$

Alpha = 0.1 is a starting experiment value, not a tuned default. This provides

$$
q(\ell\mid x)\ge\alpha q_0(\ell\mid x).
$$

Mixing baseline probabilities separately at each tree level does not provide the same leaf-level guarantee;
its worst-case relative protection can decay with depth. Evaluate the full mixture PMF for the selected leaf,
including both components, rather than dividing only by the probability under the selected technique.

For a discrete point-light experiment with true contribution C_l, use C_l / q(l). When surface sampling is added,
include its conditional density and the appropriate measure conversion. Real visibility and the actual emitter
model belong in C_l; the learned survival estimate belongs only in the proposal. An inaccurate proposal affects
variance without biasing this estimator when support and probabilities are correct. This statement does not by
itself establish correctness of later ReSTIR reuse or of existing trace-termination approximations.

Additional roulette would require dividing surviving contributions by their continuation probability. Pruning
random subtrees and merely renormalizing the survivors is not an established estimator. A shared random distance
cutoff also correlates pruning events; their marginal survival probabilities cannot be multiplied as if independent.
Roulette can save work but increases variance per attempt, so it needs an equal-time evaluation of its own.

## First experiment

Hold the light tree, emitter powers, scene, receiver queries, and training-ray population fixed across variants.

| Variant | Visibility representation | Question |
|---------|---------------------------|----------|
| A | None | Baseline sampling variance and cost |
| B | High-resolution directional and distance histogram per brick | Whether spatially shared distance statistics are useful |
| C | SH first and second moments from the same training samples | Whether the proposed compression preserves that benefit |

B is an expensive experimental reference, not a production feature. If B helps and C does not, investigate
representation loss. If B also fails, inspect origin pooling, cluster queries, and visibility weighting before
increasing SH degree. Freeze training first; do not let each variant's feedback generate a different training set.

Use three initial scenes:

- Unoccluded near and far lights: expose pure overhead and unnecessary variance.
- A wall dividing two rooms inside one 16-cubed brick: expose lost origin dependence.
- A small doorway with a bright distant light: expose over-suppression and extreme contribution weights.

Measure the raw estimator against an all-light sum in the discrete test. Check PMF normalization, convergence
across independent seeds, error at equal sample count, and the tail of contribution weights. A mean within Monte
Carlo uncertainty is evidence, not a proof of unbiasedness. Visibility hit rate is diagnostic: sampling more visible
dim lights can improve it while worsening the lighting estimate.

An offline experiment can establish statistical value. GPU acceptance must measure equal-time error with probe
generation, statistics construction/update, tree traversal, and final shadow traces included. CPU timing does not
establish GLSL/Iris performance. Add movement, geometry changes, cold starts, and ReSTIR only after the static
sampling behavior is understood.

## Experiment results on 2026-10-29

The first CPU experiment is complete. The degree-2 SH candidate does not show stable benefit under these settings.
This is a statistical result for synthetic scenes, not an Iris/GPU performance result. Keep visibility guidance as
an experimental direction; prioritize the geometric node approximation and bounded lookahead before adopting
this SH representation.

### Protocol and controls

[The sampler experiment](../../../scripts/light_sampling_experiment.py) uses 512 omnidirectional point lights,
256 diffuse receiver queries, opaque AABB occluders, and an all-light reference in each scene. Each pooled training
seed uses 64 independent origin positions and 8,192 equal-solid-angle direction-bin midpoints: 524,288 probes per
scene. The angular grid is deterministic quadrature. Training seeds are 11, 29, and 47; alpha is 0.1 and the bounded
distance scale is 16 blocks. Computation uses float64, Python 3.14.7, and NumPy 2.4.6.

The high-resolution reference retains raw distances within each angular bin and queries their empirical survival
function. A fourth representation retains two moments in those same bins, without SH, to isolate two-moment
approximation from angular compression. All variants use the same training distances, tree, and receiver queries.

Nodes use a flux-weighted center and a sphere enclosing descendant light positions. Visibility is neutral while
radius/distance exceeds 0.5. Compact nodes average five directions over the sphere's angular footprint at the nearest
sphere distance, with modulation strength (1 - radius/distance) squared. These are concrete initial heuristics,
not an established optimal cluster query.

The metric below is the sum of exact single-sample variances across receiver queries, divided by the baseline's
sum. At equal samples per query it is also the expected image MSE ratio. Ranges span the three training seeds,
not confidence intervals over scenes. Construction and lookup costs are excluded.

| Scene | Distance CDF | Binned two moments | SH two moments |
|-------|--------------|--------------------|----------------|
| Unoccluded | 1.003 | 1.003 | 1.000 |
| Two rooms sharing a brick | 1.069–1.094 | 0.987–0.992 | 1.031–1.219 |
| Bright distant light through a doorway | 0.968 | 0.973–0.975 | 0.999–1.005 |

### Diagnosis

[The saved-data diagnostics](../../../scripts/light_sampling_diagnostics.py) separate three sources of loss:

- With probe origins matched exactly to each query, the distance-CDF variance ratio in the two-room scene changes
  to 0.948. Pooling origins is therefore a material source of loss in this fixture. This control is not a proposed
  per-pixel cache.
- Keeping high angular resolution while retaining only two moments is more stable here than projecting the
  moments into degree-2 SH. The result applies to this SH fit, feasible-moment repair, scale, and query rule;
  it does not reject every possible SH or moment representation.
- Summing child scores two levels deeper, with no visibility guidance, gives ratios of 0.577, 0.815, and 0.762
  in the open, two-room, and doorway scenes. Node importance approximation is already a substantial source of
  variance.

With that same two-level lookahead, binned moments give 0.789–0.796 in the two-room scene and 0.691–0.695 in the
doorway scene, an additional reduction relative to the matching geometry-only controls. SH gives 1.712–1.719 in
the doorway scene, roughly 2.25 times the variance of its geometry-only lookahead control. More accurate tree
evaluation can expose errors that coarse-node modulation previously hid.

Complete leaf aggregation is an expensive reference control. With matched origins and a high-resolution distance
CDF, its ratios are 0.075 for two rooms and 0.0069 for the doorway. With exact visibility they are 0.062 and 0.0034.
These controls demonstrate remaining statistical opportunity; they require unavailable cheap knowledge or
all-light work and are not production sampling algorithms.

### Correctness evidence and limits

The pooled and matched-origin runs each execute 160 Monte Carlo checks with 65,536 samples per check. The maximum
absolute mean deviation is 2.35 analytic standard errors. Tree-generated PDFs match enumerated mixture PMFs;
PMFs normalize and retain the alpha-times-baseline support bound. Independently enumerated box-face intersections
agree on 128 sampled visibility connections per scene, with explicit visible and blocked doorway cases. A
two-light mutation using an unsupported conditional PDF produces 2 instead of the correct 11 and is rejected.

The degree-2 SH candidate did not pass the benefit comparison. No shader code or old production path was replaced.
FP32 behavior, dynamic updates, ReSTIR integration, target-Iris compilation, visual acceptance, and equal-time GPU
benefit remain unverified. The next experiment should compare bounded geometric lookahead with a compact positive
directional representation; current results do not justify immediately integrating this SH layout into the pack.

### Reproduction and artifacts

Run from the repository root with the declared dependencies installed, using fresh output directories:

~~~powershell
python scripts/light_sampling_experiment.py --output builds/light-sampling/20261029-pooled-controls
python scripts/light_sampling_experiment.py --output builds/light-sampling/20261029-origin-controls --origin-mode query-matched --training-seeds 11
python scripts/light_sampling_diagnostics.py --inputs builds/light-sampling/20261029-pooled-controls builds/light-sampling/20261029-origin-controls --output builds/light-sampling/20261029-tree-controls
~~~

These commands were executed. The scripts use the lowest process priority on Windows; their elapsed times are not
CPU performance benchmarks. Each run saves source snapshots and hashes, settings, light and occluder inputs,
training samples, full leaf PMFs, and numerical results.

Local generated artifacts are ignored by Git and can be recreated with the commands above:

- [Pooled results and checks](../../../builds/light-sampling/20261029-pooled-controls/results.json).
- [Matched-origin results and checks](../../../builds/light-sampling/20261029-origin-controls/results.json).
- [Tree and leaf-aggregation controls](../../../builds/light-sampling/20261029-tree-controls/diagnostics.json).
- [Pooled variance maps](../../../builds/light-sampling/20261029-pooled-controls/variance-maps.png).

The original three-variant run and the lower-sample smoke run are retained separately under
[local experiment outputs](../../../builds/light-sampling/); they are not silently replaced by the later controls.

## Deferred alternatives and open choices

- Spatial SG summaries can improve directional BRDF integration but cost more arithmetic and can lose spatial
  anisotropy or multimodality. Compare them with equal-cost bounded lookahead; emission-direction SGs are not
  a priority under the omnidirectional-light assumption.
- Learned split corrections, neural proposals, and ReGIR are later alternatives. They do not remove the need for
  a valid base proposal, explicit probability evaluation, or accounting for update cost.
- For a fixed scalar subtree estimator Z_C and equal branch cost, minimizing second moment gives branch
  probabilities proportional to sqrt(E[Z_C^2]), not necessarily E[Z_C]. This is related to existing variance-aware
  guiding, not a claimed new algorithm, and does not directly transfer to the final ReSTIR estimator.
- Sampling origins, SH degree/precision, distance scale, probe budget, confidence treatment, cluster-footprint
  queries, exploration weight, and update cadence remain experimental choices.
- Emitter extraction, tree packing/construction/update, exact cube/surface sampling, transparency, UVs, and cutout
  treatment remain deferred implementation contracts.

The next experiment should address the node-approximation and directional-representation losses identified above.
The offline implementation does not establish a runtime sampler, GPU benefit, or visual acceptance.

## References

- [PBRT light sampling](https://www.pbr-book.org/4ed/Light_Sources/Light_Sampling):
  hierarchical light selection and leaf probabilities.
- [PBRT Monte Carlo efficiency](https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency):
  mixture sampling, MIS, and roulette.
- [GPU Gems 3 variance shadow maps](https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-8-summed-area-variance-shadow-maps):
  two-moment visibility estimates and the one-sided variance bound.
- [Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields](https://research.nvidia.com/publication/2019-05_dynamic-diffuse-global-illumination-ray-traced-irradiance-fields):
  related directional distance-moment visibility representation.
- [NEE++](https://diglib.eg.org/items/54459a33-0ee3-4411-8b4c-7bb7cb1612ed):
  visibility caching for direct-light sampling and probabilistic work reduction.
- [Hierarchical Light Sampling with Accurate Spherical Gaussians](https://doi.org/10.1145/3680528.3687647):
  the richer node-summary alternative discussed.
- [Variance-Aware Path Guiding](https://graphics.cg.uni-saarland.de/papers/rath-2020-siggraph-guiding-paper.pdf):
  accounting for the second moment of nested sampling.
