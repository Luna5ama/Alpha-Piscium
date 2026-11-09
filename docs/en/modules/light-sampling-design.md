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

## Current selection and backlog (2026-11-08)

| Item | Confirmed choice |
|------|------------------|
| Conditional emitter sampling | Retain projected-solid-angle cube sampling without repeating its adoption comparison |
| Node weights | Half-space A (32B) remains the implemented default; this clustering comparison fixes A + Near/Far, separately checking its current-tree cost and benefit without switching defaults |
| Spatial grouping | Prioritize clustering repair; retain world stability for identical emitter sets while allowing replacement of fixed Morton-prefix parents and global cross-brick clustering. World Prefix remains an unaccepted quality control |
| Branching | Default to four-way, retaining binary and eight-way alternatives; all run without geometric lookahead |
| Lookahead and moment summaries | The user reopened one-level lookahead and K1/K2/K4 for the eight-hour exploration; judge actual variance/time and retain earlier evidence |
| Visibility guidance | Exclude from the current runtime baseline; defer further discussion at the user's request |
| H-PLOC | On hold at the user's request; retain the global cross-brick scope if reopened |

### Current execution: eight-hour variance and time exploration (2026-11-08)

The user authorized about eight hours of continued research, implementation, and runtime iteration, explicitly reopening lookahead/K and selecting **Camera grouping + A + Near/Far** as the performance baseline. The interval is approximately `2026-11-09 05:49 UTC` to `2026-11-09 13:49 UTC`. Retain the discussion/experiment split and follow the [current work package](light-sampling-clustering-repair-prompt.md#current-task-continuous-eight-hour-exploration-2026-11-08). This supersedes the earlier stage restrictions of no lookahead/K before clustering repair and offline diagnosis only. World stability, full support, and actual PDFs remain required.

The hard performance cap is actual raygen-pass time no more than `1.25` times the frozen Camera baseline in each validated representative scene. Initially time the complete [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl), including selection, within-emitter sampling, and true visibility rays; report full construction/preparation separately. Efficiency uses `V*T`, not `V/T`, since independent-sample equal-time mean variance scales with `V*T/B`. Combine each regional/tail V with measured time separately instead of inventing a combined score. Continue prioritizing difficult regions/tails while retaining mean and zero-observation constraints; allow a moderate mean cost for a substantial tail benefit.

First compare existing upper/flux-diagonal, lookahead, and K candidates on matched quality and time, then iterate from actual counterexamples. Keep Camera, World Prefix, and Cosine BRDF controls. A few fixed points or older finite windows cannot establish every scene. The user subsequently requested less granular delegation and fewer messages: return reviewable rounds, with autonomous experiment execution and batch review; the work package defines ownership. Before the deadline, consolidate the best validated candidate and remaining scope rather than stopping exploration at one failure. Current state and originals are linked from the [experiment record](../../../experiments/light-sampling/20261108-eight-hour-exploration.json).

#### Initial screening and independent review (2026-11-09 06:20 UTC)

The first round uses existing 1080p presets, two independent 32-frame windows, and 128-frame complete sampling-pass timing; it is not paired performance acceptance. In ls1, Camera, Prefix, frozen upper, and lookahead have identical camera, lights, and complete world voxels. Exact matching of the geometry normal, material, and depth actually read by the sampler retains 2,017,704 pixels. Changing lightmap fields do not affect this estimator; matching every packed G-buffer bit would incorrectly discard half the image. Retain the original hashes. This first batch lacks receiver-color snapshots and remains screening evidence; subsequent jobs include `colortex9`. The discussion-side review directly recomputed original counts, RGB variance, quantiles, and timings, matched the report, and inspected full images and distribution curves.

| ls1 method | Raygen median / mean ms | Worst-5%-pixel mean variance in two windows |
|---|---:|---:|
| Camera + A + Near/Far | 8.540 / 8.477 | 0.33059 / 0.33000 |
| Prefix + A + Near/Far | 9.091 / 9.214 | 0.44500 / 0.45185 |
| Frozen upper + A + Near/Far | 8.389 / 8.376 | 0.38815 / 0.38850 |
| Prefix + one-level lookahead + Near/Far | 10.791 / 10.877 | 0.23245 / 0.23502 |

Lookahead mainly improves the distribution tail; ordinary quantiles do not uniformly beat Camera. Its initial median/mean time ratios are `1.264`/`1.283`, so it has not met the 25% cap. Retain both timing statistics for `V*T`; fixed-budget theory uses expected time. An auxiliary 64-sample merge accounting for sample counts and differences between window means preserves the ordering, but 32-frame sample variance is not error against truth. See the [initial comparison](../../../builds/light-sampling/20261108-eight-hour-exploration/light-sampling-1-camera-prefix-upper-lookahead.json), [independent review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-ls1-first-review.json), and [full-frame preview](../../../builds/light-sampling/20261108-eight-hour-exploration/review-ls1-first-grid.png). Upper's construction timing still covers only loading a frozen atlas.

Moving-4 provides a counterexample: 2,022,174 common pixels pass receiver color, depth, geometry-normal, light, and occluder checks. Ordinary/distance-gated lookahead takes `1.563`/`1.450` times Camera's median time, with worst-5% variance around `0.0246`/`0.0249`, versus Camera's `0.0173`. Neither passes this scene's speed or quality screen; ls1 cannot establish general improvement. See the [initial moving-4 report](../../../builds/light-sampling/20261108-eight-hour-exploration/light-sampling-failure-moving-4-camera-prefix-near-lookahead-lookahead.json).

Next distinguish equivalent acceleration from changes to the distribution. First reject wholly back-facing parent clusters before lookahead and compact empty outer children during construction; both require preserving support and nonempty child order. A separate two-level cut expands eligible cross-brick children, normalizes Near/Far across up to 16 candidates, and directly enters the selected node, skipping the expanded level. It is not an equivalent rewrite of existing lookahead: actual PDFs must follow the new probabilities. Selecting an aggregate group and then recomputing its at-most-four children can avoid long-lived arrays. Offline screening over 28 existing per-light M2 windows checks probability conservation and observed positive support, but does not dominate full lookahead; fewer expected node reads are not GPU evidence. Combining upper with lookahead shows complementary behavior at moving points while potentially regressing the ls5 wall. These results select runtime candidates without changing the baseline.

#### Second screen, mean identity, and support limits (2026-11-09 07:00 UTC)

The discussion-side review read complete originals and independently recomputed common pixels, two-window RGB variance, quantiles, tails, and 128-frame timings for the new ls5 and moving-4 candidates, matching the experiment reports. A separate reconstruction from camera matrices and connected components retains 7,449/1,554/65,773 pixels on the two moving side planes/upward plane; the [region overlay](../../../builds/light-sampling/20261108-eight-hour-exploration/review-moving4-plane-regions.png) was inspected. These complete connected surfaces replace tiny pixel patches as the main regional diagnosis here, but do not represent every similarly oriented surface in the scene.

| Screen | Mean raygen / Camera | Worst-5% variance times mean time / Camera |
|---|---:|---:|
| ls5 global area SAH | 0.824 | About 0.85 |
| ls5 global area SAH + face mixture | 0.826 | About 0.81 |
| moving-4 global area SAH | 0.876 | About 1.64 |
| moving-4 global flux-diagonal | 0.861 | About 0.74 |
| moving-4 Prefix, upper Near/Far 75/25 | 1.010 | About 0.85 |

Global trees still load from offline atlases and omit actual dynamic construction. Initial mean timings do not replace reverse-order paired acceptance. Two separate ls5 SAH timing pairs give ratios around 0.837/0.840, while both blue-lit floor regions retain costs. The moving 75/25 candidate changes only the mixture of the two already normalized branch distributions at cross-brick parents, retaining inner-brick 50/50, Half-space A, lights, and actual PDFs; its two side-plane tail efficiency costs are about 0.83/0.54 times Camera. No new default is accepted. See the [ls5 comparison](../../../builds/light-sampling/20261108-eight-hour-exploration/light-sampling-5-camera-prefix-global-area-sah-face-mixture-global-face.json), [moving comparison](../../../builds/light-sampling/20261108-eight-hour-exploration/light-sampling-failure-moving-4-surfaces-camera-prefix-global-area-sah-global-diagonal-outer-near.json), and [independent regional review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-moving4-plane-review.json).

**Mean identity has an unexplained discrepancy.** In both moving windows, Camera/area SAH/75:25 have whole-image blue means around 0.000603, versus 0.000581 for Prefix/global diagonal; the upward connected plane gives about 0.004052 versus 0.003897. This repeated difference exceeds variation between each method's two windows and cannot simply be called ordinary random noise. Receiver color/geometry, lights, and voxels match, but this group did not save emissive face-UV tables; source for the two global trees differs only in frozen node data. The next within-job comparison must verify face UVs and actual emission inputs, distinguishing drift, finite-window missed contributions, and estimator defects. Existing statistics remain screening clues, not empirical proof of preserved brightness or unbiasedness.

**Do not remove every lamp face touching a solid neighbor.** The current endpoint acceptance checks material and distance below 0.01 without checking target voxel identity. Existing M2 includes positive contributions from lamps whose candidate visible faces this rule would all exclude. Independent CPU geometric replay also finds finite interior face positions that first hit an adjacent emitter with the same material about 0.00664 from the target; see the [counterexample](../../../builds/light-sampling/20261108-eight-hour-exploration/neighbor-hit-witness.json). This is not GPU sample attribution, but disproves the premise that all such directions must contribute zero. Keep endpoint semantics unchanged this round. The experimental alternative mixes 25% original faces with 75% reweighted faces, retains same-material adjacent faces and all original support, and evaluates the actual marginal PDF. Of 64 GPU records, 44 triggered reweighting; maximum relative conditional-PDF discrepancy was about 1.72e-4. Whole-image and cross-scene acceptance remains necessary.

A new inexpensive weight screen uses only the existing AABB: projected radius, uniform positive-half-space mean/RMS, and the enclosing sphere cone's exact maximum cosine. The first two families regress the moving top/some sides. The sphere-cone candidate is more balanced on existing M2 without new node reads/statistics, but has no measured cost or full-frame benefit yet. For a unit receiver normal, let `h=dot(n,delta)`, `d²=dot(delta,delta)`, and `r²=dot(half,half)`. Use 1 inside the sphere. Otherwise let `s=sqrt(d²-r²)`; use 1 when `h>=s`, else `(h*s + sqrt(max(d²-h²,0))*sqrt(r²))/d²`, retaining the existing cosine floor and half-space rejection. The [offline results](../../../builds/light-sampling/20261108-eight-hour-exploration/offline-cosine-moment-screen.json) select candidates for the next batch without establishing full-frame generalization.

#### Fixed emission inputs and live coarse-tree review (2026-11-09 07:50 UTC)

The old blue-channel mean discrepancy disappears in a later within-job Camera/global-diagonal comparison, while captures also demonstrate changes to the emissive face-UV table. The measurement fixture now reads the same fixed UV table, retaining actual textures, emission evaluation, and endpoint acceptance. In two separate moving-4 pairs, global diagonal/upper Near/Far 75:25 have tail `V*T` around `0.746`/`0.887` of their respective Camera controls. Direct discussion-side recomputation matches both original data and timings; see the [diagonal review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-moving4-fixed-uv-diagonal-first-review.json) and [75:25 review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-moving4-fixed-uv-near-first-review.json). This supports a surviving benefit without attributing the old discrepancy entirely to UVs. Captured lava/fire atlas crops match at the endpoints within each job, but color-atlas crops still change between jobs. All 35 emissive materials are not frozen, so cross-job means cannot be combined as fully matched emission inputs. The fixture's extra UV lookup also differs from final production timing.

Live coarse candidate `11cb4137` retains Prefix subtrees inside 64³ world-block cells and rebuilds the G16 four-way upper tree from at most 125 roots, using flux times squared AABB diagonal as split cost. An independent direct binary reader confirms all 10,235/2,172/4,243 emitters in ls1/ls5/moving-4. Within 42/20/54 coarse roots, the seven bounds/mass fields, material, and ordered nonempty children match exactly; physical child addresses and empty-slot placement need not match. Before/after semantic trees agree, with maximum internal relative mass error about `1.15e-7`; see the [independent structural review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-coarse-structure.json). CPU float64 and GPU topologies differ in two scenes, but float32 replay preserving operation order with left-side FMA matches exactly. That difference is not evidence of missing lights.

| Live coarse screen | Mean raygen / Camera | Whole-image tail V*T / Camera | Main limitation |
|---|---:|---:|---|
| ls1 | About 0.927 | About 1.58 | Substantial ceiling cost remains |
| ls5 | 0.865 | 0.682 | Left sandstone-wall tail variance rises from about 0.00407 to 0.152; the blue pillar improves while the two blue floors differ |
| moving-4 | 0.987 | 0.729 | Camera is a cross-job screening reference; emission identity and low-frequency noise retain unverified scope |

Whole-image original statistics were independently recomputed for all three scenes. The within-job moving Prefix/coarse raygen means are 4.658/4.116 ms, but the sum of construction-pass median times rises from about 0.498 to 1.170 ms, with the final stage increasing from 0.117 to 0.789 ms. Few roots do not imply cheap construction. The next version reuses 284 existing upper slots for at most 248 child records and computes the three axis costs concurrently with fixed accumulation order. Source review finds no capacity or barrier-divergence problem; runtime and boundary validation remain pending. No universal default has passed acceptance.

Retain a separate spatial-scale check for broad brightness noise: average the difference between two 32-frame linear means inside commonly covered 16²/64² regions, then take half its squared norm. It can expose low-frequency changes absent from per-pixel variance, but one window pair cannot establish correlation or input-drift causes. Under an independent-pixel approximation its expectation is `sum(V1+V2)/(64*N²)`, where N counts valid block pixels; this approximation is not the actual correlation model.

The six-direction neighbor summary remains a soft-weight candidate: mix six exposed-direction masses according to the node-to-receiver direction, retaining a 25% original-mass floor without removing lights or faces. In the [offline quantization screen](../../../builds/light-sampling/20261108-eight-hour-exploration/offline-escape-quantization-screen.json), recursive UNORM8 retains improvements across 28 existing M2 windows, with at most about 0.114% higher variance than the unquantized summary, fitting a separate 8B record. This establishes neither whole-image benefit nor GPU cost. Another [construction-cost-only screen](../../../builds/light-sampling/20261108-eight-hour-exploration/offline-exposure-clustering-screen.json) improves the ceiling while worsening the initially quieter new red pillar. It remains eligible for whole-image evaluation instead of being rejected on one relative point regression.

#### Real brick crossings, coarse build costs, and cube numerical failures (2026-11-09 08:40 UTC)

Parallel coarse binning preserves the serial candidate's ordered topology. In moving-4, the final build stage's median falls from about 0.789 to 0.396 ms, and the sum of build-pass medians from about 1.170 to 0.778 ms. Cheaper candidate `15df1311` retains subtrees inside world 64³ cells and anchors the coarse Morton phase to the minimum world coordinate of active coarse roots. Its corresponding full-build sum is about 0.484 ms versus 0.499 ms for paired Prefix. A sum of medians is not the median of total time. This is also not a prefix anchored to a fixed world origin: identical light inputs preserve grouping, but adding/removing lights on the minimum boundary can change the anchor.

Direct discussion-side recomputation of both raw moving-4 windows gives anchor mean raygen time about 4.164 ms and whole-image worst-5% variance times mean time about 0.79 of the old Camera control; parallel binning is about 0.75. See the [independent anchor review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-moving4-anchor-first-review.json) and [parallel review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-moving4-parallel-first-review.json). These are single-scene screens on common receiver pixels with Camera from a different job, insufficient for a general default.

Actual continuous motion establishes grouping stability on this trajectory. Both jobs have identical sets of 28 actual poses; exhaustive alignment of raw GPU traces finds a unique 26-frame continuous interval in each. Offsets 8 and 24 lie on opposite sides of x=112 and share two commutative fingerprints for 4,042 lights. Camera changes its ordered tree fingerprint and node count, while Prefix, coarse binning, and anchor remain stable. The two jobs' Prefix trace files even have identical hashes; see the [independent crossing review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-anchor-motion.json). These finite fingerprints omit internal bounds/mass and are not a cryptographic proof of complete input identity. Other poses change their light sets. The diagnostic trajectory is not performance or temporal-noise acceptance.

The candidate that combines the visible cube silhouette into one projected polygon has not passed its numerical audit. Runtime witnesses contain 52 invalid directions/misses, including two far outside the cube. Independent reconstruction and bounded double-precision inversion of the same sector CDF hits the cube in all 52 cases; the [replay](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-silhouette-witness.json) supports a numerical inverse-sampling problem rather than invalid silhouette geometry. In [CPU float32 replay](../../../builds/light-sampling/20261108-eight-hour-exploration/silhouette-f32-sectors.json), one last sector grows from about 4.75e-14 to 1.17e-10, making a random number close to one select the wrong sector. This is not GPU bit-exact replay and does not attribute every failure to one mechanism. The witness job failed final scene restoration; its originals are diagnostic only and excluded from quality/performance rankings.

Far-field candidate `480550f` chooses from geometry to sample uniformly on an axis-plane projection of the cube silhouette, using the actual area-to-solid-angle Jacobian for its PDF. Near and grazing cases use the original path. Independent construction of the hull of all eight corners and integration on the actual visible cube faces covers 1,207 eligible inputs in 26 regions, with maximum normalization error about 1.33e-12; see the [independent face-domain integral](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-far-plane-integral.json). This establishes double-precision geometric normalization only. Actual GPU support, textured/occluded variance, speed, and the original path's own failure rate remain runtime checks.

#### Sandstone-wall attribution, stratification, and MIS limits (2026-11-09 09:37 UTC)

Anchor's moving-4 benefit does not generalize to all scenes. In matched ls5 pairs, mean raygen time is about `0.956` of Camera, but both windows' whole-image worst-5% variance times mean time are about `1.26`; left-wall absolute tail variance rises from about `0.004` to `0.60`, while the blue pillar improves. The corresponding ls3 tail-efficiency cost is about `1.61–1.65`. These raw statistics were independently recomputed; see [paired ls5](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-ls5-anchor-first-review.json) and [ls3](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-light-sampling-3-anchor-first-review.json). The independent G32 moving-4 comparison establishes only a Prefix comparison: mean raygen ratio about `0.863`, tail-efficiency ratio about `0.26`. Its retained cells are `256×128×128` world blocks, not the same 64³ coarse level as G16. Actual projection drift invalidates the Camera pair's input acceptance.

The valid mode-3 subset at new ls5 wall display pixel `(100,295)` contains 2,172 lights and 15,264 samples per light per window, complete in both windows. Exclude the second receiver selected on an emitter. Discussion independently re-extracted conditional moments from raw mean/variance/completeness buffers, parsed the native 32B trees, and recomputed PMFs, matching the experiment result. Maximum relative second-moment identity error is about `4.87e-7`. Camera point variance is `0.0004932/0.0004966`; anchor 75/25 is `0.017578/0.017734`. These point values explain an already observed surface regression rather than replace whole-image tradeoffs. See the [independent raw/path review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-ls5-wall.json).

At the critical split, sibling bounds `(305,10,260)–(363,27,319)` contain the receiver and take the entire Near component. The important-light group `(256,3,192)–(376,33,256)` has nearest squared distance about `80.95`, zero Near share, and only `5.76%` conditional probability from Far. Final probabilities of the three lamps `(307,12..14,255,4440)` fall from Camera's approximately `12.3–13.1%` each to `0.342–0.363%`. Returning 75/25 to 50/50 only reduces the point-variance ratio to about `22.8` versus Camera; Prefix and coarse-SAH remain about `12.3/14.4`. This is the same containing-coarse-AABB mechanism as the earlier Near/Far regression, not evidence for a normal-direction-specific correction. Previous all-level bounded near/far already has moving counterexamples, which subsequent repairs must also check.

A separate study recycles a tree-CDF draw and stratifies consecutive 32-frame windows. Its theory evaluates block-mean variance with independent jitter in 32 strata; correlated samples cannot be accepted using per-frame sample variance divided by 32. The [exhaustive review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-cdf-grid.json) directly vector-evaluates all `2^23` midpoint draws and matches the experiment-side binary-search interval counts. Across the existing 28 windows and two new wall windows, predicted variance reduction is about `4.6–51.6%`, median `26.0%`; the wall gains only about `5.8%`. For these finite moments and CPU float32 proposals, maximum relative RGB norm bias is about `7.63e-6`, and the maximum 32-frame mean MSE ratio to the ideal stratified model is about `1.0000143`. This does not prove actual GPU weights/PCG frequencies or fix coarse probability underallocation. Independent runtime frame blocks and time measurements remain necessary. The [stratification reference](https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency) supplies the variance decomposition.

The [single-sample BRDF/light MIS headroom screen](../../../builds/light-sampling/20261108-eight-hour-exploration/offline-brdf-mis-headroom.json) also does not yield a universal repair for these points. The density ratio `p_BRDF/p_cube` is constant on ordinary projected faces; geometric extrema bound tiny projected-area fallback faces. Applying the mixture density to fixed M2, 50/50 raises anchor's new red-pillar point variance to about `1.30–1.31`, lowers the ceiling to about `0.81`, raises moving points mostly to `1.79–2.60`, and raises the sandstone wall to about `1.16`. It may still suit other nearby-light/sparse scenes, but complementarity alone does not justify a new default. This screen assumes both proposals evaluate the same target integral, which remains unverified with the current same-material endpoint tolerance. Reverse PMF lookup and actual cost are unimplemented; the [PBRT interface](https://pbr-book.org/4ed/Light_Sources/Light_Sampling) explains why reverse PMF evaluation is needed for actual MIS.

#### Selective refinement and limits of cheaper substitutes (2026-11-09 10:06 UTC)

The selective-containment candidate `e3cb50a1f` expands only the Near endpoint of a cross-brick box that contains the receiver and has less than one eighth of its projected span in front. It keeps the native Far endpoint and actual mixed PDF. This is a geometry heuristic with one extra level, not proof that the box is empty or invisible. Native ls5 screening after the projection fix has constant projection scales `0.5625/1.0`; the older scales were `0.48042044/0.85408080`. Compare methods within the new lens cohort and reproject the old physical regions instead of retaining their original screen coordinates.

The independent [whole-frame recomputation](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-ls5-anchor-near-gate-camera-fixed-fov-first-review.json) and [inverse-map region review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-fixed-fov-regions.json) match the raw statistics. Against the matched Camera control, candidate mean raygen is `4.333568/4.284984 = 1.0113`. Whole-image worst-5% V changes from `0.34686/0.34763` to `0.32935/0.33149`; P99 rises from `0.26873/0.26784` to `0.36910/0.37201`. The reprojected wall retains 5,644 pixels and tail V rises from `0.00489/0.00472` to `0.02792/0.02896`; the below-pillar floor also regresses, while the blue pillar and right floor improve. Relative to unchanged anchor, wall tail V falls from `0.6433/0.6951` to `0.0277/0.0285`, with essentially unchanged mean raygen. The [region overlay](../../../builds/light-sampling/20261108-eight-hour-exploration/review-ls5-fixed-fov-regions.png) was inspected.

**These are still diagnostic screens.** Emitting UV hashes differ, and this batch lacks frozen-UV and atlas-probe identity. Recomputed arithmetic and matching geometry do not establish stationary emission. Do not turn the promising regional mechanism or the 4% whole-tail efficiency difference into quality acceptance. The candidate also needs cross-scene and paired mean-time confirmation.

The [timing-flag review](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-timing-cap-review.json) finds that historical `raw_timing_cap` flags use medians. For example, ls1 fast-cull lookahead has median ratio `1.049` but mean ratio `1.636`; its flag cannot satisfy the stated cap. The [original timeline](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-lookcull-timing-diagnostic.json) also contains spikes in other passes. Preserve them, avoid an unsupported causal explanation or outlier deletion, and use qualified paired mean timings for final decisions.

Two additional fixed-point studies do not merit immediate GPU expansion. [Mixing two stable upper trees](../../../builds/light-sampling/20261108-eight-hour-exploration/offline-stable-tree-mixture.json), with one predetermined +1-brick world phase and the true marginal brick PMF, repairs the wall but leaves regressions and adds reverse-path reads; anchor/phase-75 has median variance ratio `0.920`, worst `1.596`, and median logical read ratio `1.816` to anchor. These are nonempty logical reads, not GPU costs. [SLC-2019 distance-gate adaptations](../../../builds/light-sampling/20261108-eight-hour-exploration/offline-sibling-distance-gate.json) perform worse across the 30 point-windows: applying the paper's diagonal-distance test to all four nonzero siblings and disabling upper attenuation yields median ratios about `41.2` to Prefix and `101.4` to anchor. Even replacing only the Near arm is poor. This extends the [original two-child rule](https://www.cemyuksel.com/research/stochasticlightcuts/stochasticlightcuts_techreport.pdf) to our reflectance bound and topology; it is not a reproduction of the paper. Removing distance selectivity wholesale is unsuitable for these captured conditions.

A separate [exact rational slab check](../../../builds/light-sampling/20261108-eight-hour-exploration/independent-far-boundary-witness.json) finds that two of three far-plane failure rays intersect the selected cube, while the third stored float ray misses by about `9.57e-8` world units. Allowing a tiny slab gap alone therefore does not prove a valid hit. Candidate `cc1faea0b` uses exact selected-voxel identity for newly admitted boundary cases; keep that distinction from the older same-material/distance acceptance. This review validates the counterexample and source rationale, not whole-image benefit or cost.

### Evidence and rationale from clustering repair (2026-11-08)

#### Acceptance criterion revision: absolute noise and whole-image tradeoffs (2026-11-08)

The user identified excessive reliance on relative variance at fixed points, omitting baseline noise, affected area, and benefits in difficult regions. Withdraw the requirement that both windows at all specified points be no worse than Prefix as a hard rejection or whole-image stop. Historical failure of all 14 candidates describes the former gate, not overall usability. Retain upper/flux-diagonal for whole-image evaluation without changing the baseline. Fixed points diagnose mechanisms; acceptance considers matched whole-image/regional absolute noise, tails/rare contributions, visible changes, and cost tradeoffs. Neither win counts nor one relative percentage suffice. Support, PDFs, and input identity still require independent correctness.

In the first window of the same new ls1 cohort, upper/flux-diagonal changes red-pillar RGB variance from `0.001295526` to `0.001368780`, an absolute increase of about `0.000073254`. The central ceiling changes from `0.015105597` to `0.012207626`, decreasing by about `0.002897971`. Second-window changes are approximately `0.000071955`/`−0.002906761`. The ceiling reduction is about 40 times the red increase, demonstrating why +5.6% at an already quieter point cannot alone decide screening. These two points cannot simply be summed with equal weight to represent the whole image either.

The new [whole-image and absolute-noise viewer](../../../builds/light-sampling/20261108-whole-image-review/comparison.html) opens the full frame and reuses 14 original 1-spp/32-frame images: camera/Prefix/Cosine BRDF in ls1, plus the earlier global area SAH in ls5. Fourteen shared-scale noise/difference maps bring the total to 28. Zoom, switching, and arbitrary-position 16×16 absolute-variance readouts are available. All trees use A + Near/Far. **Current upper, flat, and ls1 global SAH have no whole-image capture.** Missing combinations are explicit; the earlier SAH image does not represent upper.

Absolute statistics use common complete receiver pixels across every method within each scene, without brightness filtering. ls1/ls5 retain 2,022,469/2,066,954 pixels, excluding 5/2 incomplete/nonfinite receiver pixels. The table gives pixel-mean summed linear-sRGB channel variance, and the pixel-weighted mean inside each method's own highest-variance 5% of nonempty 16×16 tiles:

| Scene | Method | Whole-image absolute variance | Highest 5% tiles |
|---|---|---:|---:|
| ls1 | Camera | 0.02109749 | 0.22121244 |
| ls1 | Prefix | 0.02812204 | 0.31099373 |
| ls1 | Cosine BRDF | 0.03935055 | 0.34827952 |
| ls5 | Camera | 0.01602190 | 0.24930938 |
| ls5 | Prefix | 0.02406768 | 0.33401366 |
| ls5 | Global area SAH | 0.01699366 | 0.25673273 |
| ls5 | Cosine BRDF | 0.07684409 | 0.67877163 |

For ls5 SAH versus Prefix, the whole-image absolute change is `−0.007074019`. Averaging within 16×16 tiles before separating signs gives about `+0.000304122` increase and `−0.007378141` reduction budgets. Their sum reproduces the net change and quantifies local costs against wider benefits in this window; it does not establish statistical significance. Each method selects a different highest-5% tile set, describing its noise tail. Thirty-two-frame zero means and bright samples can miss rare contributions; variance is not error against a high-sample reference.

Noise maps show the square root of tile-mean RGB variance; difference maps subtract Prefix tile variance without dividing by it. Both scenes/all methods share logarithmic/symmetric-log color scales, with saturation fractions reported and excluded pixels gray. Original ls1 images use voxel cohort `023bcd36…`, distinct from new-point M2 cohort `4dfca2b6…`; keep them separate. Sources and metrics are in the [compact record](../../../experiments/light-sampling/20261108-whole-image-review.json), reproduced by the [generator](../../../scripts/light_sampling_whole_image_review.py). Every pixel in all 14 original images matches the display transform and `raw[1079-display_y]`; all 28 PNG/AVIF pairs match. Full-frame previews and difference maps were inspected directly and JavaScript syntax passed. Work settings disable local browser use, so interaction was not tested. No GPU capture or shader modification occurred.

#### Spatial noise distributions and blue-lit region feedback (2026-11-08)

The user clarified that this local comparison is **Global area Light-SAH versus Camera grouping**, and selected the quality priority: keep the overall average reasonable, but prioritize difficult cases and the noise tail. A candidate with a slightly worse average can remain worthwhile if its tail improves substantially. Do not introduce an unselected weighted score or a zero-regression gate at every point. Report matched absolute spatial distributions separately by scene so easier scenes cannot hide difficult ones.

For each pixel, define `V=Var(R)+Var(G)+Var(B)`. The main table reports V's P5/median/P95/P99, mean, spatial standard deviation, and worst-5%-pixel mean across all common complete receivers. The latter uses each method's own largest `ceil(0.05*N)` values. Spatial standard deviation measures variation of V across pixels; it is neither `sqrt(mean(V))` nor temporal brightness standard deviation. Prioritize P99 together with the worst-5% mean; use the mean for aggregate cost and the median for typical cases. P95 is only the threshold entering the worst 5%, not its severity. Median, P5, and standard deviation are not individual acceptance criteria either. Retain zero-observation rates and original images; do not remove missed contributions with a brightness filter.

The [updated viewer](../../../builds/light-sampling/20261108-whole-image-review/distribution-review.html) retains the original 28 entries and adds seven absolute-difference maps against Camera, totaling 35. The original page and captures remain unchanged. Tables switch between the full image/fixed regions and RGB/B-channel statistics, retaining Prefix and Cosine BRDF controls. See the [distribution plot](../../../builds/light-sampling/20261108-whole-image-review/ls5-variance-distribution.png) and [region map](../../../builds/light-sampling/20261108-whole-image-review/ls5-review-regions.png). B-channel statistics do not isolate contributions from blue emitters. Current upper still lacks a whole-image capture; the earlier SAH is not a substitute.

The matched ls5 input and common pixel mask give the following results. This tail sorts pixels directly and **differs from the earlier table that averages 16×16 tiles before sorting**:

| Absolute metric | Camera | Global area SAH |
|---|---:|---:|
| Mean V | 0.01602190 | 0.01699366 |
| P5 | 0 | 0 |
| Median V | 0.0000009417 | 0.0000003064 |
| P95 | 0.04444344 | 0.05122942 |
| P99 | 0.20201320 | 0.22814517 |
| Worst-5%-pixel mean | 0.26961108 | 0.28166369 |
| Spatial standard deviation of V | 0.17431567 | 0.17862437 |
| Zero 32-frame RGB mean | 42.74% | 44.95% |

Four interior polygons were selected from the user's description before inspecting their statistics. Each has a single geometric-normal encoding among its valid receivers. They diagnose local behavior rather than exhaustively accepting an entire surface or restoring point vetoes:

| Region | Complete receiver pixels | Worst-5% mean: Camera → SAH | Zero mean: Camera → SAH |
|---|---:|---:|---:|
| A: left sandstone wall | 18,955 | 0.00397086 → 0.02244277 | 0.29% → 21.68% |
| B: blue-lit pillar | 3,170 | 0.02954616 → 0.00177294 | 40.91% → 0.38% |
| C: blue floor below pillar | 3,217 | 0.00043922 → 0.00592599 | 38.64% → 88.44% |
| D: blue floor in right shadow | 9,025 | 0.00167924 → 0.01125571 | 12.18% → 57.46% |

C's P95 actually decreases from `0.000220860` to `0.000062741`, while P99, the worst-5% mean, and zero observations worsen. This directly demonstrates how a single percentile can miss sparse bright events. The four regions support the user's observed pattern. In this window, global area SAH also has a slightly worse whole-image tail than Camera, so it does not yet demonstrate a slightly worse average buying a better overall tail. This neither rejects all world-stable clustering nor dismisses the pillar improvement.

[LightSampling](../../../shaders/techniques/debug/LightSampling.glsl) estimates weights from receiver position/normal and each sibling AABB/mass, then multiplies branch probabilities. Regrouping can therefore affect pillar and floor differently. The pattern is consistent with receiver-dependent probability mismatch, but no new per-light/ancestor-branch evidence was collected here, so it does not uniquely identify visibility, emission color, or Near/Far as the cause. Retain upper for whole-image evaluation; lookahead, K, probes, and the baseline remain unchanged. Do not continue blind clustering-parameter scans.

The [generator](../../../scripts/light_sampling_distribution_review.py) and [compact record](../../../experiments/light-sampling/20261108-noise-distribution-review.json) retain definitions, region coordinates, raw-data hashes, and all metrics. These remain existing finite 32-frame observations, not high-sample truth or continuous-motion acceptance; no GPU capture was added. All 35 lossless pixel pairs, the unchanged original-page hash, and script syntax passed checks. Distribution and region figures were inspected. Local browser access remains disabled in Work settings, so interaction was not tested.

#### Original clustering experiment: starting point and measurements

After inspecting the latest runtime comparison, the user prioritizes clustering over lookahead/moment summaries. Under matched inputs, World Prefix + A + Near/Far has **1.333 times** the whole-image RGB variance of original camera grouping + A + Near/Far in ls1 and **1.502 times** in ls5, while fixed ls1 pillar/wall ROIs improve to **0.567/0.096 times**. Grouping changes alone can therefore redistribute improvements and regressions; a few improved points cannot represent the image. The user additionally identifies improvements on some left red-pillar faces and a darkened yellow-lit ceiling near the upper middle. Locate these through actual receiver geometry rather than equating them with the old two points. See the [independently reviewed grouping record](../../../experiments/light-sampling/20261108-grouping-control.json).

The retained contract is: **for identical complete world emitter keys and flux, camera position, window addresses, and atomic allocation order do not determine cluster membership**. Fixed Morton-prefix topology is now replaceable. Flattening the current tree into a global Morton list and rebuilding the same prefixes may leave clustering unchanged; physical node reordering alone does not repair it. Measure regrouping under emitter-set changes separately; identical-input stability does not establish stability when emitters enter or leave.

The initial candidate globally partitions the actual emitter distribution into spatial bins, using existing flux and AABB surface area in a light-BVH split cost, then converts the result during construction to the existing four-way, 32B contiguous-child representation. The flux, area, and axis regularization terms follow [PBRT's light-BVH construction](https://pbr-book.org/4ed/Light_Sources/Light_Sampling); this experiment adds no direction cones and treats their bounds as uniformly omnidirectional. This is a simplified candidate, not a reproduction of published quality results. Input consists of individual emitters across bricks; bricks still own extraction. Ordinary geometric SAH is only an offline explanatory control, not an automatic GPU matrix. No H-PLOC parallel builder has been selected.

Clustering should make existing coarse estimates more useful, but benefit still depends on weights and receivers; smaller AABBs/SAH do not imply lower sampling variance. Screen with true conditional M2 and regression regions, then integrate a viable candidate into the live runtime and assess motion stability and complete build cost. Runtime clustering must not use receivers, measured M2, or visibility as an oracle or apply orientation-specific factors. One-level lookahead and K1/K2/K4 are excluded from this round.

Near/Far remains enabled for the clustering comparison, but a negligible cost has not been established. In the forward/reverse means from the [old camera-grouping timings](../../../experiments/light-sampling/20261031-nearfar-iteration.json), initial sampling changes from A to A + Near/Far as follows: ls1 **4.683 → 4.928 ms (+5.23%)**, ls2 **3.003 → 2.940 ms (−2.10%)**, ls4 **4.025 → 4.233 ms (+5.19%)**, ls5 **2.089 → 2.386 ms (+14.22%)**. Nodes remain 32B with no additional build summaries. Timings include light selection, conditional sampling, and true tracing, with only two orders; they neither isolate current World Prefix query cost nor establish significance for small differences. Add narrow current ls1/ls5 A versus A + Near/Far timing pairs. Previous local counterexamples remain valid.

The first [offline clustering screen](../../../experiments/light-sampling/20261108-clustering-repair.json) (`c429b44fb`) has not selected an acceptable repair. These are only two-window conditional variance results for the newer moving-4 interior points, relative to **World Prefix + A + Near/Far = 1**:

| Receiver | Flux light-SAH | Geometric SAH |
|---|---:|---:|
| side-a | 1.116 | 0.615 |
| side-b | 0.999 | 0.783–0.784 |
| top | 0.936–0.937 | 1.295–1.299 |

The older ls1 pillar/wall points give 0.578–0.582/0.470–0.480 for the flux candidate, but belong to an earlier voxel cohort and are not the newly reported regions. The 20 point-windows include repeated points and windows; improvement in 12 windows is not acceptance by majority vote. Both flux/count terms and axis regularization `Kr`/1 change between the SAH variants, so their difference cannot be attributed to flux alone.

The discussion-side [independent recomputation](../../../builds/light-sampling/20261108-clustering-repair/review/offline_review.py) reuses the candidate builder, checks CPU logical parent/child structure, AABBs, and per-emitter flux, independently evaluates scalar Near/Far PMFs, and recomputes 20 windows × 4 methods from the original NPZs. Its [result](../../../builds/light-sampling/20261108-clustering-repair/review/offline-review.json) has maximum relative variance difference 5.98e-16 and zero positive-M2 support failures. The reported side-a ancestor receives probability 0.672759; both the globally optimal leaf distribution's conditional share and the locally optimal share with that branch's interior proposal retained equal 1.0 here. These generally differ: the former normalizes descendant `sum(sqrt(M2_i))`, while the latter normalizes each child's `sqrt(sum(M2_i / p(i|child)))`. Keep the labels distinct. No PMF/moment-alignment error explaining these offline regressions was found. Dynamic GPU construction, whole-image benefits in newly reported regions, and new costs remain unverified. The experiment chat continues the new ls1/ls5 regions and narrow Near/Far timings without reopening K/lookahead work.

The user authorized arranging the next experiments. The [current clustering task prompt](light-sampling-clustering-repair-prompt.md) defines implementation, captures, return notification, and review. It supersedes further lookahead/K1/K2/K4 selection and mandatory prefix-parent preservation in older tasks without relabeling their results. The November 1 route and later summary discussions below remain historical evidence.

#### Red-pillar counterexample for Light-SAH and Near/Far cost

The new ls1 left red pillar and upper central ceiling each have two fixed-receiver M2 windows. Both retain support for every measured positive-M2 light. Light-SAH's conditional variance was **2.615–2.623×** World Prefix on the pillar and **0.738–0.739×** on the ceiling. The pillar diverged at the root. The branch containing the worst affected emitter received 0.690, versus 0.902 for the prefix branch containing the same light. Its leaf PMF fell from 0.00858 to 0.00287. Removing Kr, using counts, or adding Kr to counts still gave **1.927–2.125×** the pillar variance. The M2 voxel identity is `4dfca2b6…`. The preceding ls1 image comparison used `023bcd36…`.

The matched-input frozen prototype on ls5 passed native tree readback and PDF verification for its 2,172 lights. Whole-image and vertical-face 32-frame variances were **0.706× / 0.616×** World Prefix and **1.061× / 3.176×** camera grouping. The red-pillar counterexample triggered the stop condition. Frozen quality coverage is limited to ls5, with dynamic construction and live crossing pending. Offline readdressing of the same 3,617 lights across four moving camera windows yielded identical compact nodes and fixed-receiver PMFs.

| World Prefix, audit off, ABBA with 128 warmup / 128 measured frames | A initial | A + Near/Far initial | Full build A / A + Near/Far |
|---|---:|---:|---:|
| ls1 | 7.306 ms | 7.514 ms (+2.84%) | 0.327 / 0.327 ms |
| ls5 | 3.078 ms | 3.632 ms (+18.02%) | 0.192 / 0.192 ms |

Full voxel materials were checked at the enclosing frozen ABBA endpoints in both scenes. ls5 had matching endpoint depth, and ls1 timing did not capture depth. Initial includes light selection, cube sampling, and tracing. In matched ls5 windows, Near/Far's whole-image and vertical-face variances were 0.259× and 0.409× A. Its conditional variances at the two new ls1 receivers were about 0.111× and 0.329× A. The ls5 frozen tree took 0.255 s to bake on the CPU and 0.0109 ms per frame to check and copy from the loaded atlas into the SSBO. Iris texture upload during reload and dynamic GPU construction remain unmeasured.

The [ls5 eight-image comparison](../../../builds/light-sampling/20261108-clustering-repair/viewer/comparison.html) shows 1 spp and 32-frame means for four methods and marks the vertical face. The [experiment record](../../../experiments/light-sampling/20261108-clustering-repair.json) and [replayable M2 reference](../../../experiments/light-sampling/20261108-clustering-repair-reference.npz) retain input identities, the root diagnosis, timing, and review evidence.

Discussion-side independent review accepts these **diagnostic findings and scoped measurements**, rejecting promotion of global SAH as the repair baseline. The [point recomputation](../../../builds/light-sampling/20261108-clustering-repair/review/new-point-review.json) matches 8 windows × 7 methods within 3.12e-16 relative difference. [Runtime artifact/image review](../../../builds/light-sampling/20261108-clustering-repair/review/quality-review.json) reproduces ls5 common-complete-pixel statistics and all eight images; 64 actual selection PDFs differ by at most 5.35e-7 relatively. [ABBA recomputation](../../../builds/light-sampling/20261108-clustering-repair/review/timing-review.json) checks original samples, loaded sources/settings, and emitter/voxel identities. Five manifests and 282 files were verified. Browser interaction uses the experiment-side record; this review directly viewed images and independently checked every raw row's orientation and lossless pixels. Two nonfinite receiver pixels in ls5 frozen quality are excluded by the common complete mask across all four methods; their cause was neither hidden nor repaired. Existing material/depth and finite-window limitations remain.

Near/Far benefit is confirmed for these points and ls5 and remains enabled in clustering experiments. This does not establish no regressions anywhere or negligible cost. Ordinary area/flux SAH has a repeatable counterexample, which neither selects H-PLOC nor invalidates all clustering approaches. The original prefix/camera comparison changed only upper grouping, while the global candidate also rebuilt brick interiors. The [offline layer-isolation task](light-sampling-clustering-repair-prompt.md#offline-follow-up-after-review-2026-11-08) is now complete, as reported below. Indivisible bricks are only a control and do not narrow the final cross-brick scope. GPU expansion/dynamic construction remains stopped; lookahead and K work remain deferred.

#### Review of layer isolation and controlled composition

Experiment commit `c184404e1` compares four logical trees across 28 point-windows in five complete input cohorts. The discussion review independently extracts maximal single-brick subtrees from world leaf coordinates, checks preservation of original subtrees in upper-only changes and upper structure in interior-only changes, verifies identical emitter keys/flux, and recomputes all variances and red-pillar root-optimum diagnostics using scalar PMFs. Maximum relative difference is 4.01e-16. Interior reductions change flux by at most 1.38e-7 relatively; pinning interface flux changes variance ratios by no more than 2.77e-8.

The review adds a fifth controlled tree: retain the upper-only hierarchy's complete upper topology and node/brick-root flux, transplanting the interior-only brick subtrees. This composes both edits while retaining complete brick boundaries. The experiment author independently composes Nodes from the three original references: all node records, materials, and child indices match, with at most 4.45e-16 absolute difference across 28 point-window variance ratios. The [check script](../../../builds/light-sampling/20261108-clustering-repair/joint_node_check.py) and [result](../../../builds/light-sampling/20261108-clustering-repair/joint-node-check.json) are retained. Both sides reuse the same SAH builder; independent checks cover composition, contracts, and PMF recomputation. The table gives two-window conditional variance at the new ls1 points, with **World Prefix + A + Near/Far = 1**:

| Construction | Left red pillar | Central ceiling |
|---|---:|---:|
| Upper-only | 0.979–0.980 | 1.164–1.165 |
| Brick-interior-only | 1.014–1.015 | 1.013–1.014 |
| Both edits, complete brick boundaries retained | 1.0000–1.0003 | 1.190–1.191 |
| Global leaf-level SAH repartition | 2.615–2.623 | 0.738–0.739 |

The red-pillar nonadditive term in the controlled composition is only approximately 0.0058–0.0059 times baseline; the global candidate adds another 1.615–1.623 times baseline relative to that composition. This localizes the major extra penalty to **the overall difference introduced by this global construction breaking up existing small clusters and regrouping across bricks**. It does not establish that every cross-brick partition regresses. Retaining the global tree's descendant proposals and optimizing root branch probabilities alone still gives **2.390–2.398 times** baseline variance, removing only about **13.9%** of the excess; consequential partitioning/weighting effects remain below the root.

Upper-only changes give approximately 0.596/0.654/0.980 at the sampled moving-4 interior side-a/side-b/top and 0.838 at the ls5 vertical point, but regress the central ceiling. No construction is therefore selected for direct replacement. These receivers remain counterexamples for new clustering costs or partition constraints; candidate status and the world-stability contract remain unchanged. The [independent script](../../../builds/light-sampling/20261108-clustering-repair/review/hierarchy_review.py) and [output](../../../builds/light-sampling/20261108-clustering-repair/review/hierarchy-review.json) preserve results, the fifth controlled tree, and root diagnostics. Scope is finite-receiver offline analysis, with no new GPU work or default changes.


#### Further clustering-cost screening: objective and diagnosis

The user requests grouping that preserves the red pillar, ceiling, and moving side faces while keeping lookahead/K deferred. The current [offline task](light-sampling-clustering-repair-prompt.md#further-clustering-cost-screening-2026-11-08) fixes A + Near/Far and the original four-way conversion. Six rules cover flux/count times squared diagonal, spatial midpoint, count median, and flux median, each tested globally and over preserved brick subtrees. Receivers, normals, M2, and occlusion do not enter construction; no parameter, grid-phase, or test-orientation fitting. All specified interior points and work proxies constrain selection rather than average win count.

Sources and boundaries: [PBRT's](https://pbr-book.org/4ed/Light_Sources/Light_Sampling) area cost is a heuristic, not this sampler's per-light variance objective. [Lightcuts](https://www.cs.cornell.edu/~kb/projects/lightcuts/lightcuts.pdf) supplies an omnidirectional cluster-size metric of flux times squared box diagonal; this round borrows that metric for top-down partitions, not the original construction. [Fast Agglomerative Clustering](https://www.graphics.cornell.edu/~bjw/IRT08Agglomerative.pdf) motivates bottom-up alternatives that form different small clusters, but its Lightcuts/traversal results cannot predict current single-light sampling quality; H-PLOC remains unselected and unimplemented. [Learning to Cluster](https://www.csie.ntu.edu.tw/~cyy/publications/papers/Wang2021LCR.pdf) learns receiver-region cuts and probabilities rather than rebuilding a receiver-independent global hierarchy; it is not introduced here.

A discussion-side [per-node variance decomposition](../../../builds/light-sampling/20261108-clustering-repair/review/cost_diagnostics.py) and [result](../../../builds/light-sampling/20261108-clustering-repair/review/cost-diagnostics.json) use measured M2 only for attribution. With `S_v = sum_i(sqrt(M2_i))`, node reach probability `q_v`, and child conditional probability `p_c`, local loss is `(sum_c(S_c² / p_c) - S_v²) / q_v`. Summing all local losses exactly reconstructs actual variance minus globally optimal leaf variance. This decomposes the global leaf optimum and is distinct from optimizing the root with fixed descendant proposals.

For the new red pillar's first window, the original global SAH's largest losses occur at depths 4/3, contributing about 42.2%/17.2% of excess variance. At depth 3, a child's bounds contain the receiver and monopolize the near half; its mixed share is 0.58445 versus a measured-M2 optimum of 0.00693. At depth 4 another child with flux only 5 receives 0.22040 versus an optimum of 0.89631. Both empty space inside bounds and the difference between emitted flux and receiver contribution matter; whether diagonal cost suffices remains untested. For moving side-b under the upper-only control, the largest residual is already between two equal-flux leaves: actual probabilities are about half each, but only one has positive observed M2. Finite-window zeros do not establish permanent invisibility, and clustering alone is not promised to eliminate all variance.

#### New cost results and the next boundary

Six fixed top-down rules were tested globally and over preserved brick roots (`05c59e0a8`), followed by two exact bottom-up controls using the same flux-diagonal metric (`6c9d07a28`). **None of the 14 candidates preserves every required interior receiver.** Inputs and evaluation are unchanged, lookahead/K are untouched, and there is no new GPU experiment. The [experiment index](../../../experiments/light-sampling/20261108-clustering-repair.json) retains all windows and failures under `clustering_rule_screen` and `clustering_agglomerative`. Each table cell is the worst window at that named point, relative to Prefix + A + Near/Far:

| Construction | New red pillar | Central ceiling | moving side-a | moving side-b | moving top | ls5 vertical |
|---|---:|---:|---:|---:|---:|---:|
| Top-down upper, flux × squared diagonal, no Kr | 1.057 | 0.808 | 0.603 | 0.569 | 0.490 | 0.476 |
| Top-down global, same cost | 1.597 | 0.855 | 0.484 | 0.569 | 0.467 | 0.647 |
| Bottom-up upper, same cost | 2.243 | 0.451 | 3.746 | 2.987 | 3.573 | 0.457 |
| Bottom-up global, same cost | 1.338 | 0.800 | 28.021 | 4.231 | 13.870 | 1.881 |

Top-down upper flux-diagonal without Kr remains closest. Its red-pillar windows are 1.05654/1.05543. Expected nonempty child-record reads at the listed points are about 0.711–0.870 times Prefix and maximum depth decreases, but these are not GPU timings. Preserving complete bricks remains a control and cannot be presented as completed global cross-brick clustering. Count/flux median splitting also fails; balance does not substitute for quality.

The discussion-side [rule review](../../../builds/light-sampling/20261108-clustering-repair/review/rule_screen_review.py) independently enumerates 420 partitions and recomputes 420 variance/work entries using scalar PMFs, with at most 9.13e-16 relative variance difference; its [output](../../../builds/light-sampling/20261108-clustering-repair/review/rule-screen-review.json) retains every case. The [bottom-up review](../../../builds/light-sampling/20261108-clustering-repair/review/agglomerative_review.py) recomputes 112 entries within 8.75e-16; the [output](../../../builds/light-sampling/20261108-clustering-repair/review/agglomerative-review.json) also decomposes side-a. In that global tree, a depth-1 branch containing all positive observed M2 receives only about 0.06645 while another receives 0.65235; this node accounts for about 58.2% of excess variance. The author's 20 small subsets match naive globally minimum-pair construction, including 39 tied minima. Full construction preserves permutation determinism, light support, and primitive identities. Outputs specify reused constructors and independent checks.

A further [flat normalization diagnostic](../../../builds/light-sampling/20261108-clustering-repair/review/flat_reference_review.py) treats all leaves as a single sibling set and normalizes their A + Near/Far near/far terms to half each, removing coarse hierarchy estimates. It also changes the normalization domain and the levels at which Near/Far mixing occurs; it is not an ablation of box looseness alone. It is an O(N) offline reference, neither a runtime proposal nor the true optimal PDF. Red-pillar variance is still **1.24801/1.25139** times Prefix; the central ceiling is about 0.290, moving side-a/side-b/top about 0.170/0.434/0.403–0.411, and ls5 about 0.628. The experiment author's [independent scalar check](../../../builds/light-sampling/20261108-clustering-repair/review/flat_reference_independent.py) and [result](../../../builds/light-sampling/20261108-clustering-repair/review/flat-reference-independent.json) reproduce all 28 windows within 2.78e-16 absolute variance-ratio difference, without lost positive-M2 support or nulls. Three dominant red-pillar increment lights drop from 4.104%/2.280%/1.297% probability to 1.724%/1.279%/0.541%, explaining about 72%–73% of the net variance increase. Prefix happens to give these measured important lights more probability; approximating purely geometric flat leaf estimates is therefore insufficient to preserve this receiver.

The next **proposal, neither implemented nor dispatched for capture**, is a construction cost that measures distortion of the actual grouped proposal against representative receiver-region contributions while retaining spatial compactness. A small set of fixed world-space probes could provide temporary construction features containing occlusion/contribution differences. Current validation points/M2 must not train the candidate and then be reused to claim success, and probes must not change with the camera. [Matrix Row-Column Sampling](https://cseweb.ucsd.edu/~viscomp/classes/cse274/fa22/readings/a26-hasan.pdf) motivates clustering sampled light-transport responses, but its representative-light accumulation differs from the current single-light four-way sampler; its quality/cost results do not transfer directly. Before pursuing this route, separate occlusion, actual emission, and within-light distribution effects at the red pillar to identify minimal useful construction features. Evaluate with the current sampler rather than feature distance alone. Using occlusion or observed contributions would expand construction dependencies from the light set to lights, occlusion snapshots, and probe state. Stability must then be defined for that complete input, rather than promising identical trees for identical lights alone. Probe coverage, training/validation separation, update stability, and construction budget remain open. Lookahead/K and runtime defaults remain unchanged. This round accepts diagnostic findings only, not a repair baseline.

#### Current action: first decompose the red-pillar contribution mismatch (2026-11-08)

The user chose to distinguish occlusion, actual emission, and within-light sampling before adopting the probe proposal above, continuing lookahead/K, or adding another clustering candidate. See the [red-pillar factor-diagnosis prompt](light-sampling-clustering-repair-prompt.md#current-task-decompose-the-red-pillar-contribution-mismatch-2026-11-08). First reuse existing two-window M2, separating the light-selection mean term from within-light variation through `M2_i=||mu_i||²+s_i²`, then fill missing evidence for dominant lights and necessary controls. Within-light variation includes occlusion/texture changes; it does not imply a sampler implementation defect.

Necessary capture covers only the existing red-pillar and central-ceiling control. Pair actual/constant-mass emission with/without occlusion on the same direction/face samples, retaining each cell's own mean and M2. Unoccluded target emission must be evaluated independently for blocked samples rather than using blocker material or zero. Also check cube face selection, within-light PDF, and tiny-sliver fallback, distinguishing proxy mismatch, finite-sample variance, and implementation errors. Keep Prefix, upper/flux-diagonal, and flat light-selection distributions fixed. Do not mix old/new occluder snapshots or call counterfactual changes physical image improvements. This round authorizes diagnostics, minimal additional capture, and notification to this discussion for review; it does not accept a new baseline or feed receiver data into construction.

#### Red-pillar factor-diagnosis review (2026-11-08)

Experiment commit `f100765c7` reused the original two mode-3 windows and archived models without new GPU jobs. Results are in `red_pillar_factor_diagnosis` in the [experiment index](../../../experiments/light-sampling/20261108-clustering-repair.json), the [detailed report](../../../builds/light-sampling/20261108-clustering-repair/red-pillar-factors/result.json), and [per-light moments/selected original single-frame samples](../../../builds/light-sampling/20261108-clustering-repair/red-pillar-factors/per-lamp-moments.npz). Accept the offline diagnostic scope only. Complete physical-factor pairs and executed PDFs remain missing; no unique cause or clustering repair is established.

| Same red-pillar receiver, relative to Prefix | Two-window variance ratios | Decomposition of net increase |
|---|---|---|
| upper/flux-diagonal | 1.05654/1.05543 | Almost entirely the per-light mean term; within-light net change is about 0/−1.50% of the total increase |
| flat | 1.24801/1.25139 | Within-light variation accounts for about 63.38%/63.75% of the net increase |

Cross products of independent-window means reproduce this separation without clamping negative residuals. At the central-ceiling control, upper is about 0.808 and flat about 0.290, with both terms decreasing. Each receiver/light/window has 3,232 complete valid samples. Thus the upper-tree red-pillar failure persists after removing within-light stochastic variation, whereas flat amplifies the cost of lights with high conditional variance. That variation includes occlusion and texture changes; it does not directly establish a cube-sampler defect.

For occlusion, CPU projected quadrature over archived voxels/models gives red-pillar visibility around 36.6%, 18.1%, 93.0%, and 91.3% for the four dominant lights `(154,146,78,10254)`, `(150,146,78,10254)`, `(157,145,76,682)`, and `(146,153,88,682)`. The original GPU frame has 28, 13, 96, and 89 nonzero samples out of 101 respectively; nonzero rates are not themselves true visibility rates. The two zero-observed-M2 lights receiving the largest probability gains are blocked on every ray in both CPU quadrature orders. Conditional on the nominal projected proposal and CPU visibility, a variance identity grouped by the visibility event attributes about 71%–78% of the first two lights' own within-light variance to that event, versus about 21%–22% for the latter two. This is neither the causal benefit of removing occlusion nor a global noise share. A zero-width `visibility_interval` is not a confidence interval including quadrature error.

For actual emission, normalized nonzero RGB varies within a lamp with receiver/albedo fixed. For example, `(142,153,93,682)` is visible on all CPU quadrature rays and has 101/101 nonzero original-frame samples, with normalized B P95−P05 around 0.177. A scalar PDF and binary visibility cannot produce this chromaticity change; one mass cannot represent all material/texture emission variation. However, target texture emission in blocked directions was not saved, so full actual-emission, occlusion-selection, and interaction effects remain inseparable.

None of the four dominant lights triggers tiny-sliver fallback in the nominal geometry. With projected-integral face selection, `cosine/(pi*pOmega)` is ideally constant on ordinary faces. Controls also cover the area fallback's face probability and Jacobian. This checks formulas, archived sources, and CPU integration; executed direction/PDF pairs were not saved, so GPU numerical or sampling implementation defects are not ruled out. The 13 selected lights cover only about 51%–52% of measured upper net excess and 64%–65% of flat net excess. Neutral-mass blocked/unblocked differences apply only to this coverage and the altered integration target.

The discussion-side [independent reviewer](../../../builds/light-sampling/20261108-clustering-repair/review/red_pillar_factor_review.py) recomputes 12 variance/decomposition cases from the original compact reference with scalar summation, within `3.47e-18` maximum absolute error, and checks original RGB for 26 receiver/light combinations. Enumerating voxel AABBs intersected by open segments with independently coded quad intersection agrees with the author's DDA on 320 rays. Archived addressing/model data and previously reviewed PMFs are reused; the full visibility integral was not independently repeated. Input/source hashes and preservation of the previous index are verified. The [review output](../../../builds/light-sampling/20261108-clustering-repair/review/red-pillar-factor-review.json) has SHA-256 `b456ab278af2e1a6e7f1a891b659ce3cf9509528231f2a616f4c3e69e46e2925`.

The discussion-side read-only check confirms `1,064,130 B` remaining in Vibris' 3 GiB archive budget, versus `268,435,456 B` for the current complete voxel-material buffer. The current evidence protocol cannot accommodate the required capture; this does not mean the machine's disk is full. Active source is `a1818489-0837-496e-b0b9-89eb324d111b`, generation 53, with no reported queued/active jobs; no read-only scene/frozen-state receipt was obtained. This review performed no runtime mutation, restart, or deletion. The next operation remains the original two-receiver paired capture of blocked target emission and actual PDF, after satisfying artifact retention and input-identity requirements. Lookahead, K, probes, and new clustering repairs remain unchanged.

### Previous route and evidence (2026-11-01)

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

### Near/Far combinations and interior-region revision (2026-11-08)

After inspecting the [live viewer](../../../builds/light-sampling/20261108-cluster-iris/comparison.html), the user noted that old side-26/top-28 are too close to corners. In moving-4, one-level lookahead appears more consistent, K1 severely regresses on sides, and K4 looks worse than K2 on sides and grass. Keep old points for traceability and add two side-face interior receivers and one upward-face interior receiver with small regions. Freeze positions from baseline geometry before collecting conditional M2, avoiding conclusions about whole surfaces from corner-center pixels or a black short-window result.

This round rechecked every live atlas component and all 252 valid GPU selection-audit records across four candidates, with maximum PDF relative difference 8.70×10⁻⁷. Reconstruct current native addressing: 5,606 physical node addresses differ between the old conditional capture and new live capture, so old brick-local addresses cannot be reused. Current evidence found no GPU/CPU weight or PDF mismatch in these records.

Replaying actual K2/K4 membership shows that K4 is not a refinement of K2 at 57 of 344 cross-brick clusters. Node 70's K4-to-K2 membership intersections are `[[40,0],[1,47],[0,76],[0,13]]`. Its flux-weighted within-component variance sum falls from 119,539 to 53,735, yet the ball cosine of the component containing the critical nearby emitter reaches its floor. Better spatial fitting does not guarantee better contribution estimation. Non-nested grouping is another source of change; even enforcing nesting would not make this ball kernel additive across parent/children or guarantee monotonically decreasing sampling variance with K.

The user selected the complete paired matrix: World Prefix + Half-space A remains the baseline. Keep Near/Far off for World Prefix A, lookahead and K1/K2/K4, and add Near/Far on for each, retaining Cosine BRDF-only. Six scenes and 1-spp/32-frame choices total 132 images. Preserve reusable existing 72 images and add 60; recapture matched controls for scenes whose inputs drift. See the [combination experiment prompt](light-sampling-nearfar-combination-prompt.md) for formulas, interior diagnostics, prototype boundaries and completion callbacks.

The first combination accumulates each method's representatives' unnormalized near/far/zero-near scores, normalizes over current siblings and retains the original half-and-half mixture. Lookahead representatives are direct children. Ward representatives are existing components, with added actual member AABBs for distance bounds, retaining existing ball cosines and whole-cluster half-space tests. Near/Far off preserves current candidates; native Prefix on must reproduce model 6. Appending the old Near/Far branch after a new scalar weight would overwrite the correction, and variance balls cannot substitute for support bounds. Existing cosine underestimation and zero-near monopolization may remain; improvement from combining them is an experimental hypothesis. This discussion changes neither shaders nor defaults.

### Near/Far per-light experiment on face interiors (2026-11-08)

Using the same projection, voxels and 4,243-light tree in moving-4, Prefix A geometry fixed two side-face regions and one upward-facing 9×9 region. The new receivers are at the region centers, with the old three points retained for comparison. Two live conditional captures each contain two independent 32-frame windows, with 7,808 samples per receiver, emitter and window. Both captures include side-b. The calculation aligns M2 by world light key and uses each proposal's complete PMF to estimate single-sample RGB variance. All ten proposals retained support for every emitter with positive measured M2 in every window.

Values are first-window variance ratios to the fixed Prefix A baseline, shown as Near/Far off → on.

| Interior point | Prefix A | One-level lookahead | Ward K1 | Ward K2 | Ward K4 |
|---|---:|---:|---:|---:|---:|
| side-a | 1 → 0.090 | 0.392 → 0.068 | 93.07 → 0.126 | 0.130 → 0.057 | 0.244 → 0.088 |
| side-b | 1 → 0.097 | 0.332 → 0.083 | 1.026 → 0.093 | 0.195 → 0.086 | 0.145 → 0.080 |
| top | 1 → 0.429 | 0.567 → 0.430 | 3.074 → 0.529 | 1.529 → 0.655 | 9.688 → 0.692 |

Side-a reproduces the severe K1 regression. The ball cosine for node 153, which contains the critical nearby emitter, reaches the 0.001 floor. K1 gives the branch 0.254%, while one-level lookahead's direct children retain 97.5%. With K1 and Near/Far the receiver falls inside the actual AABB, giving that node the entire near half and raising its branch share to 50.2%. This alleviates the variance at this point and shows how strongly zero-near can still steer the distribution.

At the top interior, the critical light is the same one as at the old top-28, affected by K4 grouping at node 70. Its K4 component has height plus radius of −8.903, compared with 18.304 for the corresponding K2 component. K4 selects the light with 3.69% probability, compared with 19.22% for K2. Near/Far raises these to 35.42% and 36.67%, respectively. With Near/Far, native Prefix and lookahead remain close at the top, while K2 and K4 have about 1.53 and 1.61 times the variance of Prefix. K4 has a small advantage over K2 at side-b. The rankings hold in both windows.

The six-scene offline atlas occupies 16,824,960 bytes, including 8,778,240 new bytes for component AABBs. K1 reuses the node bounds. K2 and K4 add two RGBA32F reads per active component. Weighted by path probability at the three interior points, these account for about 38–64 and 76–126 fetches per sample, or 0.61–1.02 and 1.22–2.02 kB of requested texture bytes. Lookahead continues reading the direct children of cross-brick candidates. Static requests and offline baking exclude GPU build and timing results.

The six-scene Iris comparison covers 11 methods at 1 spp and a 32-frame mean, for 132 images. It reuses 48 images from four scenes. Scenes 2 and 4 recapture all 11 methods after voxel changes. Scene 2 also rebakes atlas slot 2 after emitter-key changes. Scene 4 retains matching native light-tree and atlas addresses. The original 72 images remain available in the archived comparison. [Current comparison](../../../builds/light-sampling/20261108-cluster-iris/comparison.html) · [Original comparison](../../../builds/light-sampling/20261108-cluster-iris/comparison-original-72.html)

On the common complete receiver pixels in each scene, Near/Far lowers 32-frame window variance in 29 of 30 pairs. Scene 3 Ward K4 is at 1.063 times its own off variant and 0.292 times fixed Prefix A. All five Near/Far tree variants fall below fixed Prefix A across the six scenes. In moving-4, Ward K1 reduces zero-direct-light pixels from 79/81 to 0/81 at side-a, and Ward K4 reduces them from 11/81 to 0/81 at the top. The whole-image and region statistics describe the captured windows. The [experiment index](../../../experiments/light-sampling/20261108-nearfar-combination.json) records the per-light variance and capture provenance.

### Independent Near/Far combination review (2026-11-08)

Review of result commit `9a40fc98f` accepts the six-scene quality comparison and the three interior-point diagnoses within their stated scope. Independent reconstruction from raw buffers matched the camera, emitter keys/masses, ordered tree and world voxels of all 66 cases to their scene's cohort before and after capture. Scene 2 has nine added and eight removed keys; scene 4 has unchanged keys. Both use newly captured controls for all eleven methods without mixing historical inputs.

An independent scalar implementation checked 2,503 valid GPU selection records across 40 new tree configurations. Maximum full-path PDF relative difference was 1.146×10⁻⁶. All eight receiver/window references across three new points and two fixtures were recomputed from raw conditional captures: 7,808 complete samples per light per window, and maximum absolute full-PMF difference from the report of 8.37×10⁻⁹. All 121 pixels in each 11×11 geometry neighborhood have consistent normals/materials and plane error below 3.73×10⁻⁶ blocks. M2 remains a finite conditional-sampling estimate, not an analytic ground truth.

Independent common-complete-pixel 32-frame statistics also match the report. To assess the extra benefit of the more complex methods, the following whole-image window-variance ratios use **Prefix A + Near/Far** as their denominator. Every column has Near/Far enabled; the agreed experimental baseline remains Prefix A.

| Scene | One-level lookahead | Ward K1 | Ward K2 | Ward K4 |
|---|---:|---:|---:|---:|
| ls1 | 0.579 | 0.958 | 0.714 | 0.607 |
| ls2 | 0.800 | 0.941 | 0.743 | 0.729 |
| ls3 | 0.792 | 0.977 | 0.943 | 0.773 |
| ls4 | 0.856 | 0.999 | 0.831 | 0.784 |
| ls5 | 0.903 | 0.893 | 0.922 | 0.685 |
| moving-4 | 0.860 | 1.371 | 0.959 | 1.148 |

Near/Far does alleviate the K1 side-face and K4 top-face counterexamples, but this does not accept the summary model. At side-a, K1 node 153 receives the entire near half, raising its branch probability from 0.254% to 50.175% while its ball cosine remains at the floor. With Near/Far, K4's top-point per-light variance estimate remains about 1.61 times Prefix with Near/Far. K4 slightly beats K2 at side-b, so surface orientation alone does not determine a consistent K ranking. Subsequent discussion should prioritize retaining one-level lookahead with Near/Far and keep K4 as a candidate. No default change or equal-time performance gain is accepted, nor are the earlier local Near/Far regressions in ls1 and other scenes established as fully repaired.

The review checked the 132-image matrix and original 72 image hashes. It sampled all eleven moving-4 methods and three scene-2 methods at 32 frames: AVIF/PNG pixels match, and sampled rows match the display transform and `raw[1079-display_y]` orientation. Local frames were inspected directly. The available browser connection exposed only an empty MCP Apps backend, so interaction acceptance reuses the experiment side's Chrome evidence. No new GPU work or shader edit occurred. The roughly 540 kB [compact per-light reference](../../../experiments/light-sampling/20261108-nearfar-combination-reference.npz) retains model IDs, world light keys, compact tree/child indices, native addresses, actual atlas columns, three origins/normals/PMFs and eight conditional-moment sets. Point variances remain reproducible after large raw captures are cleaned. The [experiment index](../../../experiments/light-sampling/20261108-nearfar-combination.json) records hashes, provenance and review scope.

### Historical October 31 images versus current results (2026-11-08)

The supplied Discord filenames identify the old ls5 and ls1 Half-space A + Near/Far 1-spp primary images. Both downloaded 1920×1080 images were inspected alongside the current model-6 view. The old AVIF payloads are absent, so matching filenames to tracked AVIF hashes does not re-establish pixel identity. Cross-checking the [old primary-image provenance](../../../experiments/light-sampling/20261031-temporary-baselines.json) against accepted job-result hashes establishes different actual emitter inputs:

| Scene | Old primary native-header emitters | Current raw-header emitters |
|---|---:|---:|
| ls1 | 3,921 | 10,235 |
| ls5 | 877 | 2,172 |

Preset and scene hashes still match in both scenes. They do not establish identical actual voxels, emitters or occlusion. Old counts belong to the primary-image captures, independently of conditional fixtures. The population change remains unexplained; counts alone cannot attribute it to chunk loading, view distance or emitter-extraction code.

Using the [compact per-light reference](../../../experiments/light-sampling/20261108-cluster-summaries-reference.npz), hold current lights, receivers and measured conditional RGB M2 fixed and change only original camera-relative versus complete world-prefix grouping. Independently recompute full-path model-6 PMFs. All twelve point-windows retain unit total probability and no positive-M2 support failures. Every row below enables Half-space A + Near/Far and reports world-prefix / original-camera estimated single-sample variance.

| Receiver | Two-window variance ratio |
|---|---:|
| ls1 pillar | 0.3205–0.3207 |
| ls1 wall | 0.1682–0.1684 |
| moving-4 historical side-26 | 1.766–1.768 |
| moving-4 historical side-27 (four windows across two groups) | 2.268–2.276 |
| moving-4 historical top-28 | 1.4316–1.4321 |

Grouping still has local regressions, but these two ls1 points do not support attributing their degradation to the switch to world prefix. These are fixed-point results under finite conditional M2. Historical moving points retain their near-edge limitations and do not represent whole images or the new face-interior points.

Mapping the retained old ls1 fixture's 3,921 keys into the current 10,235 lights matches every old key and mass; the reconstructed camera-relative tree also matches the old fixture's unordered topology hash. The additional 6,314 lights account for about 64.71% of current total flux. At the pillar/wall, they contribute only about 6.93% / 8.42–8.44% of the summed RGB components, yet receive 51.90% / 55.66% of current prefix model-6 selection probability. Hold current true evaluation fixed and estimate only the old-light illumination component: including the additional lights in the proposal raises that component's prefix-estimator variance by about 2.18 / 2.65 times. This isolates how additional candidates change probability allocation. It is not a historical render and does not classify unobserved finite-window contributions as strictly invisible. The old fixture comes from the model-1 primary capture; its key hash does not independently establish the full key set of the old model-6 primary image.

The source comparison spans old `78823b540` and current capture `e9132607`. Model-6 half-space, near/far mixing and normalization formulas are preserved. Brick-local emitter identification, counting, mass, in-light cube sampling and true voxel evaluation are unchanged. Outer grouping and layout change; new model ≥7 summary paths do not execute for model 6. Together with the earlier PDF audit, this investigation finds no additional weight/PDF implementation regression, without claiming that every runtime regression is excluded.

The [investigation record](../../../experiments/light-sampling/20261108-historical-comparison.json) retains image provenance, input identities, twelve point-window results, the old-light-component counterfactual and reproduction steps. Neither grouping alone nor emitter count alone apportions the whole-image historical gap. If whole-image attribution is pursued, freeze one ls1/ls5 voxel and emitter input and compare model 6 with only original camera-relative versus world-prefix grouping changed; investigate voxel-input drift separately. This round changed no shader, ran no GPU work and switched no default. Raw cleanup remains paused.

The user subsequently recalled increasing render distance from 16/24 to the current 32 chunks and authorized proceeding. This supplies a plausible emitter-growth explanation, without verifying old settings. Execution now moves to the [same-input grouping control](light-sampling-grouping-control-prompt.md): fix the actual live render distance and compare original camera model 6, World Prefix model 6 and Cosine BRDF in ls1/ls5, retaining 1-spp/32-frame outputs. Isolate whole-image grouping effects before considering a distance matrix. The experiment chat owns source/runtime operations and returns results to this chat for independent review; no default is switched.

### Same-input live grouping control (2026-11-08)

The comparison holds emitter keys and flux, full world voxels, receiver geometry and camera fixed within each scene, with 10,235 emitters in ls1 and 2,172 in ls5. The model-6 variants change only the outer quaternary grouping. The client render distance was 32 chunks and its simulation distance setting was 8.

| Scene or fixed surface | World Prefix / camera grouping, 32-frame RGB variance |
|---|---:|
| ls1 whole image | 1.333× |
| ls1 pillar / wall | 0.567× / 0.096× |
| ls5 whole image | 1.502× |
| ls5 vertical face within the old floor region | 5.154× |

Both light-sampling variants had lower whole-image variance than Cosine BRDF. World Prefix raised the whole-image fraction of zero-mean pixels from 3.83% / 42.74% to 5.34% / 52.71% (ls1 / ls5), while reducing variance on both fixed ls1 surfaces. All three methods were zero on the selected face in the old ls5 wall region, so that variance ratio is unavailable. Under matched inputs, World Prefix had higher measured whole-image variance in the 32-frame windows of both scenes. The historical render distance and attribution of the cross-date gap remain unresolved.

The [local 12-image comparison](../../../builds/light-sampling/20261108-grouping-control/viewer/comparison.html) contains 1 spp and 32-frame means for all three methods in both scenes. The [experiment record](../../../experiments/light-sampling/20261108-grouping-control.json) retains the source refs, input and tree identities, regional means and variances, raw paths, and image verification.

Independent review accepts the same-input diagnostic scope of result commit `416e050f2`. Raw-buffer reconstruction of all six cases confirms emitter keys/masses, ordered trees, complete world voxels and receiver inputs, with 453/118 identical brick-local subtrees in the two scenes. The maximum full-path PDF relative difference over 254 valid GPU selection records is 6.73×10⁻⁷. All 270 files in five manifests and the source-load receipts pass verification. Independently recomputed whole-image and four fixed-surface means, variances and zero coverage match the report. Common complete pixel counts are 2,022,469/2,066,956, excluding 5/0 incomplete or nonfinite receiver pixels.

The ls1 light proposals switch sources within one frozen job. Full voxel materials are captured only at the pair's start/end; each method additionally has its own camera, tree and receiver endpoints. The global material pair must not be described as separate material endpoints for each window. New BRDF full input endpoints also match. Old ls5 Prefix/BRDF emitter/voxel/camera endpoints and their single receiver snapshots match both endpoints of the new camera case, permitting reuse; the old windows have no additional receiver-end snapshots. The two tree sources change only outer layout/construction and share model 6 and evaluation. This review finds no additional weight or PDF implementation error.

Each of the four fixed surfaces has a 121-pixel neighborhood with matching normal/material and maximum inverse-depth plane-fit error below 3.59×10⁻⁸. All twelve images match the display transform and AVIF/PNG pixels, with every row satisfying `raw[1079-display_y]`. The mean images for both trees/scenes were inspected directly; browser interaction verification reuses the experiment-side check. Whole-image mean RGB channels differ by at most 0.109% between tree methods, without establishing unbiasedness or convergence.

The result distinguishes historical emitter-population growth from grouping regressions under fixed inputs. Retain world-stable ownership, continue repairing coarse-cluster weights in regressing regions, and require same-input whole-image comparisons; five-point improvements cannot establish whole-image improvements. Current ls1 shares the preceding emitter set but has different voxel/receiver inputs, so its 1.333 ratio must not be multiplied by older lookahead or Ward gains. Historical view distance, continuous movement and performance remain unverified. This review ran no new GPU work and changed no shader or default.

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
