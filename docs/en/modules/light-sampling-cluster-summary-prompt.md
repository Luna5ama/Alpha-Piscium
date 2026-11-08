# Light Sampling Cluster-Summary Experiment Prompt

Language: English | [简体中文](../../sc/modules/light-sampling-cluster-summary-prompt.md)

## Authorization and handoff (2026-11-08)

The human user in discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12` explicitly requested:

> 那就先用已有五点 M2，对比一层前瞻、单中心矩和两分量、四分量摘要，成本也一起看。写好提示词让实验agent做，完事让它通知你复查。

That requests comparison of one-level lookahead, single-center moments and two/four-component summaries using the existing five-point M2, including cost, delegation to the experiment agent, and a completion message requesting independent review. The user then added:

> 如果需要vibris 十侧你可以自行启动。

Existing experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` executes this round and reports back to the discussion chat for independent review. The human authorization covers dispatch, completion/substantive-blocker callbacks, in-scope revisions, and starting necessary Vibris measurements and startup preparation. Retain the experiment chat's current model and effort. The original authorization can be checked in the discussion chat.

Use [`I:/code/mcshaders/Alpha-Piscium-4`](../../../) (wt4), branch `2.0/light-sampling`. At preparation time HEAD was `1f4f0fd68` and the working tree was clean. Inspect subsequent changes before execution; do not reset. The experiment chat coordinates this round's working-tree changes and live runtime; the discussion side reviews after dispatch.

## Objective, starting evidence and current input gap

Deliver a quality/cost comparison at the five specified receivers to determine whether a small fixed number of distribution components improves coarse-cluster importance estimates, and which costs make it worthwhile or impractical. Start from the [current design discussion](light-sampling-design.md#lookahead-spatial-moments-and-multiple-center-summaries-2026-11-08), [reviewed five-point result index](../../../experiments/light-sampling/20261101-moving4-point-m2.json) and [previous conditional-reference task](light-sampling-world-grouping-prompt.md#current-task-moving-4-per-light-m2-and-five-point-repair-screening-2026-11-01).

Preparation found the entire wt4 `builds` directory missing. Historical `20261101-moving4-point-m2` originals and derived arrays cannot currently be read. The same path in other Alpha-Piscium worktrees and a targeted filename search under `G:/Graphics/Alpha-Piscium` also found no copy. Tracked indexes, source and fixtures remain. One read-only Vibris manifest query returned `SERVER_OFFLINE`, with `127.0.0.1:50051` refusing the connection. Historical verification does not establish that these files have now been recovered.

First make a bounded search for recoverable native artifacts, archives or derived arrays; validate their SHA, world light keys/mass, tree and receiver identity against the tracked index. Native statistics can regenerate per-light mean/M2; aggregate variance, rankings or fixtures alone cannot recover complete per-light M2. Historical capture work deleted matching remote manifests after verifying local archives; check whether any remote copy is actually available.

If no reference can be recovered, use the additional Vibris authorization to recapture missing conditional references at the same five receivers, explicitly recording this as new acquisition needed for recovery. Restore necessary services/startup preparation through the existing entry point, respecting repository restrictions on restarting a running Minecraft instance or launcher. Verify provenance and mapping with one point group before completing missing captures. If inputs differ from history, create a new cohort and recompute all candidates plus camera-A/world-prefix-A controls for that cohort; retain historical numbers as historical records. If services cannot be restored or the specified surfaces cannot be reproduced, retain evidence and report a substantive blocker. Do not substitute other points or geometric proxies for M2.

The five receivers are moving-4 side-26, side-27 and top-28, plus the old ls1 pillar and wall. See the [point-moment tool](../../../scripts/light_sampling_point_moments.py) and [ls1 fixture](../../../experiments/light-sampling/20261031-ls1-point-fixture.json) for locations. Historical textures are [moving4 sides](../../../shaders/textures/light_sampling_moving4_sides.bin), [moving4 side-top](../../../shaders/textures/light_sampling_moving4_side-top.bin) and [ls1 reference](../../../shaders/textures/light_sampling_reference.bin). They provide saved light/receiver identities, not verification of the current actual emitter set.

## Fixed comparison scope

Run the primary comparison on the complete world-prefix four-way tree, retaining the 32B A node as the base representation. Change weights only for internal candidates whose actual bounds span multiple world 16³ bricks. Brick-local nodes and leaves retain A, separating outer summaries from changes to local emitter weights.

| Method | Definition for this round |
|---|---|
| World-prefix A | Same-input primary baseline |
| One-level lookahead | Existing `outer-lookahead-1`: use the sum of direct children's A weights for each cross-brick internal candidate, following its verified definition |
| Single-center moments K=1 | One flux-weighted center and scalar spatial variance for each cross-brick candidate, evaluated with the ball kernel below |
| Two components K=2 | At most two components per cross-brick candidate, evaluated separately with the same ball kernel and summed |
| Four components K=4 | The same construction with a four-component limit |

Keep same-input original camera A as an additional quality control. Flat per-light A is an aggregation-error diagnostic and can be reconstructed from existing source. Historical two-level lookahead may be cited without extending this round's runtime matrix. SG, full covariance, higher moments, adaptive cuts, visibility, H-PLOC and near/far parameter searches remain outside this round.

Using the same ball kernel is this prompt's first-round experimental choice to isolate component count and compression. K=1 applies only to outer nodes and includes A's whole-cluster half-space test; distinguish it explicitly from the old whole-tree models 4/5.

## Component construction and weight evaluation

Each leaf starts as one component: mass from the current root-reachable emitter, center at voxel low + 0.5, and variance trace 5/12, following the [existing leaf-moment definition](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl). Process each K independently, bottom-up. Each parent gathers its children's at-most-K components, then compresses to at most K. K=1 must reproduce total mass, mean and variance computed directly from all descendant leaves.

For the first round, repeatedly merge the pair with the smallest increase in weighted squared spatial error. For components a and b, use `delta = Ma*Mb/(Ma+Mb) * dot(muA-muB, muA-muB)`. Choose the minimum-cost pair, merge mass and mean, and combine variance using each component's internal variance plus its squared displacement from the merged center. Repeat until K remain. At most four direct children mean each compression starts with at most 4K components. Implementations may optimize minimum search/candidate maintenance but must not silently change the merge objective.

Resolve ties through canonical world identities of the constituent components, such as their minimum descendant `(worldX,worldY,worldZ,material)` key. Ordering and tie-breaking must not depend on atomic allocation addresses, camera slots or receivers. Retain enough component information to audit merging. Require mass conservation, no duplicate leaf mass, and finite nonnegative statistics. Compare the combined first/second moments of every parent's components against direct leaf statistics. Reordering storage/input enumeration with fixed input must produce consistent canonical summaries.

First apply the existing A half-space test using the candidate node's actual AABB. A cluster proven entirely behind the receiver has zero weight. Otherwise, evaluate every positive-mass component as follows:

```text
delta = mu - origin
d2 = dot(delta, delta)
r2 = 2 * variance
c = clamp((dot(normal, delta) + sqrt(r2)) / sqrt(max(d2, 0.25)), 0.001, 1)
w = mass * c / max(d2, r2, 0.25)
clusterWeight = sum(componentWeights)
```

Only the original candidate's conservative AABB supports strict half-space rejection. The variance radius is an effective scale, not a bound enclosing every member; do not zero a component merely because its center lies behind the surface. Sum component weights before actual sibling normalization, then multiply probabilities along the real tree. Components estimate weights; the emitter set and main-tree parent/child membership still follow fixed world input.

Label the high-precision offline reference separately from the implementable float32 expression. Reuse the [current merging and weight implementation](../../../shaders/techniques/debug/LightSampling.glsl) and check numerical error at absolute world coordinates. Persistent storage may retain only cross-brick summaries, but construction cost must include required brick-local processing and all temporary data. Simply copying original child summaries is an equivalence check for one-level lookahead, not a new K-component fit.

## Five-point quality and attribution

Use two disjoint conditional-reference windows per primary receiver. Recovered previous data include side-27's repeated two-window capture across groups, totaling twelve point-windows. If recapture is necessary, follow the previous paired-point mode-3, 32-frame/8×4-batch protocol at the specified native 1920×1080 receiver pixels. Save complete world inputs before/during/after, actual origins/normals/albedo, trees and provenance. Derive sample counts from actual light count and resolution; retain zero samples and missing/nonfinite counts. True textures, conditional emitter sampling and visibility enter the conditional evaluation.

Reuse [moment extraction](../../../scripts/light_sampling_point_moments.py), [PMF screening](../../../scripts/light_sampling_point_screening.py), [layout reconstruction](../../../scripts/light_sampling_moving_analysis.py) and the [weight reference](../../../scripts/light_sampling_weight_reference.py). Reuse applicable validation after recovering originals; newly captured inputs require fresh checks of the per-pixel second-moment identity and input identity. Align trees by world position/material key and mass, use the audited actual receiver origin, and avoid applying the normal offset twice.

Enumerate every actual leaf probability and null probability. Compute RGB variance as `sum(M2_l/q_l) - dot(sum(mean_l),sum(mean_l))`. Report support failure when a positive-M2 light has zero probability. Never discard noncontributing lights and renormalize. For each window, report ratios to world-prefix A, camera A and flat per-light A. Keep all five points and complete arrays available for independent recomputation.

Focus attribution on the low root cluster's probability, observed M2 share, dominant penalty lights and the few splits contributing most to error. Compare candidate coarse-cluster weights with sums of descendant per-light A weights, checking whether compression absorbs small nearby light groups. sqrt(M2) is for offline attribution only, never component construction or parameter selection. Quality can be nonmonotonic in K: preserve counterexamples, and do not tune coefficients or hardcode orientations against these five points.

## Cost and conditional Vibris measurement

First report static cost for every method on a common basis. Query cost includes reach-probability-weighted node/summary slot inspections, component-kernel evaluations, half-space tests and normalization work per sample, plus maximum path work. Native four-way nodes contain empty slots; counting only nonempty logical children is insufficient. Include inspections of unused components when a fixed-K representation requires them. Separate logical records, physical slots and bytes; static inspection counts are neither memory transactions nor GPU time.

For storage, distinguish statistical payload, actual packed stride, counts/indices, active summaries, allocated capacity, persistent buffers and peak construction scratch. K=1 may reuse the parent's mass; K=2/4 require separate mass per component. Provide one concrete implementable layout and access order, not just the 20B component payload. Construction accounting covers all required levels, component merges, candidate-pair comparisons and temporary reads/writes. Separate once-per-frame preparation from per-sample queries. Label CPU Python elapsed time only as offline-analysis time.

When a candidate has clear quality gains or a worthwhile quality/cost tradeoff to resolve, the user authorizes a minimal runtime implementation and Vibris measurement. Freeze candidate source and verify the loaded source, revision, preset, settings, emitter set and relevant passes. Prioritize moving-4 measurements of world-prefix A, one-level lookahead and worthwhile K candidates, adding ls1 if necessary. Prefer 720p for ordinary timing; specified-point/PMF audits may use native 1080p. State the purpose and keep each comparison consistent.

Include full emitter extraction/tree construction/summary preparation and the actual [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl) path. Report a separate summary stage individually, plus relevant sibling passes and a stable sentinel. Match conditions, warmup, frame counts and statistics; use adjacent forward/reverse order and repeat when noise requires it. The prototype must actually consume summaries and change selection probabilities; align GPU diagnostics with the offline reference first. If the timing scene differs from the M2 cohort, label them separately instead of treating the combination as same-input equal-time quality acceptance.

Candidates rejected offline may stop at static cost reporting. If runtime feasibility forces a different kernel, compression rule or support domain, report that design change rather than using a different algorithm's timing to endorse the original. Live-service recovery failure blocks only dependent work; complete what available data permits and report gaps. Promotion to production defaults still requires discussion-side review and user acceptance.

## Artifacts, self-check and completion callback

Follow [AGENTS](../../../AGENTS.md), [script rules](../../../scripts/AGENTS.md) and [documentation rules](../../AGENTS.md). Use PowerShell 7 without a profile, lowest priority for non-benchmark builds/long offline tasks, and preferably the latest GraalVM. Read shader rules and the opengl-glsl skill before GLSL changes; commit maintained sources and generated outputs together. Use durable Vibris waits without extra progress-report polls. Exit conditional diagnostic mode at completion and report actual source, settings, live jobs and working-tree state.

Keep this round's inputs, component arrays, complete PMFs, quality/cost tables, generation/recomputation scripts, originals and hashes in an independent directory, suggested as `builds/light-sampling/20261108-cluster-summaries/`. Because old builds are currently unreadable, also deliver a compact recoverable reference package containing world light keys/mass, the complete reachable tree, per-light mean/M2, receivers, completeness statistics and provenance hashes. Track it with the result index under [experiments](../../../experiments/) when reasonably sized. Store large raw images separately and retain them at least until independent review completes. Preserve historical result identities and account separately for recovery and new acquisition.

At minimum, check mass/statistical conservation, K=1/direct-moment agreement, canonical input-reordering stability, complete emitter sets and PDF support, and cost models including empty slots and all construction dependencies. Existing failure points are required counterexamples. A reviewable shortlist or a conclusion that these fixed summaries still fail to serve all five points is a valid result; do not force a winner. Commit necessary tools/prototypes, one tracked result index and bilingual findings after link checks and `git diff --check`/`git diff --cached --check`.

After completion, commit and self-check, call `send_message_to_thread` with host=`local`, target=`01a18c0e-8678-7800-8ab8-04b69f072f12`, explicitly requesting:

> 请现在复查五点 M2 的一层前瞻、单中心矩、两分量／四分量摘要质量与成本对照。

Include absolute paths to this prompt and the result index, commits, recovered/recaptured data status, actual candidate formulas and component construction rules, complete-array/original-artifact entry points, five-point/two-window results, static/GPU costs and untested scope, and runtime cleanup status. Report substantive blockers through the same callback. Routine intermediate progress needs no itemized callback; delivering a plan alone does not complete this experiment.
