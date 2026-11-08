# Light Sampling Near/Far Combinations and Face-Interior Diagnostics

Language: English | [简体中文](../../sc/modules/light-sampling-nearfar-combination-prompt.md)

## Authorization, objective and handoff (2026-11-08)

The user in discussion chat `01a18c0e-8678-7800-8ab8-04b69f072f12` requests retaining World Prefix + Half-space A as the baseline. Keep existing World Prefix A, one-level lookahead and Ward K1/K2/K4, and add a Near/Far variant of each. Retain independent Cosine BRDF-only. The user also notes that the old side-26/top-28 markers sit near corners and wants to understand why one-level lookahead looks more consistent, K1 regresses severely, and K4 can be worse than K2.

Existing experiment chat `01a18648-f66e-78e0-8d01-cd906ca45429` executes and reports back to the discussion chat on host=`local` for independent review. Continue the human authorization for experiments, necessary Vibris startup, dispatch/completion/substantive-blocker callbacks and in-scope revisions; retain current model and effort. Use [wt4](../../../), `I:/code/mcshaders/Alpha-Piscium-4`, branch `2.0/light-sampling`. Preparation HEAD was `4a164df4b`, with a clean working tree. Inspect subsequent differences before execution; do not reset. You coordinate source changes and runtime; the discussion side reviews methods and results.

Deliver an interpretable same-input, eleven-method live comparison showing improvements and regressions from combining Near/Far with coarse-cluster corrections. Preserve counterexamples: an offline regression is not a reason to omit a requested combination. Repair or recapture correctness/input-identity failures; report concrete blockers if necessary rather than presenting invalid frames as valid comparisons.

## Starting evidence and known issues

Read the [design and independent review](light-sampling-design.md#independent-cluster-summary-review-2026-11-08), [five-point results and reference](../../../experiments/light-sampling/20261108-cluster-summaries.json), [live result index](../../../experiments/light-sampling/20261108-cluster-iris.json) and [current 72-image viewer](../../../builds/light-sampling/20261108-cluster-iris/comparison.html). Live prototype source is on local branch `experiment/light-sampling-cluster-iris` (`a4188825`); the moving-4 atlas is on `experiment/light-sampling-moving4-iris` (`fdd330fc`). These prototypes are not enabled by default on the working branch; its shader alone does not identify captured execution.

The discussion side additionally matched every live atlas component to the reviewed float32 summaries. Recomputing all 252 valid moving-4 audit records for lookahead/K1/K2/K4 using current native addresses gave a maximum GPU PDF relative difference of about 8.70×10⁻⁷. Between the old conditional capture and live capture, 5,606 physical node addresses changed. Align through the current actual tree or world identity, not old brick-local addresses.

K2 and K4 are compressed separately at every level. At 57 of 344 cross-brick clusters, K4 is not a refinement of K2. Node 70's K4-to-K2 membership intersections are `[[40,0],[1,47],[0,76],[0,13]]`; its flux-weighted within-component variance sum falls from about 119,539 to 53,735, yet the critical nearby emitter at top-28 gets a worse estimate. The fitting objective differs from receiver contribution, and variance balls do not enclose members. These checks support the existing model counterexample without establishing that every full-frame path is free of other problems.

## Fixed comparison matrix

Retain the six scenes `light-sampling-1` through `light-sampling-5` and `light-sampling-failure-moving-4`, native 1920×1080, a 1-spp frame and a 32-frame linear mean (8×4 batches). Keep direct-only, G16, four-way branching, cube sampling, RC/reuse disabled and the existing display transform; retain independent Cosine BRDF-only. Eleven methods × six scenes × two frame choices produce 132 images. Reuse the existing 72 images if their inputs still match, adding 60 images. When a scene's input changes, acquire matching controls for that scene, retain historical provenance and avoid presenting different cohorts as a controlled comparison.

| Structure/summary | Near/Far off | Near/Far on |
|---|---|---|
| World Prefix | World Prefix + Half-space A | World Prefix + Half-space A + Near/Far |
| One-level lookahead | One-level Lookahead + Half-space A | One-level Lookahead + Half-space A + Near/Far |
| K1 | Ward K1 + Half-space A | Ward K1 + Half-space A + Near/Far |
| K2 | Ward K2 + Half-space A | Ward K2 + Half-space A + Near/Far |
| K4 | Ward K4 + Half-space A | Ward K4 + Half-space A + Near/Far |

The viewer may separate method and Near/Far into dimensions, but BRDF-only must correspond to one actual case. Compare Near/Far on/off in pairs; the baseline remains World Prefix + Half-space A without Near/Far.

## Combination formula for this round

The following is the discussion side's selected first-round definition, an experimental model without an assumed repair benefit. Retain the original Near/Far half-and-half mixture and zero-near-distance rule. Do not add distance clamping, visibility, coefficient searches or a different clustering objective.

The existing Near/Far branch recomputes and overwrites a coarse node's scalar weight. Appending it after current lookahead/Ward weighting can erase those corrections. Instead, each candidate child j supplies three unnormalized scores `N[j], F[j], Z[j]`; normalize and mix over the current sibling set. Do not normalize within each child's representatives before summing them.

Every representative r retains its method's nonnegative angular amplitude `a_r = mass_r * cosine_r` and an actual member-enclosing AABB `B_r`. Use squared nearest/farthest distances from the receiver to `B_r`:

```text
N[j] = sum(a_r / near2_r for near2_r > 0)
Z[j] = sum(a_r for near2_r == 0)
F[j] = sum(a_r / far2_r)

if sum(F) == 0: return existing null event
p_near[j] = Z[j] / sum(Z) if sum(Z) > 0 else N[j] / sum(N)
p[j] = 0.5 * (p_near[j] + F[j] / sum(F))
```

Skip division for zero-amplitude representatives. Select with the actual normalized `p[j]`, multiply actual branch probabilities along the tree and preserve existing null semantics. Strict half-space tests retain each original method's actual AABB and tolerance. Define representatives as follows:

- World Prefix: one native candidate node is one representative, using A's angular amplitude and native AABB. Near/Far on must reproduce current model 6 branch probabilities and null events.
- One-level lookahead: a cross-brick internal candidate uses its direct children, each with A's angular amplitude, actual AABB and half-space test. Accumulate all three scores before current-sibling normalization. Other nodes use the native representative.
- Ward: a cross-brick internal candidate uses its existing K components and ball-kernel angular amplitudes, replacing only distance weighting with the Near/Far rule. Keep the whole-candidate AABB half-space test without adding component-level hard rejection. Add actual member AABBs to components through min/max during the same Ward merging. Do not use `sqrt(2*variance)` as a support bound or give every component its parent's AABB. K1 component bounds must equal node bounds.

With Near/Far on, brick-local and leaf candidates uniformly use native Half-space A + Near/Far; off retains the verified A path. Ward membership, mass, mean and variance must not change with the switch. New bounds supplement statistics without changing pair costs or tie-breaking. The existing ball cosine may still underestimate nearby lights: retain that counterexample instead of silently repairing it too.

If this definition cannot be reconciled with the existing consumer or original Near/Far semantics, send a minimal counterexample to the discussion side first. Do not silently replace it with a mixture of complete proposals or collapse all four advanced combinations into model 6. Data layout, shader organization, baking and configuration IDs may change within this semantics; update all generated consumers with maintained source. Offline baking remains acceptable for this prototype, explicitly excluding dynamic construction cost.

## Face-interior diagnostics and implementation checks

Keep side-26/side-27/top-28 as historical anchors rather than using individual corner pixels to represent entire faces. First choose two side-face interior receivers and one upward-face interior receiver, each with a small region, in moving-4. Verify a continuous surface using actual gbuffer, depth and world position, avoiding geometry edges, occlusion contours, grass blades and material/normal discontinuities. Choose from baseline geometry and freeze them before comparing methods; do not select regions by which method looks favorable. Save pixel coordinates, world origins/normals, materials and an illustration. Clearly distinguish new and old markers, allow hiding region overlays and do not burn them into images.

Start with finite 32-frame interior-region statistics from existing originals, reporting zero-sample coverage and complete-pixel counts. Darkness or a missed event alone does not establish bias. Capture same-input, two-window per-light M2 for the three new receivers with the existing audited conditional-reference protocol. Compare all ten tree methods using complete PMF/null, true-M2 variance and dominant penalty branches. The old five-point package remains usable for recomputation, but the old ls1 emitter cohort differs from live captures and must not be mixed with them.

Prioritize these checks:

- K2/K4 membership, mass and moment conservation; new support bounds enclose every member and match the GPU atlas.
- Near/Far off matches the existing implementation; Prefix on matches model 6; lookahead/Ward on actually consume their representatives instead of having their scores overwritten by the final Near/Far branch.
- Every new configuration executes in target Iris. Verify actual source, revision, scene, settings, light keys/mass, voxels, tree and selection PDFs. Use each capture's actual native addresses. Check complete path probabilities for valid audit records rather than treating a configured switch as execution evidence.
- Identify where critical nearby emitters lose probability: compression, cosine floor, zero-near-distance monopolization, or implementation/mapping. Separate demonstrated mechanisms from unresolved possibilities.

## Delivery, cost, cleanup and callback

Update the [same live comparison page](../../../builds/light-sampling/20261108-cluster-iris/comparison.html), preserving reusable existing images and adding combination images and interior markers. Check all 132 valid combinations, image orientation, lossless conversion, browser method/frame/scene switching and marker zoom behavior. Identify recaptures, missing cases or input changes. Never substitute simulated images for Iris frames without saying so.

Report the five Near/Far on/off pairs, comparisons against the fixed baseline, and interior/historical-point counterexamples. First report added prototype bounds/atlas bytes and query work. Baking time and static accesses are not GPU construction time or equal-time benefit. This task requests quality combinations, not an additional full performance matrix. If measuring the actual initial path opportunistically, use matched conditions and state scope separately; omitting preparation cost cannot establish production acceleration.

Follow repository and relevant shader/script/documentation instructions, PowerShell 7 without a profile and Idle priority for long non-benchmark work. Use durable Vibris waits and coordinate the runtime alone. Existing authorization permits cold startup of needed services, not unrequested restarts of a running Minecraft instance or launcher. Verify actual restored state and live jobs at completion. Do not promote production defaults this round.

Apply the user's latest workspace-cleanup convention: prefer reusing wt4. Creators clean new temporary checkouts, duplicate source copies/archives and logs while protecting unique implementation, uncommitted changes, active tasks and Git object dependencies. The final viewer/images are user deliverables, retained until superseded. Full originals remain temporarily for capture validation and discussion-side independent review; report exact paths, purpose and who cleans them after review. Keep useful conclusions in the existing tracked index and bilingual topic summary rather than archiving complete output indefinitely. Do not bypass an existing automatic-policy rejection.

Commit prototype source and necessary generated outputs, the result index and bilingual findings after applicable checks and `git diff --check`/`git diff --cached --check`. On completion or a substantive blocker, call `send_message_to_thread`, host=`local`, target=`01a18c0e-8678-7800-8ab8-04b69f072f12`. Include this prompt, commits, viewer/index paths, the actual combination formula, evidence for the three new points, verified/unverified scope and cleanup/runtime status, explicitly requesting review. Routine progress needs no itemized callback. A plan or offline screening alone does not deliver the requested complete live viewer.
