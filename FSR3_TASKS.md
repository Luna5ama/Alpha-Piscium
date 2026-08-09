# FSR3 Serial Execution Ledger

This file is the durable source of truth for the active FSR3 remediation goal.

```text
Worktree: I:\code\mcshaders\Alpha-Piscium-8
Branch: 1.10/fsr3
Baseline HEAD: 0a8f784367b43f0d615a50744167ef42d5e821df
Baseline subject: fi fsr3
Created: 2026-08-09
```

Every goal continuation must read this file completely before inspecting code, editing, validating, or committing. The older review handoff is background evidence; this ledger owns task order, state, and completion criteria.

## Goal

Correct the existing FSR3 Upscaler integration so that its temporal exposure and color-space contracts are explicit and mathematically sound, bright saturated edges remain stable, Off/TAA/FSR3 share the intended post-processing behavior, and the result is compiled, visually validated, measured, documented, and committed in atomic steps.

Execution rule:

> One goal turn completes at most one task. A completed task must be validated and committed atomically before a later goal turn starts the next task.

The goal stays active until every required task is complete or a genuine external blocker is recorded.

## Mandatory turn protocol

At the start of every goal turn:

1. Read this file completely.
2. Run:

   ```powershell
   git status --short --branch
   git rev-parse HEAD
   git worktree list
   ```

3. Confirm the worktree is exactly `I:\code\mcshaders\Alpha-Piscium-8` on `1.10/fsr3`.
4. Preserve all pre-existing user changes. If the tree is dirty, establish ownership before editing.
5. Select exactly the first task marked `READY`.
6. Perform only that task. Do not start the next task during the same turn.
7. Run its task-specific validation plus:

   ```powershell
   git diff --check
   git diff --cached --check
   ```

8. Update this ledger with concise results and evidence in the same commit as the task.
9. Stage only files owned by the task, review the staged diff, and create one atomic commit whose subject contains its `FSR3-Txx` ID.
10. Stop after reporting that commit. Leave the goal active so the next continuation rereads this file.

Do not write a literal commit hash into the commit that creates it. `Commit: this task's commit` identifies the commit containing that task's completed status. Later sessions can resolve it with:

```powershell
git log --oneline --grep="FSR3-Txx"
```

If a task reveals a new defect:

- Do not hide the fix inside an unrelated task.
- Add a narrowly scoped remediation task immediately before the next dependent validation/finalization task.
- Leave the failed validation task `READY` until the remediation commit exists and its affected cases pass again.
- Commit only one remediation task per later turn.

If a task is genuinely blocked, record the exact decision/state required and evidence. Do not mark it complete and do not start a dependent task.

## Status vocabulary

- `DONE`: completed, validated, and committed.
- `READY`: the next executable task.
- `PENDING`: waiting for dependencies.
- `BLOCKED`: waiting for a named external decision or state.
- `SUPERSEDED`: replaced by a documented task, with reason.

Exactly one task should normally be `READY`.

## Constraints

- Do not modify imported AMD FFX implementation code.
- Project-owned integration code may be changed even when stored beside FFX sources.
- Do not preserve obsolete paths through compatibility aliases or parallel implementations.
- Do not add an abstraction without at least two real uses.
- Keep generated files synchronized with maintained generator inputs.
- Do not hand-edit generated wrappers, `shaders/base/{Options,TextOptions,Textile}.glsl`, `shaders/lang/*.lang`, or root `shaders/shaders.properties`.
- Run generators from `scripts/`; use `kotlin options.main.kts` for a complete generation when program/options/property sources change.
- Use Shadesmith only when `shaders/shadesmith.json` changes or a fresh-worktree bootstrap is required.
- Vibris MCP is available for shader loading, runtime diagnostics, captures, comparisons, and performance measurement.
- The user-selected target runtime is Minecraft 1.21.11/Iris.
- No automated check replaces target Minecraft/Iris compilation and visual validation.
- Keep every commit atomic and exclude unrelated user changes and ignored artifacts.

## Protected and project-owned files

Protected vendor implementation includes the AMD algorithm bodies, particularly:

```text
shaders/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_*.glsl
shaders/techniques/ffx/fsr1/ffx_fsr1*.glsl
shaders/techniques/ffx/spd/ffx_*.glsl
shaders/techniques/ffx/ffx_core.glsl
```

These must remain byte-identical to baseline `0a8f7843` unless the user explicitly relaxes the constraint.

Project-owned integration code includes:

```text
shaders/techniques/ffx/fsr3upscaler/Integration.glsl
shaders/techniques/ffx/fsr3upscaler/README.md
shaders/techniques/ffx/fsr1/RCAS.glsl
shaders/pass/composite/FSR3*.comp.glsl
shaders/pass/composite/RCAS.comp.glsl
shaders/pass/composite/TAAPrepare.comp.glsl
shaders/util/AgxInvertible.glsl
scripts/programs.main.kts
scripts/options.main.kts
scripts/shaders.properties
shaders/shadesmith.json
maintained bilingual documentation
```

For every implementation commit, verify protected files against baseline:

```powershell
git diff --name-only 0a8f7843 -- shaders/techniques/ffx
```

Review any result carefully: `Integration.glsl`, `README.md`, and project `RCAS.glsl` are allowed; imported `ffx_*` algorithm files are not.

## Source material

Primary review handoff:

```text
I:\code\mcshaders\Alpha-Piscium-4\.codex\FSR3_REVIEW_HANDOFF.md
```

Reviewed snapshot in that handoff:

```text
5416d6f8d6b0635dfaba04993bc58b061e7f7765
```

The current branch tip `0a8f7843` was rechecked while initializing this ledger. Its FSR3 exposure, luma-pyramid, RCAS, and AgX behavior remains materially identical to the reviewed snapshot.

Official references:

- AMD FSR3 Upscaler manual: <https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-upscaler/>
- AMD FSR3 source tree: <https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/tree/release-FSR3-3.0.4/sdk/include/FidelityFX/gpu/fsr3upscaler>
- Imported source revision recorded in the branch: `60f4ea81909200d8542eca14dccb2628b763a9a3`
- Local Iris references: `.agents/iris-docs/`
- Maintained pipeline documentation: `docs/en/modules/`

## Current pipeline and color domains

The intended high-level order is:

```text
render-resolution lighting/composites
  -> DOFPrepare
  -> TAAPrepare
  -> mode branch
       Off:  TAAResolve non-temporal path
       TAA:  TAAResolve -> FXAA
       FSR3: MotionVectors
             -> PrepareInputs
             -> LumaPyramid
             -> ShadingChangePyramid
             -> ShadingChange
             -> PrepareReactivity
             -> LumaInstability
             -> Accumulate
  -> shared RCAS for TAA/FSR3
  -> Bloom at output resolution
  -> PostComposite / display transform
  -> ExposureMip / ExposureGather for later display exposure
  -> Final
```

Current FSR3 color path:

```text
unexposed scene-linear HDR
  -> multiply Integration.Exposure()
  -> AMD max-channel Tonemap / YCoCg reconstruction and accumulation
  -> AMD InverseTonemap
  -> divide Integration.Exposure()
  -> unexposed scene-linear atlas history/output
  -> multiply shaderpack display exposure in shared RCAS
  -> agxInvertible_forward
  -> FSR1 RCAS
  -> agxInvertible_inverse
  -> exposed linear main
  -> Bloom
  -> display transform
```

`DeltaPreExposure() == 1` is currently consistent with un-pre-exposed scene-linear input and history. It must not be replaced with a guessed exposure ratio without re-deriving the complete storage equations.

AMD requires linear HDR input. FSR's internal `Tonemap`/`InverseTonemap` is a temporary reconstruction-domain transform whose output is restored to the input domain. Do not move final display AgX wholesale before HDR FSR3.

## Confirmed defects

### F1 — Independent FSR3 exposure is computed but unused

`FSR3LumaPyramid` writes `FRAME_INFO_EXPOSURE`, but `Integration.Exposure()` returns:

```glsl
exp2(global_aeData.expValues.z)
```

No consumer reads the computed FSR3 exposure. Upsample, Shading Change, Prepare Reactivity, Luma Instability, Reproject, and Accumulate therefore follow the shaderpack display auto-exposure fade.

### F2 — FSR3 auto exposure is independent but still temporally smoothed

After reset, its log luminance uses approximately:

```text
previous + (current - previous) * (1 - exp(-DeltaTime()))
```

It is independent from display fade, not literally time-independent.

### F3 — Imported dark-luma expression collapses sub-1 luminance

The protected luma-pyramid source contains:

```glsl
ffxMax(FSR3UPSCALER_EPSILON, log(fLuma))
```

All `0 < fLuma <= 1` values therefore collapse to approximately epsilon. The source must not be edited. Its exact upstream provenance and a project-owned workaround must be established.

### F4 — Linear HDR placement is correct

FSR3 should receive scene-linear HDR. Its max-channel internal transform scales RGB uniformly, temporarily compresses range for temporal reconstruction, then reverses before output. Feeding final display-tonemapped AgX into the current HDR configuration would violate the documented contract.

### F5 — The current AgX wrapper is lossy under defaults

`agxInvertible_forward()` also performs Bloom highlight compression, while `agxInvertible_inverse()` does not reverse that compression. At defaults (`compression=3`, RGB mode), exact numerical evaluation produced:

```text
(1024, 0, 0)  -> approximately (938, 3.8, 3.8)
(4096, 0, 0)  -> approximately (2666, 63, 63)
(65504, 0, 0) -> approximately (12620, 1467, 1468)
```

This can change bright saturated pixels even when spatial sharpening is disabled.

### F6 — AgX around RCAS cannot affect earlier FSR temporal decisions

Rectification, luma instability, deringing, and history accumulation happen before RCAS. A later AgX wrapper cannot make those decisions match final AgX/highlight-compression appearance. This mismatch is real, but it does not prove that replacing AMD's internal transform is the correct fix.

### F7 — FSR3 SPD and SST debug share atomic counter 15

Both use `global_atomicCounters[15]`. Simultaneous use can corrupt synchronization/debug state even though accesses remain in bounds.

## Task queue

### FSR3-T00 — Initialize branch and execution ledger

Status: `DONE`
Dependencies: none
Commit: this task's commit

Completed scope:

- Checked out `1.10/fsr3` in `I:\code\mcshaders\Alpha-Piscium-8` as explicitly requested.
- Read the source handoff completely.
- Rechecked the current branch tip and confirmed the reviewed defects remain.
- Recorded constraints, pipeline, findings, serial tasks, per-task gates, and continuation protocol.
- Changed no shader behavior.

Evidence:

- Branch: `1.10/fsr3` tracking `origin/1.10/fsr3`.
- Baseline HEAD: `0a8f784367b43f0d615a50744167ef42d5e821df`.
- Worktree was clean before adding this ledger.

### FSR3-T01 — Audit the exposure contract and select a vendor-safe design

Status: `DONE`
Dependencies: FSR3-T00
Commit: this task's commit

Do not change runtime shader behavior in this task.

Deliverables:

1. Confirm whether exact upstream revision `60f4ea...` contains the suspicious dark-luma expression or whether the local port introduced it.
2. Trace every use of `Exposure()`, `DeltaPreExposure()`, `FrameInfo()`, input color, current luma, previous luma, history color, and stored output.
3. Derive equations for first frame, stable exposure, changing exposure, history reset, and display-exposure adaptation.
4. Compare project-owned workaround options without touching protected vendor source.
5. Select the smallest design that makes FSR temporal exposure independent from display fade and preserves scene-linear history.
6. Specify exact T02 files/resources/pass-order changes and reset behavior.
7. Define numeric dark-scene cases and expected finite positive results.

Required evidence:

```powershell
rg -n "FRAME_INFO_EXPOSURE|Exposure\(\)|DeltaPreExposure\(\)|FrameInfo\(" shaders/techniques/ffx/fsr3upscaler
rg -n "global_aeData\.expValues|global_fsr3FrameInfo" shaders scripts
git log --oneline --follow -- shaders/techniques/ffx/fsr3upscaler/Integration.glsl
git show 98382249 -- shaders/techniques/ffx/fsr3upscaler/Integration.glsl shaders/pass/composite/RCAS.comp.glsl
git show 2dec4f08 -- shaders/techniques/ffx/fsr3upscaler/Integration.glsl
```

Acceptance:

- T02 has no remaining architecture decision.
- `DeltaPreExposure()` is mathematically justified rather than guessed.
- No shader, generator, generated output, or protected vendor file changes.
- This ledger is the only file in the task commit.

#### Upstream provenance

The exact imported revision is the public AMD commit `60f4ea81909200d8542eca14dccb2628b763a9a3`, titled `AMD FSR SDK 2.3.0`. The current path at that revision is:

```text
Kits/FidelityFX/upscalers/fsr3/include/gpu/fsr3upscaler/ffx_fsr3upscaler_luma_pyramid.h
```

It contains the same expression as the local GLSL port:

```text
ffxMax(FSR3UPSCALER_EPSILON, log(fLuma))
```

The expression was not introduced by the GLSL port. The old `sdk/include/...` URL in the source handoff is stale for this revision; use:

<https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/blob/60f4ea81909200d8542eca14dccb2628b763a9a3/Kits/FidelityFX/upscalers/fsr3/include/gpu/fsr3upscaler/ffx_fsr3upscaler_luma_pyramid.h>

The same upstream revision computes:

```text
deltaPreExposure = currentPreExposure / previousFramePreExposure
```

and aliases the input-exposure SRV to `frame_info.x` when FSR3 auto exposure is enabled.

#### Complete current dataflow

Let:

```text
C_t     = current un-pre-exposed scene-linear RGB
L_t     = dot(max(C_t, 0), Rec.709 luminance coefficients)
E_t     = FSR reconstruction exposure for the current frame
P_t     = application pre-exposure; P_t = 1 in this shaderpack
D_t     = P_t / P_(t-1); therefore D_t = DeltaPreExposure() = 1
H_(t-1)= stored scene-linear FSR color history
Q_(t-1)= stored scene-linear luma history
A_t     = shaderpack display exposure, exp2(global_aeData.expValues.z)
T       = AMD's temporary HDR max-channel Tonemap/YCoCg reconstruction transform
```

The reader/writer trace is:

| Value | Writer/storage | Readers and domain |
|---|---|---|
| Input color `C_t` | `TAAPrepare` leaves `usam_main` unexposed in FSR3 mode | `LoadInputColor`; Upsample uses `E_t * C_t` |
| Current luma `L_t` | Prepare Inputs stores alternating R channels in the render-history atlas | Luma Pyramid reads it unexposed; Shading Change, Prepare Reactivity, and Luma Instability multiply it by `E_t` |
| Previous luma `L_(t-1)` | Opposite parity current-luma tile | Shading Change uses `E_t * D_t * L_(t-1)`; reset returns zero |
| Luma history `Q_(t-1)` | Luma Instability stores after dividing the entire vector by `E_t` | Next frame samples it, then multiplies by `E_t * D_t`; reset returns zero |
| Color history `H_(t-1)` | Accumulate stores RGB after inverse internal tonemap and division by `E_t` | Reproject loads it and multiplies by `E_t * D_t`; reset returns zero |
| Current FSR output | Current parity color-history tile | Shared RCAS reads it as unexposed scene-linear RGB |
| `frameInfo.x` | Luma Pyramid `StoreFrameInfo` | Currently no reader; T02 makes `Exposure()` read it directly |
| `frameInfo.y` | Luma Pyramid smoothed log luminance | Only the next Luma Pyramid dispatch reads it |
| `frameInfo.z` | Luma Pyramid arithmetic scene-average luminance | `SceneAverageLuma()` exists but has no call site |
| `frameInfo.w` | Shared RCAS writes `frameCounter` | `FSR3HistoryReset()` detects skipped FSR3 frames/mode changes |
| Display exposure `A_t` | Exposure Gather after the display pipeline | Shared RCAS applies it once after FSR3; it must not enter FSR temporal reconstruction |

`LoadFrameInfo()` is the Luma Pyramid read/write callback. `FrameInfo()` is nominally the read-only callback, but the local port currently aliases it to the reset-substituting `LoadFrameInfo()`. T02 restores the upstream distinction so current frame data remains visible after the Luma Pyramid writes it, including on a reset frame.

`Exposure()` consumers are Upsample, Shading Change Pyramid, Prepare Reactivity, Luma Instability, Reproject, Accumulate, and the unused imported FSR3 RCAS path. The active shared FSR1 RCAS intentionally uses display exposure instead.

#### Exposure and history equations

Current and reprojected color enter the same current-frame reconstruction domain:

```text
N_current = E_t * C_t
N_history = E_t * D_t * H_(t-1)
```

AMD performs rectification and accumulation on `T(N_current)` and `T(N_history)`, reverses `T`, then stores:

```text
H_t = max(0, inverse_T(reconstruct(T(N_current), T(N_history), ...)) / E_t)
```

Because `D_t = 1`, `H_t` remains un-pre-exposed scene-linear RGB for any finite positive `E_t`. A changing FSR exposure is applied equally to current and history and is divided out before storage. Replacing `DeltaPreExposure()` with an FSR exposure ratio would apply an extra ratio to history and is incorrect.

Luma comparisons follow the same rule:

```text
luma_current_domain  = E_t * L_t
luma_previous_domain = E_t * D_t * L_(t-1)
luma_history_input   = E_t * D_t * Q_(t-1)
Q_t                  = updated_luma_history / E_t
```

Required cases:

| Case | Result |
|---|---|
| First frame or reset | Luma Pyramid writes current `E_t` before any consumer; history loaders return zero and `FrameIndex()` returns zero; stored output returns to scene-linear via `/ E_t` |
| Stable FSR exposure | Current and history use the same constant scale, which cancels at storage |
| Changing FSR exposure | Both current and stored scene-linear history use the new `E_t`; no prior FSR exposure is required |
| Display-exposure adaptation | `A_t` is absent from all FSR temporal equations and is applied once after upscale, so its fade cannot move FSR clipping/history decisions |
| Shader reload / mode switch | initialization or the missing `frameInfo.w == frameCounter - 1` marker resets history; the current Luma Pyramid dispatch still overwrites exposure before use |
| Resize / render-scale change | exposure has no persistent size-dependent state; the current dispatch reduces the new `RenderSize()`. Temporal resource/reset correctness remains covered by T06/T07 |
| Teleport / camera cut | exposure is recomputed from the current frame and does not reuse a previous exposure; existing FSR history validity/reactive handling remains independently validated in T06/T07 |

#### Workaround comparison and decision

1. Reading the existing `frameInfo.x` without other changes is rejected: it preserves the upstream sub-1 collapse and the internal temporal smoothing.
2. Recomputing exposure from the global arithmetic mean in `frameInfo.z` is dark-safe and small, but it is overly biased by bright outliers and diverges more from AMD's documented average-log estimator.
3. Adding a new exact log-luminance reduction pass/resource is vendor-safe, but duplicates SPD work and changes scheduling/resources without evidence that it is needed.
4. Encoding luminance before the protected source's `log()` is rejected because it either overflows the FP16 pyramid or destroys the independent arithmetic-luma channel.
5. Changing `DeltaPreExposure()` to an exposure ratio is mathematically wrong for scene-linear stored history.
6. Selected: repair the log channel at the existing project-owned SPD storage callback. At FSR3 Luma Pyramid level 5, `value.y` still contains the valid 64x64 linear-luma reduction. Store `log(max(value.y, 6.10e-5))` in `value.x` while preserving `value.y`. Higher SPD levels then average valid per-tile log luminance and retain the arithmetic scene-average channel. For a uniform scene this exactly reproduces the documented estimator; for a non-uniform scene it is the geometric mean of 64x64 tile arithmetic means.

The selected design also makes FSR exposure deliberately frame-local: `LoadFrameInfo()` supplies the `1.0e4` no-smoothing sentinel only to the Luma Pyramid update, while the stored `frameInfo.y` remains the current corrected log value for diagnostics. This skips the imported temporal smoothing branch without changing protected source.

#### Exact T02 implementation contract

T02 changes only project-owned files:

1. `shaders/techniques/ffx/fsr3upscaler/Integration.glsl`
   - Return raw `global_fsr3FrameInfo.x` from `Exposure()`.
   - Keep `DeltaPreExposure() == 1.0`.
   - In `StorePyramid`, only for `FSR3_BIND_LUMA_PYRAMID` and level 5, replace `value.x` with `log(max(value.y, 6.10e-5))` before storing.
   - Make `LoadFrameInfo()` return reset/current data with only its local `.y` changed to `1.0e4`, disabling FSR exposure smoothing every frame.
   - Make read-only `FrameInfo()` return raw `global_fsr3FrameInfo`, not the reset-substituting writer view.
2. `shaders/techniques/ffx/fsr3upscaler/README.md`
   - Document current-frame internal exposure, the level-5 vendor-safe correction, scene-linear history, and display-exposure separation.
3. `FSR3_TASKS.md`
   - Record implementation and validation evidence.

No new image, SSBO field, pass, program-order change, option, generated output, or generator run is required. The existing order `PrepareInputs -> LumaPyramid -> remaining FSR3 stages` guarantees the current exposure is written before its readers.

For a uniform scene, the selected result is:

```text
E(L) = 1 / (9.6 * max(L, 6.10e-5))
```

Numeric gates:

| Uniform luminance `L` | Expected `E(L)` |
|---:|---:|
| `0` | `1707.650273` |
| `0.000001` | `1707.650273` |
| `0.000061` | `1707.650273` |
| `0.001` | `104.1666667` |
| `0.18` | `0.5787037037` |
| `1` | `0.1041666667` |
| `10` | `0.01041666667` |
| `100` | `0.001041666667` |
| `65504` | `0.000001590233675` |

Every result is finite and positive. The three distinct sub-1 cases `0.001`, `0.18`, and `1` must remain distinct. By contrast, the uncorrected upstream expression produces approximately `0.1041603127` for every `0 <= L <= 1`.

### FSR3-T02 — Implement independent FSR3 exposure

Status: `DONE`
Dependencies: FSR3-T01
Commit: this task's commit

Scope:

- Implement exactly the T01 design in project-owned integration/pass/resource code.
- Connect the chosen independent exposure to every intended FFX stage.
- Preserve scene-linear input/history/output domains.
- Make dark luminance distinguishable without modifying protected vendor source.
- Apply shaderpack display exposure exactly once before Bloom/display processing.
- Handle first frame, shader reload, Off/TAA/FSR3 switch, resize, render-scale change, teleport/camera cut, and history reset.

Required checks:

- Static reader/writer trace proves independent exposure is live.
- Numeric cases cover luminance below, equal to, and above 1.
- Exposure stays finite and positive.
- Protected vendor source is identical to baseline.
- Vibris shader load and error query.
- Target Minecraft/Iris compile.
- Generator only when maintained inputs require it.
- Diff and staged-diff checks.

### FSR3-T03 — Define the reversible color transform and add regression coverage

Status: `DONE`
Dependencies: FSR3-T02
Commit: this task's commit

Scope:

- Identify the matrix/log portion that can serve as a truly reversible AA working transform.
- Separate optional lossy Bloom highlight compression from that transform.
- Place lossy compression at one intentional documented stage.
- Ensure the option named Bloom Highlight Compression does not silently contaminate an allegedly reversible AA roundtrip.
- Add or extend a compact test under existing `scripts/test` conventions when feasible; do not introduce a framework for one test.

Required numeric set:

```text
neutral gray, pure red, pure green, pure blue, and mixed saturated colors
at 0.001, 0.18, 1, 100, 1024, 4096, and a near-FP16-maximum value
```

Acceptance:

- Reversible path roundtrips finite documented-domain inputs within an explicit tolerance.
- Pure saturated values do not gain other channels beyond numeric tolerance.
- Lossy highlight compression is tested independently and its behavior is intentional.
- Off/TAA/FSR3 call sites remain explicit.
- Protected vendor source is unchanged.

### FSR3-T04 — Correct shared RCAS and zero-sharpness behavior

Status: `DONE`
Dependencies: FSR3-T03
Commit: this task's commit

Scope:

- Make RCAS operate in the chosen consistent working domain.
- Define and implement sharpness zero behavior. Unless a separate intentional post-AA color operation is documented, zero must be a spatial and color identity path.
- Apply exposure and reversible transforms exactly once.
- Keep Bloom after upscale/AA at output resolution.
- Keep one shared RCAS path; do not restore separate FSR3/TAA implementations or compatibility aliases.
- Verify Off, TAA, and FSR3 branches.

Acceptance:

- Sharpness zero passes the agreed identity test.
- Default sharpness does not create numeric channel leakage on saturated edges.
- Bloom and display transform receive the intended exposed-linear domain.
- Protected vendor source remains unchanged.

### FSR3-T05 — Resolve the SPD/debug atomic-counter collision

Status: `DONE`
Dependencies: FSR3-T02
Commit: this task's commit

Scope:

- Inventory every `global_atomicCounters` index and lifetime.
- Give FSR3 SPD and SST step debug independent state, or prove and enforce non-overlap through project-owned integration/reset ordering.
- Update producers, consumers, clears, and relevant comments as one contract.
- Do not modify vendor SPD code.

Acceptance:

- FSR3 and `SETTING_DEBUG_SST_STEPS` can be enabled together safely.
- All indices remain in bounds and reset at the correct lifetime.
- No unrelated SSBO layout changes.

### FSR3-T06 — Perform static integration review and cleanup

Status: `DONE`
Dependencies: FSR3-T02, FSR3-T03, FSR3-T04, FSR3-T05
Commit: this task's commit

Review and cleanup:

- Trace color/exposure domains after every pass from `TAAPrepare` through `PostComposite`.
- Verify exposure is neither omitted nor applied twice.
- Verify current/history/reset behavior.
- Verify render/upscale sizes and output-resolution post-processing.
- Verify G-buffer gradient/LOD bias across supported render scales.
- Verify motion, reactive/composition masks, translucent handling, atlas formats, bounds, and lifetimes.
- Verify Off/TAA/FSR3 program conditions.
- Verify generated-source parity.
- Verify protected vendor source is unchanged from baseline.
- Remove dead callbacks, resources, aliases, and stale comments made obsolete by completed fixes.
- Synchronize maintained English and Simplified Chinese module documentation.

Acceptance:

- No blocking static findings remain.
- Only required cleanup/docs/ledger evidence is committed.
- Generator and diff checks pass.

### FSR3-T07 — Validate target compile, resets, and night-scene image quality

Status: `DONE`
Dependencies: FSR3-T06
Commit: this task's commit

Target decision:

- On 2026-08-09 the user explicitly selected the already-running Minecraft 1.21.11/Iris client as the target and rejected waiting for 1.21.5.
- Earlier 1.21.5 labels in task evidence were stale assumptions; the retained captures were produced by 1.21.11 and are recorded below as target evidence.

Completed target validation:

- AP8 source `d7cd20bb` passed the complete ten-case Off/TAA/FSR3 sharpness/compression matrix with no shader errors at `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\65ae390d-8efb-3a2d-9930-595a0dbb9d6f`.
- Reload/reset frames 1, 2, 4, 8, 16, 32, and 64 were captured at `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\70343481-936a-3525-9d39-1e226141e6ca` and `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\d2f373ad-ba70-39f9-8a04-d84e36936389`; still-camera frames 16-64 converged without flash or stale history.
- Off-to-FSR3 and FSR3-to-TAA-to-Off switches passed at `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\98469671-2588-3373-9caf-09723dfb4a14` and `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\202cd929-45e3-30da-8829-1c96d326f4ae`.
- Render-scale changes 65-to-50 percent and 50-to-100 percent passed without stale-size borders or incorrect Bloom scale at `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\32d30b9e-ec0c-3b6f-a148-5781f634a1f9` and `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\a1a23def-9e8b-355c-a50c-4f95c2b61709`.
- Auto-exposure dark-to-bright and bright-to-dark transitions were captured at `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\a1492b14-7ad6-3103-afde-78e978676abf` and `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\07b23c1c-60f4-314c-905c-35e3c6aa521d` without exposure flash or old-scene history overlay.
- Sky, cloud, vegetation, water, translucent, mushroom, emissive, GI-lit, teleport, and camera-cut diagnostics are retained at `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\33063ede-7d7d-3aa3-a817-a3cbec090996` and `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\5720ad52-13f2-352e-b3df-061ba938405c`. Visual inspection found no white leakage, desaturated outline, ringing, ghosting, flicker, or incorrect Bloom scale.
- Current AP8 HEAD `c4d38edd` loaded in the 1.21.11 target with `pack_loaded: true`, zero errors, and zero diagnostics. The exact `night-gi-1`, FSR3, 65-percent configuration, frame-64 screenshot, frame-info buffer, shader log, and provenance are retained at `R:\vibris\artifacts\73537349-650c-4cda-b65b-7d974491b86e\0233cead-6817-3be3-9cb4-8ac0d7acafc1`.
- Display auto exposure versus manual EV 3 was compared at `R:\vibris\artifacts\40dfef5c-5804-4608-9ff9-6eee076d7ad4\a618e52b-e6c3-3f42-a14b-ffc47be2acbd`. Display exposure changed from `2167.96` to `8.0`, while FSR exposure changed only from `0.16157846` to `0.16246067`; both matched `1 / (9.6 * exp(frameInfo.y))` within `1.9e-8`.
- A real window resize from a 1920x1080 client to 1280x720 and back passed at `R:\vibris\artifacts\73537349-650c-4cda-b65b-7d974491b86e\2c77de24-fdf4-3f13-b1a4-f0682f6022f4` and `R:\vibris\artifacts\73537349-650c-4cda-b65b-7d974491b86e\71146ab2-d80c-30b3-beb1-9ab1b674f1d5`; Vibris read back both exact sizes, shader inspection stayed clean, and visual inspection found no stale-size border, old history, flash, or incorrect Bloom scale.
- Ten frames of continuous camera rotation across saturated emissives and red, cyan, and yellow edges are retained at `R:\vibris\artifacts\73537349-650c-4cda-b65b-7d974491b86e\89baa9c8-cc39-30ce-a286-f67c58b1779f`. The view changed continuously and returned toward the saved view; visual inspection found no white leakage, desaturated trail, ringing, ghosting, flicker, or exposure flash, and final shader inspection remained clean.

Primary scene:

```text
Vibris preset ID: night-gi-1
save: craftcollection2
dimension: overworld
FOV: 70
nominal resolution: 1920x1080
```

Minimum matched matrix:

| Case | AA mode | Sharpness | Highlight compression | Exposure | Purpose |
|---|---:|---:|---|---|---|
| A | FSR3 | 0 | 0 | fixed/manual | zero-sharpness identity baseline |
| B | FSR3 | 0 | default | fixed/manual | isolate compression-only behavior |
| C | FSR3 | default | 0 | fixed/manual | isolate RCAS behavior |
| D | FSR3 | default | default | fixed/manual | normal configuration |
| E | TAA | 0 and default | 0 and default | fixed/manual | compare shared RCAS path |
| F | Off | N/A | matched | fixed/manual | native reference |

Temporal cases:

- Frames 1, 2, 4, 8, 16, 32, and 64 after reset/reload.
- Still-camera convergence.
- Camera motion across saturated emissive silhouettes.
- Dark-to-bright and bright-to-dark transitions.
- Shader reload and Off/TAA/FSR3 switching.
- Render-scale switch, resize, teleport, and camera cut.
- Sky, cloud, vegetation, translucent, mushroom/emissive, and GI-lit edges.

Evidence:

- Target Minecraft 1.21.11/Iris compile result and shader errors.
- Exact configuration for every case.
- Matched screenshots/captures.
- Relevant exposure/frame-info data when useful.
- Explicit checks for white leakage, desaturated outline, ringing, ghosting, flicker, flash, stale history, and incorrect Bloom scale.

Use Vibris MCP for loading, runtime diagnostics, captures, comparisons, and runtime data. It does not replace final target-Iris/user visual judgment.

If a case fails, leave T07 `READY`, insert one remediation task before it, complete that task in its own later turn/commit, and rerun the affected matrix.

### FSR3-T08 — Measure FSR3 performance and regressions

Status: `READY`
Dependencies: FSR3-T07
Expected commit: measured evidence in this ledger; performance fixes require inserted tasks

Scope:

- Use identical scene, resolution, warmup, frame count, and settings.
- Measure Off, TAA, and FSR3 at relevant render scales.
- Record total GPU time and FSR3/RCAS/Bloom pass costs.
- Repeat runs and report median plus variability.
- Investigate only material regressions; do not optimize speculatively.
- Use Nsight only when Vibris cannot explain a material GPU regression.

Acceptance:

- Measurements are comparable and reproducible.
- No unexplained material regression remains.
- Accepted cost/quality tradeoffs are documented.

### FSR3-T09 — Finalize documentation and manual-acceptance handoff

Status: `PENDING`
Dependencies: FSR3-T07, FSR3-T08
Expected commit: final docs/ledger cleanup

Scope:

- Confirm every inserted remediation task is complete.
- Synchronize maintained English and Simplified Chinese FSR3/pipeline documentation.
- Ensure README paths and integration comments match final behavior.
- Remove task-time debug code and unneeded artifacts.
- Confirm protected vendor source is unchanged from baseline.
- Run generators only when maintained inputs require them.
- Run final diff, staged-diff, repository-status, and target-Iris checks.
- Record the final commit series and any remaining manual-acceptance notes.

Acceptance:

- All implementation, static, runtime, and performance tasks are `DONE`.
- Repository is clean after the task commit.
- The result is ready for user manual acceptance.

## Dependency order

```text
T00
  -> T01 -> T02 -> T03 -> T04
                 -> T05
                 -> T06 -> T07 -> T08 -> T09
```

The serial goal executes numeric task order even where dependencies permit parallel work.

## Global acceptance checklist

- [x] Independent FSR3 exposure is live and not unintentionally driven by display fade.
- [x] Dark-scene luminance produces distinct finite positive exposure values.
- [x] Current/history exposure equations are proven across adaptation/reset.
- [x] HDR FSR input remains scene-linear unless an alternative is fully derived and explicitly approved.
- [x] The reversible AA transform is actually reversible over its documented range.
- [x] Lossy highlight compression is separated and intentionally placed.
- [x] Sharpness zero has the agreed identity behavior.
- [x] Off, TAA, and FSR3 share consistent RCAS/Bloom contracts.
- [x] FSR3 SPD and SST debug do not share atomic synchronization state.
- [x] Render/upscale sizes, G-buffer LOD bias, reactive masks, atlas bounds, and history reset are correct.
- [x] Generated outputs match maintained sources.
- [x] `night-gi-1` shows no white leakage, desaturated outline, ringing, flash, or unstable edge.
- [x] Target Minecraft 1.21.11/Iris compilation passes.
- [ ] Vibris performance comparison is recorded with no unexplained material regression.
- [x] English and Simplified Chinese documentation are synchronized.
- [x] Imported AMD FFX implementation remains byte-identical to baseline.
- [x] Every task commit passes `git diff --check` and `git diff --cached --check`.

## Copyable continuation prompt

> Work only in `I:\code\mcshaders\Alpha-Piscium-8` on `1.10/fsr3`. Continue the active FSR3 goal. First read `I:\code\mcshaders\Alpha-Piscium-8\FSR3_TASKS.md` completely, verify worktree/branch/status, then execute exactly the first `READY` task. Do not start another task in the same turn. Preserve user changes, do not modify imported AMD FFX implementation source, run all task gates, update the ledger, and create one atomic commit whose subject includes the task ID. Leave the goal active for the next continuation.

## Task result log

Append concise task evidence here only when the task section is insufficient. Keep raw screenshots, captures, logs, and binaries outside Git unless they are intentional maintained project artifacts.

### FSR3-T00

- Checked out `1.10/fsr3` in Alpha-Piscium-8.
- Created this serial execution ledger from the complete prior review handoff.
- Reverified at branch tip that the exposure callback, dark-luma expression, RCAS wrapper, and AgX highlight-compression behavior remain unchanged from the reviewed snapshot.

### FSR3-T01

- Confirmed the suspicious dark-luma expression is byte-for-byte present in AMD revision `60f4ea8`; the local port did not introduce it.
- Verified upstream `deltaPreExposure = currentPreExposure / previousPreExposure`; the shaderpack's unit pre-exposure makes `DeltaPreExposure() == 1` exact.
- Traced all exposure, frame-info, luma, color-history, and output readers/writers and derived first-frame, stable, changing-exposure, reset, and display-adaptation equations.
- Selected the existing project-owned level-5 SPD callback correction plus frame-local exposure, with no new resource or pass and no vendor-source edit.
- Evaluated uniform luminance from zero through FP16 maximum; the selected exposure remains finite and positive and distinguishes useful dark-scene values.
- Changed no runtime shader, generator, generated output, or protected vendor file in this task.

### FSR3-T02

- `Exposure()` now reads the frame-local FSR3 exposure, while `DeltaPreExposure()` remains one and scene-linear input/history/output storage is unchanged.
- The project-owned level-5 luma-pyramid callback rebuilds log luminance from the unaffected linear-luminance channel; frame-info reads disable the SDK temporal exposure smoothing without changing imported AMD source.
- Uniform-luminance checks from `0` through `65504` matched `E(L) = 1 / (9.6 * max(L, 6.10e-5))`; all results were finite and positive, including distinct results for `0.001`, `0.18`, and `1`.
- Diffed against baseline `0a8f7843`: protected FSR3, FSR1, SPD, and FFX core implementation files were unchanged; only project-owned FSR3 Integration and README changed under `shaders/techniques/ffx`.
- Vibris case `ap8-t02--fsr3-65` loaded AP8 validation snapshot `267e8062cec8d0fcf27244cb90a26c8785ee4ddd` with FSR3 at 65% render scale in Minecraft 1.21.11/Iris; load and `inspect_shader` both returned `status: ok`, `pack_loaded: true`, with no errors or diagnostics.
- No generator was run because no maintained generator input changed.

### FSR3-T03

- Restricted `AgxInvertible` to the reversible AgX inset-matrix/log transform and expanded its log range to 33 EV, covering nonnegative RGB whose maximum channel is `0.001` through `65504` without hitting either encoded boundary.
- Removed Bloom highlight compression and its color-space dependencies from the AA transform. Compression now runs exactly once on exposed-linear main-color samples entering Bloom downsample level 1; it changes only the Bloom contribution and never the main image or AA history/RCAS roundtrip.
- Added `scripts/test/agx-invertible-check.main.kts`. Its FP32 matrix/log cases cover gray, pure red/green/blue, and three saturated mixed colors at `0.001`, `0.18`, `1`, `100`, `1024`, `4096`, and `65504`, with absolute tolerance `5e-7` and relative tolerance `4e-6`.
- The numeric test reported maximum roundtrip error `0.0859375` and maximum pure-channel leakage `0.005859375`; all values were finite and within tolerance. Independent compression checks covered strengths 0–4, RGB ratio change, Luma ratio preservation, and black input without division by zero.
- Updated paired English and Simplified Chinese post-processing documentation and removed the now-unused `AgxInvertible` include from Post Composite.
- Vibris validation snapshot `67f0e8c31875aa00044fc50aee7e4d424892bca4` passed all four Minecraft 1.21.11/Iris cases: Off/RGB, TAA/RGB, FSR3/RGB, and FSR3/Luma-low. Every load and `inspect_shader` returned `status: ok`, `pack_loaded: true`, with no errors or diagnostics (`passed: 4`, `failed: 0`).
- Protected AMD FSR3, FSR1, SPD, and FFX core sources remained byte-identical to baseline `0a8f7843`. No generator was run because no maintained generator input changed.

### FSR3-T04

- Unified TAA and FSR3 on `SETTING_AA_SHARPNESS` through one `SETTING_RCAS_SHARPNESS` integration macro; removed the mode-specific TAA/FSR3 aliases and the TAA motion-dependent sharpness override.
- Defined sharpness `0` as the shared spatial/color identity path: both TAA and FSR3 return the center sample without evaluating the RCAS neighborhood filter, while Off continues to omit the RCAS pass entirely.
- Kept both active modes in the same reversible AgX matrix/log working domain. Static call-site and program-order review confirmed display exposure and reversible transforms occur once, then Bloom consumes exposed-linear main color after AA/upscale at output resolution.
- Extended `scripts/test/agx-invertible-check.main.kts` with the shared zero-sharpness roundtrip, default-sharpness pure-primary edge cases, and static integration assertions. It reported maximum zero-sharpness error `0.0859375` and maximum pure-channel leakage `0.021972656`, finite and within absolute tolerance `5e-7` plus relative tolerance `4e-6`.
- Vibris validation snapshot `54f069ac030e2c3f0cb95983d7605b5d287f0cca` passed all five Minecraft 1.21.11/Iris cases: Off/zero, TAA/zero, TAA/default, FSR3/zero, and FSR3/default. Every load and `inspect_shader` returned `status: ok`, `pack_loaded: true`, with no errors or diagnostics (`passed: 5`, `failed: 0`).
- Updated paired English and Simplified Chinese post-processing documentation. Protected AMD FSR3, FSR1, SPD, and FFX core sources remained byte-identical to baseline `0a8f7843`; no generator was run because no maintained generator input changed.

### FSR3-T05

- Inventoried every `global_atomicCounters` access. Generic SPD indexes directly by dispatch slice: HiZ and Exposure use slice 0, while the three-layer GI mip dispatch uses slices 0–2. Slots 0–13 are now reserved for generic SPD, slot 14 is dedicated to FSR3 SPD, and slot 15 is dedicated to SST step debugging; every named index is within the unchanged 16-element array.
- Replaced all FSR3 and SST literal indices with shared constants. Both FSR3 pyramid callbacks use only slot 14; the conditional SST producer plus `SSTStepDebug` and final debug-text consumers use only slot 15.
- Verified lifetimes: `UpdateGlobalData` clears all 16 counters at frame begin, generic SPD resets each active slice after its final workgroup, each sequential FSR3 pyramid pass resets slot 14 after its final workgroup, and FSR3 no longer modifies the frame-local SST value in slot 15.
- Vibris validation snapshot `10f4e04491c7f70f3ac1e6c97cd9b3aaa123f2f9` passed Minecraft 1.21.11/Iris with FSR3 at 65% render scale for both `SETTING_DEBUG_SST_STEPS` disabled and enabled. Every load and `inspect_shader` returned `status: ok`, `pack_loaded: true`, with no errors or diagnostics (`passed: 2`, `failed: 0`).
- The global SSBO member order, member sizes, and 16-counter layout were unchanged. Protected AMD FSR3, FSR1, SPD, and FFX core sources remained byte-identical to baseline `0a8f7843`; only project-owned integration and README files changed under `shaders/techniques/ffx`. No generator was run because no maintained generator input changed.

### FSR3-T06

- Traced every pass from `TAAPrepare` through `PostComposite`. FSR3 receives unexposed scene-linear HDR; its frame-local reconstruction exposure scales current and scene-linear history equally and is divided out before storage. Shared RCAS then applies display exposure exactly once, returns exposed-linear `main`, and output-size Bloom/Post Composite follow it.
- Verified reset, motion, jitter, depth, reactive/composition, and translucent contracts. History loaders reject reset frames, the RCAS completed-frame marker rejects interrupted mode sequences, motion is current-to-previous UV, and untracked dynamic/translucent surfaces receive reactive/composition coverage.
- Verified Iris CPU-side render/view sizes, output-resolution post-processing, image formats, four render-history tiles, three output-width atlas regions, FSR SPD packing, and Bloom mip packing/lifetimes. The G-buffer gradient factor is algebraically equivalent to AMD's `log2(render/output) - 1` mip bias for every supported 50–100% scale.
- Fixed the one blocking atlas finding: packed Bloom downsample reads now clamp to source-tile texel centers, matching upsample and preventing bilinear reads from crossing into adjacent or previous-frame atlas RGB. Representative packed extents include `961x812` within `1920x1080`; atlas bounds passed every supported scale in representative views, and FSR/Bloom packing passed exhaustive square sizes `64..8192`.
- Removed the unused `FSR3UPSCALER_BIND_SRV_FRAME_INFO` integration define and `FrameInfo()` callback. The only SDK helper they enabled, `SceneAverageLuma()`, has no call site; live `LoadFrameInfo()`/`StoreFrameInfo()` exposure callbacks remain.
- Corrected the FSR3 README accumulate workgroup from `8x8` to `16x8` and synchronized English/Simplified Chinese size, exposure, history, LOD, mask, atlas, and Bloom-boundary contracts.
- Bootstrapped the ignored Shadesmith fragment in this fresh worktree and ran `kotlin options.main.kts`; complete generation passed and left no tracked generated diff. `kotlin test/agx-invertible-check.main.kts` passed AgX, Bloom compression, and shared RCAS checks.
- Vibris loaded AP8 validation snapshot `b458aff921a1f7829558fefa67128a9745d61e67` in Minecraft 1.21.11/Iris with FSR3 at 65% render scale. `load_shader` and `inspect_shader` both returned `status: ok`, `pack_loaded: true`, with no errors or diagnostics. T07 retains the reset and night-scene visual matrix.
- Diffed all FFX files from baseline `0a8f7843`: changes are restricted to project-owned FSR1 RCAS, FSR3 Integration, and README; imported AMD FSR3, FSR1, SPD, and FFX core implementation remains unchanged.

### FSR3-T07

- Retargeted final runtime validation to Minecraft 1.21.11/Iris exactly as requested by the user and corrected stale 1.21.5 labels in earlier runtime evidence.
- Loaded current AP8 HEAD `c4d38edd` in `night-gi-1` at 1920x1080, FOV 70, FSR3 65 percent, sharpness `0.5`, RGB highlight compression `3`; load and inspection passed with zero errors or diagnostics.
- Revalidated the complete matched Off/TAA/FSR3 matrix, reset frames 1-64, convergence, mode and scale switches, exposure transitions, teleport/cut, translucent content, vegetation, sky/cloud, mushroom, emissive, and GI-lit edges from retained 1.21.11 artifacts.
- Confirmed FSR frame-local exposure remained independent while display exposure changed from approximately `2168` to `8`; all sampled frame-info values were finite and matched the integration equation within `2.4e-8`.
- Performed a real 1920x1080 to 1280x720 to 1920x1080 window resize. Both exact framebuffer sizes were captured cleanly, and the original window bounds were restored.
- Captured ten frames during continuous camera rotation across saturated night-scene emissives. Visual inspection found no white leakage, desaturated outline/trail, ringing, ghosting, flicker, flash, stale history, or incorrect Bloom scale.
- No shader or generated file changed in T07; raw artifacts remain outside Git.
