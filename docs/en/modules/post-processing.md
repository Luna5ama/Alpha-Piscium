# Post-processing and Display Transform

Language: English | [简体中文](../../sc/modules/post-processing.md)

Post-processing begins after GI, air/water volumes, and translucency are composed. The main chain is optional DOF,
temporal resolve, spatial antialiasing/sharpening, bloom, post composition, exposure, overlay, and final display
transform.

## Code map

| Path                                                                                                                                                                             | Responsibility                                    |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| [`shaders/techniques/DOF.glsl`](../../../shaders/techniques/DOF.glsl)                                                                                                            | Shared DOF sampling and circle-of-confusion logic |
| [`DOFFocus.comp.glsl`](../../../shaders/pass/composite/DOFFocus.comp.glsl), [`DOFPrepare.comp.glsl`](../../../shaders/pass/composite/DOFPrepare.comp.glsl)                       | Automatic focus and DOF input preparation         |
| [`TAAPrepare.comp.glsl`](../../../shaders/pass/composite/TAAPrepare.comp.glsl), [`TAAResolve.comp.glsl`](../../../shaders/pass/composite/TAAResolve.comp.glsl)                   | Temporal-AA preparation and resolve               |
| [`GenerateMotionVectors.comp.glsl`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl), [`FSR3Accumulate.comp.glsl`](../../../shaders/pass/composite/FSR3Accumulate.comp.glsl) | Shared motion generation, internal FSR3 inputs, and accumulation |
| [`FXAA.comp.glsl`](../../../shaders/pass/composite/FXAA.comp.glsl)                                                                                                               | Spatial antialiasing                              |
| [`RCAS.comp.glsl`](../../../shaders/pass/composite/RCAS.comp.glsl), [`techniques/ffx/fsr1/`](../../../shaders/techniques/ffx/fsr1/)                                              | RCAS sharpening                                   |
| [`techniques/Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl)                                                                                                      | Bloom downsample/upsample pyramid                 |
| [`ExposureMip.comp.glsl`](../../../shaders/pass/composite/ExposureMip.comp.glsl), [`ExposureGather.comp.glsl`](../../../shaders/pass/composite/ExposureGather.comp.glsl)         | Auto-exposure weights, mip, and statistics        |
| [`PostComposite.comp.glsl`](../../../shaders/pass/composite/PostComposite.comp.glsl), [`OverlayComposite.comp.glsl`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | Main post composition and overlay                 |
| [`superresolution.v3.json`](../../../shaders/superresolution.v3.json)                                                                                                               | External Super Resolution interface and trigger  |
| [`techniques/displaytransform/`](../../../shaders/techniques/displaytransform/)                                                                                                  | Exposure, DRT, and display transform              |
| [`FinalGlobalDataUpdate.comp.glsl`](../../../shaders/pass/composite/FinalGlobalDataUpdate.comp.glsl), [`Final.frag.glsl`](../../../shaders/pass/composite/Final.frag.glsl)       | Next-frame state and final screen output          |

## DOF

[`DOFFocus`](../../../shaders/pass/composite/DOFFocus.comp.glsl) runs only when DOF is enabled without manual focus.
After volumes/translucency, [`DOFPrepare`](../../../shaders/pass/composite/DOFPrepare.comp.glsl) writes the current main
color to `transient_dofInput` for later post composition. Manual focus skips only the focus pass; preparation still
runs.

Settings cover focal length, f-stop, aperture shape, quality, maximum sample radius, masking heuristic, three-part
manual-focus distance, focus time, and focus-plane debug.

## Anti-aliasing and super resolution

[`TAAPrepare`](../../../shaders/pass/composite/TAAPrepare.comp.glsl) applies the common DOF input before the pipeline
branches. The program list then enables one of these paths:

| Path | Pass flow | Purpose |
|------|-----------|---------|
| Off | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) | Writes the unfiltered current frame without temporal or spatial AA. |
| TAA | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) → [`FXAA`](../../../shaders/pass/composite/FXAA.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | Resolves `history_taa`, applies spatial AA, and sharpens the render-resolution output. |
| FSR 3 | [`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) → [`FSR3PrepareInputs`](../../../shaders/pass/composite/FSR3PrepareInputs.comp.glsl) → [`FSR3LumaPyramid`](../../../shaders/pass/composite/FSR3LumaPyramid.comp.glsl) → [`FSR3ShadingChangePyramid`](../../../shaders/pass/composite/FSR3ShadingChangePyramid.comp.glsl) → [`FSR3ShadingChange`](../../../shaders/pass/composite/FSR3ShadingChange.comp.glsl) → [`FSR3PrepareReactivity`](../../../shaders/pass/composite/FSR3PrepareReactivity.comp.glsl) → [`FSR3LumaInstability`](../../../shaders/pass/composite/FSR3LumaInstability.comp.glsl) → [`FSR3Accumulate`](../../../shaders/pass/composite/FSR3Accumulate.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | Builds motion/reactive inputs, accumulates a full-resolution result, then uses the shared RCAS pass for sharpening and exposed-linear output. |
| External SR | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) → [`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) → normal render-resolution post-processing → [`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) → external SR | Bypasses internal TAA, FXAA, FSR3 accumulation, and RCAS; SR reads the completed display-referred frame and produces the full-resolution output. |

The Anti-Aliasing / Super Resolution screen controls the mode, render scale, jitter, TAA current/history filters,
and the shared RCAS sharpening strength. Current/previous-jitter custom uniforms
are generated from the R2 frame sequence in [`scripts/shaders.properties`](../../../scripts/shaders.properties);
changing the sampling sequence requires updating reprojection conventions as well.

When `SR_ENABLE` is active, the external interface owns render scale and jitter through `SR_RENDER_SCALE_FACTOR` and
`SRJitterOffset`. Internal TAA, FXAA, FSR3 accumulation, and RCAS are disabled regardless of `SETTING_AA_MODE`.
Frame-generation-only remains active through the same interface, but `SR_SHOULD_APPLY_SCALE` and
`SR_SHOULD_APPLY_JITTER` are zero, so the pack renders at native resolution without jitter.

TAA and FSR 3 feed the same reversible matrix/log working domain to the shared RCAS implementation and use the selected
sharpening strength directly. A strength of zero bypasses the spatial filter and returns its center sample; only the
required display exposure and reversible working-domain roundtrip remain. Off does not schedule RCAS. Every mode writes
exposed-linear `main` before output-resolution Bloom and the display transform.

FSR 3 consumes render-size, unexposed scene-linear HDR and stores its color and luma histories in that same scene-linear
domain. Its frame-local reconstruction exposure scales current and history values equally and is divided out before
storage; shared RCAS applies display exposure exactly once afterward. A completed-frame marker and the common temporal
reset state reject stale histories after reloads, mode switches, or other temporal discontinuities. Motion is
current-to-previous in UV units, and jitter is supplied in render-pixel units. The SDK max-channel reconstruction
tonemap is replaced at the project-owned accumulate entrypoint by the reversible AgX inset-matrix/log transform. Its
offset log preserves black without a hard dark-EV clamp, and its FSR range covers FP16 input multiplied by the maximum
frame-local reconstruction exposure.

Reactive/composition masks cover dynamic or otherwise untracked solid surfaces and overlays. Translucent SST is already
composed into the input color and deliberately follows the underlying solid depth, motion, and masks. It has neither a
separate reactive-mask contribution nor a temporal SST denoiser because that combination makes its rough reflection and
refraction noise flash between frames.

[`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) computes the shared
current-to-previous UV vector once. Compile-time output macros route it either to the internal FSR3 history, together
with reactive/composition masks, or to render-resolution `colortex31.rg` for external SR. The external path uses
`colortex0` color, `noTranslucentDepthtex` depth, and `colortex31` motion. It is triggered after
[`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl), after the display transform, so the
declared SR input is SDR with constant pre-exposure `1.0`. SR writes the full-resolution result back to `colortex0`.
In frame-generation-only mode it still consumes all three inputs but deliberately does not write that color output.

The generated vectors cover camera reprojection, sky, hand, and solid surfaces. They do not provide complete
per-object motion for skeletal animation, particles, or arbitrary procedural deformation; those surfaces can therefore
produce external upscaling or frame-generation artifacts. Screen overlays also inherit the underlying scene motion.

The FSR 3 estimator runs at render size and accumulates at view/output size; shared RCAS and all later post-processing also
use output size. Explicit G-buffer gradients multiply low-resolution raster derivatives by
`0.5 * uval_mainImageScale`, which is equivalent to AMD's `log2(render/output) - 1` mip bias for the actual per-axis render
scale.

## Bloom

When `SETTING_BLOOM` is enabled, [`Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl) builds levels 1–10
with `BLOOM_DOWN_SAMPLE` and `BLOOM_PASS=1..10`, then reconstructs levels 10–2 with `BLOOM_UP_SAMPLE`.
`SETTING_BLOOM_PASS` disables unused high levels at the program layer. The non-internal-FSR3 path uses `transient_bloom`; the FSR3
path reuses the third region of `usam_fsr3UpscaleAtlas` after accumulation and RCAS.
All packed-pyramid reads are clamped to source-tile texel centers, preventing bilinear filtering from crossing into an
adjacent tile or stale atlas data.

Bloom highlight compression is an intentionally lossy Bloom-only operation. It is applied exactly once to exposed-linear
main-color samples entering the first downsample level, before pyramid filtering. It does not modify the main image and is
not part of the reversible matrix/log AA working transform in [`AgxInvertible.glsl`](../../../shaders/util/AgxInvertible.glsl).

## Post composition and exposure

| Order | Pass                                                                                                                                                                            | Purpose                                                                                                         |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| 1     | [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl)                                                                                                      | Combines the main color, DOF, bloom, and other post inputs; the RTWSM companion pass owns independent resources |
| 2     | [`ExposureMip`](../../../shaders/pass/composite/ExposureMip.comp.glsl)                                                                                                          | Builds luminance/weight levels                                                                                  |
| 3     | [`ExposureGather`](../../../shaders/pass/composite/ExposureGather.comp.glsl)                                                                                                    | Reduces exposure statistics                                                                                     |
| 4     | [`FinalGlobalDataUpdate`](../../../shaders/pass/composite/FinalGlobalDataUpdate.comp.glsl) and [`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | Persists global/exposure state for the next frame and composes the overlay                                      |

The exposure screen controls manual EV, minimum/maximum EV, emissive/distance/center weighting, average-luminance time
and target range, highlight/shadow percentiles, and adaptation parameters. `transient_exposureWeights` carries per-pixel
weights.

## Final display transform

Root [`final.fsh`](../../../shaders/final.fsh) includes [
`Final.frag.glsl`](../../../shaders/pass/composite/Final.frag.glsl). That entry point uses [
`Exposure.glsl`](../../../shaders/techniques/displaytransform/Exposure.glsl), [
`DRT.glsl`](../../../shaders/techniques/displaytransform/DRT.glsl), and [
`DisplayTransform.glsl`](../../../shaders/techniques/displaytransform/DisplayTransform.glsl) to convert internal working
color space to DRT working space, then applies tone mapping and the output color space/transfer function.

After tone mapping, the result is converted to the configured color-grading color space. [`PrimaryColorCalibration`](../../../shaders/techniques/displaytransform/PrimaryColorCalibration.glsl) adjusts its linear RGB primaries, then the configured grading transfer function is applied and [`HSLColorMixer`](../../../shaders/techniques/displaytransform/HSLColorMixer.glsl) adjusts eight hue bands. The result is decoded and converted to the monitor output space. Both stages have independent toggles.

Related settings cover material transfer/color space, internal working space, DRT working space, color-grading color space/transfer function, tone-map look/dynamic range/offset/slope/power/saturation, and monitor color space/transfer function.

## Debug and special modes

Debug output in [`shaders/techniques/debug/`](../../../shaders/techniques/debug/) can inspect TAA, PostFX, tone-mapping,
or final boundaries; [`SSTStepDebug`](../../../shaders/pass/composite/SSTStepDebug.comp.glsl) is a conditional program.
Screenshot mode adjusts animation/temporal clamps, and video-render mode adjusts long-term temporal effects. They do not
define parallel production post-processing branches.

## Maintenance invariants

- Keep TAA history color space, pre-exposure, jitter, and motion/reprojection aligned.
- Bloom pass count controls both program-enable conditions and pyramid access.
- Update exposure state at frame end; final must not observe a partially updated state.
- Validate display-transform changes numerically with color charts/gradients and across SDR, wide-gamut, and HDR
  configurations for clipping, NaN, and negative values.
- Test static detail, moving edges, disocclusion, bright emissives, dark adaptation, translucent edges, and UI/overlays.
