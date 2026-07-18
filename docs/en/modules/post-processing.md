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
| [`FXAA.comp.glsl`](../../../shaders/pass/composite/FXAA.comp.glsl)                                                                                                               | Spatial antialiasing                              |
| [`RCAS.comp.glsl`](../../../shaders/pass/composite/RCAS.comp.glsl), [`techniques/ffx/fsr1/`](../../../shaders/techniques/ffx/fsr1/)                                              | RCAS sharpening                                   |
| [`techniques/Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl)                                                                                                      | Bloom downsample/upsample pyramid                 |
| [`ExposureMip.comp.glsl`](../../../shaders/pass/composite/ExposureMip.comp.glsl), [`ExposureGather.comp.glsl`](../../../shaders/pass/composite/ExposureGather.comp.glsl)         | Auto-exposure weights, mip, and statistics        |
| [`PostComposite.comp.glsl`](../../../shaders/pass/composite/PostComposite.comp.glsl), [`OverlayComposite.comp.glsl`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | Main post composition and overlay                 |
| [`techniques/displaytransform/`](../../../shaders/techniques/displaytransform/)                                                                                                  | Exposure, DRT, and display transform              |
| [`FinalGlobalDataUpdate.comp.glsl`](../../../shaders/pass/composite/FinalGlobalDataUpdate.comp.glsl), [`Final.frag.glsl`](../../../shaders/pass/composite/Final.frag.glsl)       | Next-frame state and final screen output          |

## DOF

[`DOFFocus`](../../../shaders/pass/composite/DOFFocus.comp.glsl) runs only when DOF is enabled without manual focus.
After volumes/translucency, [`DOFPrepare`](../../../shaders/pass/composite/DOFPrepare.comp.glsl) writes the current main
color to `transient_dofInput` for later post composition. Manual focus skips only the focus pass; preparation still
runs.

Settings cover focal length, f-stop, aperture shape, quality, maximum sample radius, masking heuristic, three-part
manual-focus distance, focus time, and focus-plane debug.

## Temporal and spatial AA

| Order | Pass                                                                 | Purpose                                                                                              |
|-------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| 1     | [`TAAPrepare`](../../../shaders/pass/composite/TAAPrepare.comp.glsl) | Produces temporal input, luma difference, and supporting data                                        |
| 2     | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) | Uses `history_taa`, motion/reprojection data, and current-frame input to write `transient_taaOutput` |
| 3     | [`FXAA`](../../../shaders/pass/composite/FXAA.comp.glsl)             | Writes `transient_fxaaOutput`                                                                        |
| 4     | [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl)             | Performs final spatial sharpening                                                                    |

All four passes are present in the program list; settings select behavior inside the shaders rather than
enabling/disabling separate program numbers.

TAA settings cover enable, jitter, current/history filters, and CAS sharpness. Current/previous-jitter custom uniforms
are generated from the R2 frame sequence in [`scripts/shaders.properties`](../../../scripts/shaders.properties);
changing the sampling sequence requires updating reprojection conventions as well.

## Bloom

When `SETTING_BLOOM` is enabled, [`Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl) builds levels 1–10
with `BLOOM_DOWN_SAMPLE` and `BLOOM_PASS=1..10`, then reconstructs levels 10–2 with `BLOOM_UP_SAMPLE`.
`SETTING_BLOOM_PASS` disables unused high levels at the program layer, and the resource is `transient_bloom`.

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

Related settings cover material transfer/color space, internal working space, DRT working space, tone-map look/dynamic
range/offset/slope/power/saturation, and output color space/transfer function.

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
