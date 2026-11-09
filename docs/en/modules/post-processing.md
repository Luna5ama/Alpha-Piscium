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
| [`Final`](../../../shaders/pass/composite/Final.frag.glsl) | Final screen output |

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

`SETTING_BLOOM_SHAPE` offers Standard (0, default), Diamond Four (1), Horizontal Streak (2), Cross Four (3),
Camera Six (4), Camera Ten (5), Cinema Star (6), Prismatic Streak (7), 67 (8), Trollface (9), and Custom (10).
Standard combines eight scales with equal weights to form a concentrated core and a long tail. The other presets
fit the complete RGB response, including the intended halo and shape. Each input channel has its own weights for
color dispersion. See [Bloom kernel fitting](../development/scripts.md#bloom-kernel-fitting) for the editor and export.

| Order | Pass/stage | Purpose |
|---|---|---|
| 1 | [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl), downsample levels 1–8 | Five bilinear reads per output texel. Level 1 reads the decoded linear main image and applies Bloom highlight compression per read. |
| 2 | [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl), reconstruct from level 8 to level 1 | Upsample the coarser result, add the selected same-level taps from the raw pyramid, and write the combined result. |
| 3 | [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl) | Upsample level 1 to the viewport, then add Bloom using the intensity and underwater settings before the display transform. |

[`programs.main.kts`](../../../scripts/programs.main.kts) owns the order and enable conditions.
[`Bloom.glsl`](../../../shaders/techniques/Bloom.glsl) confines reads to the input mip tile. Downsampling and raw
injection zero-extend the bilinear footprint at the viewport boundary, and reconstruction clamps to the valid mip.

- `transient_bloomRaw` holds the frame-local RGBA16F downsample pyramid with ceil-sized, padded mip tiles.
- `transient_bloomReconstruction` holds the frame-local RGBA16F results from the coarse-to-fine passes.
  The allocation in [`shadesmith.json`](../../../shaders/shadesmith.json) reuses atlas space after FXAA and RCAS.

Highlight compression runs in the first Bloom downsample, after AA decoding. RGB and luma modes use the same
curves. The decoded writes in [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) and the TAA-off path of
[`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) clamp to the RGBA16F limit of 65504.

### Bloom validation and limits

The CPU fitter measures the complete impulse response across held-out decimation phases. Dyadic sampling creates
phase-dependent variation. Larger emitters broaden the directional patterns and can fill fine gaps. The CPU
estimate counts weighted bilinear reads across the pyramid. GPU cost also depends on the selected tap layout and
the scene. Compare the affected reconstruction passes, PostComposite, and a stable neighboring pass on the same
Iris setup.

## Post composition and exposure

| Order | Pass                                                                                                                                                                            | Purpose                                                                                                         |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| 1     | [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl)                                                                                                      | Combines the main color, DOF, bloom, and other post inputs; the RTWSM companion pass owns independent resources |
| 2     | [`ExposureMip`](../../../shaders/pass/composite/ExposureMip.comp.glsl)                                                                                                          | Builds luminance/weight levels                                                                                  |
| 3     | [`ExposureGather`](../../../shaders/pass/composite/ExposureGather.comp.glsl)                                                                                                    | Reduces exposure statistics                                                                                     |
| 4 | [`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | Composes the overlay |

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
- Keep the eight Bloom levels, layer weights, generated taps and atlas addressing consistent. Refit directional
  shapes if the raw level-1 carrier changes.
- Keep exposure production, adaptation and final display consumers synchronized.
- Validate display-transform changes numerically with color charts/gradients and across SDR, wide-gamut, and HDR
  configurations for clipping, NaN, and negative values.
- Test static detail, moving edges, disocclusion, bright emissives, dark adaptation, translucent edges, and UI/overlays.
