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
Camera Six (4), Camera Ten (5), Cinema Star (6), Prismatic Streak (7), 67 (8), and Trollface (9).
Standard is an eight-scale, long-tailed Gaussian approximation with a concentrated core. Each scale contributes
1/8 of the normalized filter; it is not a single Gaussian. The fixed eight levels preserve the broad halo that
was lost in the rejected four-level prototype.

The filters use offline-fitted offsets and nonnegative weights in
[`BloomTaps.glsl`](../../../shaders/techniques/BloomTaps.glsl). Fitting accounts for decimation phases and bilinear
sampling, following the [DPF approach](https://huakaifugui2.github.io/CZ-DPF-Paper/). Runtime uses fixed RGB filters
and a fixed pyramid depth. The four-scale fitting proxy calibrates reusable taps; it does not set the runtime depth.

| Order | Pass/stage | Purpose |
|---|---|---|
| 1 | [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl), downsample levels 1–8 | Five bilinear reads per output texel. The first level reads the decoded linear main image and applies Bloom-only highlight compression. |
| 2 | [`BloomShape`](../../../shaders/pass/composite/BloomShape.comp.glsl), optional | Initializes the selected tap table in workgroup shared memory, then samples raw level 1 in a bounded loop and writes a separate half-resolution image. Standard and Bloom Off disable this pass. |
| 3 | [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl), reconstruct levels 8–2 | Eight bilinear reads per output texel. The deepest contribution and each finer downsample receive weight 1/8; the invocation reads the finer texel that it then overwrites. |
| 4 | [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl) | Fuses the last upsample, adds the optional shape with centered bilinear reconstruction, and applies Bloom intensity and underwater boost. |

[`programs.main.kts`](../../../scripts/programs.main.kts) owns this order and its enable conditions.
[`Bloom.glsl`](../../../shaders/techniques/Bloom.glsl) owns common addressing and sampling. Downsample and shape reads
zero-extend their bilinear footprint at screen edges while keeping atlas reads inside the input tile. Upsampling
clamps to the valid input mip. This prevents a long streak from repeatedly copying a bright edge texel.

- `transient_bloom`: frame-local RGBA16F atlas with ceil-sized padded mip tiles; reconstructed in place from coarse
  to fine.
- `transient_bloomShape`: frame-local RGBA16F logical image; only its half-resolution region is written and read.
  It lives from the shape pass through post composition. The allocation in
  [`shadesmith.json`](../../../shaders/shadesmith.json) reuses a slot after FXAA/RCAS have finished with it, without
  increasing the current physical atlas dimensions. It remains a separate logical resource from the pyramid.

Each shape adds energy over the full Standard core and tail. Camera and Prismatic shapes spread each input channel
with its own weights. The two novelty kernels align their visual bounds to the light and are most legible around
small emitters. Larger emitting surfaces broaden the shapes and can fill fine gaps. Parameters and custom
PNG/function input are maintained by
[`fit-bloom-shape.py`](../../../scripts/fit-bloom-shape.py); see [kernel fitting](../development/scripts.md#bloom-kernel-fitting).

Highlight compression belongs to the first Bloom downsample, after AA decoding. RGB and luma modes retain their
existing curves. AA retains its original log encoding and dark-value floor without Bloom rolloff; decoded writes in
[`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) and the TAA-off path of
[`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) clamp to the RGBA16F limit of 65504.

### Bloom validation and limits

Shader snapshot `0ed64baf9a70…` rendered all ten modes and Bloom Off at 1280×720 in Small lamps night (EV 9.45),
outdoor point lights (EV 11.07), and Craft close lights (EV 7.34). Camera and world time were fixed within each
scene, with 192 warmup frames per mode. Camera Six produced finite, nonzero shape output with TAA disabled.
Bloom Off suppressed the first downsample and shape passes with Trollface selected.

On RTX 3080 Ti in Small lamps night, two rounds bracketed the forward and reverse shape orders with Standard.
Each window used 96 warmup frames and 64 sampled frames. GPU times are in milliseconds.

| Mode | Shape pass p50 | Bloom + complete PostComposite p50 / p95 |
|---|---:|---:|
| Standard | — | 0.191 / 0.201 |
| Diamond Four | 0.254 | 0.455 / 0.471 |
| Horizontal Streak | 0.094 | 0.289 / 0.296 |
| Cross Four | 0.193 | 0.386 / 0.396 |
| Camera Six | 0.357 | 0.553 / 1.114 |
| Camera Ten | 0.399 | 0.593 / 1.234 |
| Cinema Star | 0.263 | 0.460 / 0.831 |
| Prismatic Streak | 0.179 | 0.375 / 0.397 |
| 67 | 0.199 | 0.395 / 0.411 |
| Trollface | 0.311 | 0.507 / 0.992 |

The combined column sums the aligned program samples per frame before calculating quantiles. The unchanged
volumetric-composite reference pass had a 0.049 ms median in every mode.

Independent CPU impulse models support a near-radial standard kernel: at radii 12, 24, 48 and 96 pixels, the
eight-phase mean fourfold angular component was about 1.0–1.9% of ring brightness, versus 2.6–3.2% for the previous
COD filter. Dyadic sampling still has pixel-grid and phase error; this is an approximation, not exact rotational
symmetry. Odd viewport sizes, boundary energy and phase centroids have CPU coverage; odd-size runtime, underwater
rendering and other GPUs remain unmeasured. Shape fitting error describes the added raw-level-1 branch, not the
complete Bloom result.

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
