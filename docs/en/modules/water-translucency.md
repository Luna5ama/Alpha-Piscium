# Water, Caustics, and Translucency

Language: English | [简体中文](../../sc/modules/water-translucency.md)

Alpha Piscium separates translucent rasterization, reflection/refraction tracing, water scattering, and final
composition. Water and glass can therefore reuse the solid G-buffer and completed lighting while retaining layered depth
and volumetric results.

## Code map

| Path                                                                                                                                                                                                                                                      | Responsibility                                                     |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [`shaders/pass/geometry/GBufferTranslucent.*.glsl`](../../../shaders/pass/geometry/)                                                                                                                                                                      | Water, transparent surface, weather, and particle raster           |
| Root `gbuffers_water.*`, `gbuffers_hand_water.*`, and `dh_water.*` wrappers                                                                                                                                                                               | Water-path wrappers                                                |
| [`shaders/techniques/WaterWave.glsl`](../../../shaders/techniques/WaterWave.glsl)                                                                                                                                                                         | Shared water-wave normal/displacement calculations                 |
| [`shaders/util/Translucent.glsl`](../../../shaders/util/Translucent.glsl)                                                                                                                                                                                 | Shared translucent data and composition logic                      |
| [`CausticsPhotonTrace`](../../../shaders/pass/composite/CausticsPhotonTrace.comp.glsl), [`CausticsRemap`](../../../shaders/pass/composite/CausticsRemap.comp.glsl)                                                                                        | Screen-space caustics generation                                   |
| [`EpipolarScatteringWater`](../../../shaders/pass/composite/EpipolarScatteringWater.comp.glsl)                                                                                                                                                            | Water epipolar scattering                                          |
| [`TranslucentBackComposite`](../../../shaders/pass/composite/TranslucentBackComposite.glsl), [`TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl), [`TranslucentComposite`](../../../shaders/pass/composite/TranslucentComposite.glsl) | Back-layer composition, translucent tracing, and final composition |
| [`techniques/atmospherics/water/`](../../../shaders/techniques/atmospherics/water/)                                                                                                                                                                       | Underwater-scattering constants and epipolar implementation        |

## Geometry and resources

Translucent geometry writes its packed G-buffer to `colortex11/12`, transmittance to `colortex14`, and CSR R32F-atlas
near/far depth instead of replacing solid lighting immediately. [
`TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl) later writes the following reflection/refraction
tiles consumed by final composition; tile format and lifetime are owned by [
`shaders/shadesmith.json`](../../../shaders/shadesmith.json):

- `transient_translucentReflection` and `transient_translucentRefraction`.
- `transient_translucentZLayer1/2/3`.
- `transient_lmCoord` and `transient_shadow`.
- `transient_caustics_input`, `transient_caustics_final`, and `transient_screenPixelSize`.

Water waves use the custom `usam_waveNoise` (2048×2048 R16) and `usam_waveHFCurl` (512×512 RG16_SNORM) textures declared
in [`scripts/shaders.properties`](../../../scripts/shaders.properties). The water shadow variant and [
`EvaluateShadowWaterNormal`](../../../shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl) provide the later
water-shadow normal.

## Caustics

When `SETTING_WATER_CAUSTICS` is enabled:

| Order | Pass                                                                                           | Purpose                                                                             |
|-------|------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| 1     | [`EvaluateScreenPixelSize`](../../../shaders/pass/composite/EvaluateScreenPixelSize.comp.glsl) | Estimates the trace footprint                                                       |
| 2     | [`CausticsPhotonTrace`](../../../shaders/pass/composite/CausticsPhotonTrace.comp.glsl)         | Generates photon contribution from the water surface and lighting conditions        |
| 3     | [`CausticsRemap`](../../../shaders/pass/composite/CausticsRemap.comp.glsl)                     | Remaps the input to `transient_caustics_final` for later lighting and volume passes |

All three programs are disabled together when the setting is off.

## Volumetrics and final composition

After GI denoising:

| Order | Pass                                                                                           | Purpose                                                                                  |
|-------|------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 1     | [`EpipolarScatteringAir`](../../../shaders/pass/composite/EpipolarScatteringAir.comp.glsl)     | Resolves air volumetrics                                                                 |
| 2     | [`EpipolarScatteringWater`](../../../shaders/pass/composite/EpipolarScatteringWater.comp.glsl) | Evaluates underwater and through-water paths                                             |
| 3     | [`TranslucentBackComposite`](../../../shaders/pass/composite/TranslucentBackComposite.glsl)    | Captures the lit scene and volumes behind translucent surfaces                           |
| 4     | [`TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl)                        | Traces reflection and refraction in screen space                                         |
| 5     | [`TranslucentComposite`](../../../shaders/pass/composite/TranslucentComposite.glsl)            | Uses layered Z, material absorption, reflection, and refraction to update the main color |

## Settings

| Category                      | Settings                                                                                          |
|-------------------------------|---------------------------------------------------------------------------------------------------|
| Surface                       | Water roughness, wave frequency/speed, and normal scale                                           |
| Parallax                      | Enable, strength, and linear/secant steps                                                         |
| Refraction/caustics           | Refraction approximation and `SETTING_WATER_CAUSTICS`                                             |
| Medium                        | Water scattering RGB/multiplier, absorption RGB/multiplier, and refraction-approximation contrast |
| Shadow/volume                 | Light-shaft softness, water-shadow samples, and sample-pool size                                  |
| General translucent materials | Roughness reduction/min/max and absorption saturation/gamma/alpha curve/multiplier                |

These settings are declared in the Terrain and Atmospherics water screens of [
`scripts/options.main.kts`](../../../scripts/options.main.kts).

## Maintenance invariants

- Change layered-Z ordering, clears, and reads together.
- Keep reflection/refraction alpha and packing identical between geometry and composition.
- Preserve consistent path-length units for water absorption and scattering; do not hide errors with clamps.
- Keep the caustics program group enabled/disabled as one feature.
- Test above/below water, both sides of the surface, hand water, DH water, overlapping glass, and medium-boundary
  crossings.
