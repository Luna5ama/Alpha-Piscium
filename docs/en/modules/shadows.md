# Shadows

Language: English | [简体中文](../../sc/modules/shadows.md)

Alpha Piscium's shadow path consists of shadow rasterization, RTWSM warping, screen-space shadow sampling, and
direct-light consumption. RTWSM analyzes importance near the end of a frame to prepare an adaptive mapping for later
shadow rasterization.

Forward analysis is performed in `Shadow VS` during shadow rasterization. Backward analysis is performed alongside [
`ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl) during screen-space filtering.

## Code map

| Path                                                                                                                                                                                                                                         | Responsibility                                       |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| [`shaders/pass/geometry/ShadowPass.*.glsl`](../../../shaders/pass/geometry/)                                                                                                                                                                 | Shadow vertex, geometry, and fragment entry points   |
| Root `shadow*`, `dh_shadow*` wrappers                                                                                                                                                                                                        | World, block, cutout, entity, water, and DH variants |
| [`shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl`](../../../shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl)                                                                                                                  | Evaluates water normals after shadow geometry        |
| [`RTWSM.glsl`](../../../shaders/techniques/rtwsm/RTWSM.glsl), [`Backward.glsl`](../../../shaders/techniques/rtwsm/Backward.glsl)                                                                                                             | Warped shadow coordinates and backward mapping       |
| [`IMapCollapse.comp.glsl`](../../../shaders/techniques/rtwsm/IMapCollapse.comp.glsl), [`IMapBlur.comp.glsl`](../../../shaders/techniques/rtwsm/IMapBlur.comp.glsl)                                                                           | Screen-importance collapse and blur                  |
| [`GetWarp.comp.glsl`](../../../shaders/techniques/rtwsm/GetWarp.comp.glsl), [`Write2DWarp.comp.glsl`](../../../shaders/techniques/rtwsm/Write2DWarp.comp.glsl)                                                                               | Builds and writes the next warp                      |
| [`ShadowSampleSetup`](../../../shaders/pass/composite/ShadowSampleSetup.comp.glsl), [`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl), [`ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl) | Screen-space shadow-sample queue and output          |

## Per-frame flow

1. [`ClearRTWSM`](../../../shaders/pass/begin/ClearRTWSM.comp.glsl) clears per-frame RTWSM accumulation/queue state.
2. Shadow wrappers rasterize through the current RTWSM warp using the shared `ShadowPass` for each geometry category.
3. [`EvaluateShadowWaterNormal`](../../../shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl) evaluates shadow-water
   normals.
4. [`ShadowSampleSetup`](../../../shaders/pass/composite/ShadowSampleSetup.comp.glsl) creates the sampling work.
5. [`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl) runs through indirect dispatch from
   SSBO 0 offset 32.
6. [`ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl) performs main shadow filtering; [
   `DirectLighting`](../../../shaders/pass/composite/DirectLighting.glsl) consumes its result.
7. After translucency/volumetrics, [`IMapCollapse`](../../../shaders/techniques/rtwsm/IMapCollapse.comp.glsl) collapses
   current screen importance; [`IMapBlur`](../../../shaders/techniques/rtwsm/IMapBlur.comp.glsl) blurs it, [
   `GetWarp`](../../../shaders/techniques/rtwsm/GetWarp.comp.glsl) computes a warp, and [
   `Write2DWarp`](../../../shaders/techniques/rtwsm/Write2DWarp.comp.glsl) writes the 2D warp back.

## Resources

Fixed RTWSM tiles in [`shaders/shadesmith.json`](../../../shaders/shadesmith.json):

- `persistent_rtwsm_importance2D`: 256×256 R32F.
- `persistent_rtwsm_importance1D`, `persistent_rtwsm_importance1DBlurred`: 256×2 R32F.
- `persistent_rtwsm_warp`, `persistent_rtwsm_texelSize`: 256×2 R32F.
- `persistent_rtwsm_warpTexelSize`: 256×256 RGBA16.

The fixed tiles persist importance, warp, and texel-size data across the raster/composite boundary. [
`scripts/shaders.properties`](../../../scripts/shaders.properties) also owns shadow blending; do not move blend behavior
into shader heuristics.

## Settings

- Base: `SETTING_SHADOW_MAP_RESOLUTION`, `shadowDistance`.
- RTWSM forward/backward: `SETTING_RTWSM_F*`, `SETTING_RTWSM_B*`.
- PCSS: blocker-search count, sample count, blocker/visibility penumbra factors.
- SSS: sample count and depth/diffuse ranges in the material screen.

Profiles scale shadow resolution, distance, blocker search, and sample count. Settings are declared in [
`scripts/options.main.kts`](../../../scripts/options.main.kts); project properties derive RTWSM minimum values as custom
uniforms.

## Distant Horizons

[`scripts/shaders.properties`](../../../scripts/shaders.properties) currently sets `dhShadow.enabled=false` because the
Fabric DH shadow path is still marked broken. `dh_shadow.*` entry wrappers exist for integration code, but runtime DH
shadows must not be assumed enabled in documentation or tests.

## Maintenance invariants

- RTWSM is cross-frame feedback; validate current shadow stability and next-frame warp response.
- Keep importance, warp, and shadow-coordinate resolution/orientation consistent.
- Change setup, indirect layout, SSS, and main-sample consumers together.
- Test sun/moon transitions, fast turns, near/far motion, water, thin cutouts, SSS, and shadow distances.
