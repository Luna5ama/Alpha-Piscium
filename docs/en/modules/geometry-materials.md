# Geometry, G-buffer, and Materials

Language: English | [简体中文](../../sc/modules/geometry-materials.md)

The geometry stage converts Minecraft, Distant Horizons, and optional Voxy draws into the unified surface data consumed
by later compute passes. Root Iris wrappers select macros/includes; the real entry shaders live under [
`shaders/pass/geometry/`](../../../shaders/pass/geometry/).

## Entry points

| Path                                                                                                                                                                                         | Responsibility                                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [`GBufferSolid.vert.glsl`](../../../shaders/pass/geometry/GBufferSolid.vert.glsl), [`GBufferSolid.frag.glsl`](../../../shaders/pass/geometry/GBufferSolid.frag.glsl)                         | Solid and cutout terrain, entities, hand, blocks, and particles                                                                         |
| [`GBufferTranslucent.vert.glsl`](../../../shaders/pass/geometry/GBufferTranslucent.vert.glsl), [`GBufferTranslucent.frag.glsl`](../../../shaders/pass/geometry/GBufferTranslucent.frag.glsl) | Water, translucent particles, and active transparent surfaces; weather is currently NOOP                                                |
| [`ShadowPass.vert/geom/frag.glsl`](../../../shaders/pass/geometry/)                                                                                                                          | Shadow-map geometry; see [Shadows](shadows.md)                                                                                          |
| Root [`gbuffers_*.vsh/.fsh`](../../../shaders/)                                                                                                                                              | Thin wrappers from Iris program names to solid, translucent, or explicit NOOP implementations                                           |
| [`dh_terrain.*`](../../../shaders/), [`dh_water.*`](../../../shaders/), [`dh_shadow.*`](../../../shaders/)                                                                                   | Distant Horizons entry points                                                                                                           |
| [`voxy_opaque.glsl`](../../../shaders/voxy_opaque.glsl), [`voxy_translucent.glsl`](../../../shaders/voxy_translucent.glsl), [`voxy.json`](../../../shaders/voxy.json)                        | Implements `voxy_emitFragment`, writes `colortex16/17/18`, then merges in [`VoxyMerge`](../../../shaders/pass/composite/VoxyMerge.glsl) |

Do not duplicate material logic in every root wrapper. Add a draw category through the existing macro branches and
shared geometry pass.

## Material path

| File                                                                                                                                                                                               | Contents                                                         |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| [`shaders/util/Material.glsl`](../../../shaders/util/Material.glsl)                                                                                                                                | Unified material structure and accessors                         |
| [`MaterialIDConst.glsl`](../../../shaders/util/MaterialIDConst.glsl)                                                                                                                               | Material ID constants used by shaders                            |
| [`HardcodedPBR.glsl`](../../../shaders/util/HardcodedPBR.glsl)                                                                                                                                     | Decodes project material parameters from `usam_pbrLUT0`          |
| [`GBufferData.glsl`](../../../shaders/util/GBufferData.glsl)                                                                                                                                       | G-buffer packing, unpacking, and shared reads                    |
| [`shaders/util/BSDF.glsl`](../../../shaders/util/BSDF.glsl), [`Fresnel.glsl`](../../../shaders/util/Fresnel.glsl); [`shaders/techniques/Lighting.glsl`](../../../shaders/techniques/Lighting.glsl) | Shared BSDF and lighting functions for later direct/GI shading   |
| [`block.properties`](../../../shaders/block.properties)                                                                                                                                            | Minecraft block ID to project material/emissive classification   |
| [`item.properties`](../../../shaders/item.properties)                                                                                                                                              | Currently only `item.0=air` for held-item/empty-hand ID handling |

The solid fragment samples base texture and available LabPBR normal/specular data, receives light-map coordinates, and
packs raw surface data. Shading paths normally unpack [`GBufferData`](../../../shaders/util/GBufferData.glsl) and call
`material_decode`; a few edge/classification paths read packed fields directly for performance.

## Main frame resources

Raster first writes three solid G-buffer attachments:

- `colortex8` / `usam_gbufferSolidData1` (RGBA32UI): packed geometry normal/tangent, PBR specular, shading normal,
  light-map coordinates, and material ID.
- `colortex9` / `usam_gbufferSolidData2` (R32UI): packed albedo and hand/bitangent/PBR flags.
- `colortex10` / `usam_gbufferSolidViewZ` (R32F): solid view-Z.

Format-reference comments and clear flags live in [`shaders/base/Configs.glsl`](../../../shaders/base/Configs.glsl),
sampler aliases live in [`shaders/base/Textures.glsl`](../../../shaders/base/Textures.glsl), and pack/unpack logic lives
in [`shaders/util/GBufferData.glsl`](../../../shaders/util/GBufferData.glsl).

Later compute derives these screen tiles from the G-buffer:

- `transient_lmCoord`: light-map coordinates.
- `transient_geomViewNormal`, `transient_viewNormal`, and their histories: geometry and shading normals.
- `transient_solidAlbedo`: working-space albedo and emissive; exposure currently reads its A-channel emissive value.
- `transient_specularPBRData`: R is `sqrt(roughness)` and G is the dielectric flag; `history_roughness` stores temporal
  roughness.
- `history_viewZ`: depth history used by reprojection. `history_avgViewZ` is currently retained only in tile
  configuration; no active pass reads or writes it.

These are compute-derived tiles, not the raster attachments themselves. Their formats and lifetimes are owned by [
`shaders/shadesmith.json`](../../../shaders/shadesmith.json); producers and consumers must change with the tile
declaration.

## Pipeline integration

1. `begin` updates current/previous camera, jitter, and global state.
2. Shadow geometry prepares direct-light visibility.
3. Solid geometry writes surface data; optional Voxy data merges in [
   `VoxyMerge`](../../../shaders/pass/composite/VoxyMerge.glsl).
4. Translucent geometry writes its G-buffer, transmittance, and near/far depth; [
   `TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl) produces reflection/refraction later.
5. [`HiZGen`](../../../shaders/pass/composite/HiZGen.csh) builds Hi-Z; [
   `GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl)
   builds edge/depth layers; [`ShadowSampleSetup`](../../../shaders/pass/composite/ShadowSampleSetup.comp.glsl) and [
   `GIDenoiserEdgeDilation`](../../../shaders/pass/composite/GIDenoiserEdgeDilation.comp.glsl) perform shadow setup/edge
   dilation; [`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl) and [
   `ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl) complete shadow sampling; [
   `DirectLighting.glsl`](../../../shaders/pass/composite/DirectLighting.glsl) runs direct lighting; and [
   `GIReSTIRInitalSampleRayGenTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayGenTrace.comp.glsl) starts
   GI.

## Settings

Material settings live in the Terrain screens of [`scripts/options.main.kts`](../../../scripts/options.main.kts):

- PBR mode: `SETTING_PBR_MATERIAL`.
- Normal mapping: `SETTING_NORMAL_MAPPING`, `SETTING_NORMAL_MAPPING_STRENGTH`, `SETTING_TBN_PACKING`.
- BRDF limits: minimum F0, solid/water/translucent roughness, and maximum specular luminance.
- Emissive: global/particle/entity strength and the emissive curve. Fire/lava temperature controls are currently
  declared only in options/UI; no current shader reads those two macros.
- SSS: sample count, diffuse range, and depth range.

## Maintenance invariants

- Change G-buffer bit layout, writers, and every reader together.
- Keep geometry normals distinct from shading normals; denoiser edge/history explicitly depend on both.
- Extend existing material IDs/rules instead of creating a one-block shading path.
- Test block, entity, held-item, and distant paths after [`block.properties`](../../../shaders/block.properties)/[
  `item.properties`](../../../shaders/item.properties) mapping changes.
- Validate vanilla terrain, modded smooth normals, and LabPBR normal maps after TBN packing changes.
