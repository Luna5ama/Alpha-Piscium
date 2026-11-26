# Geometry, G-buffer, and Materials

The geometry stage converts Minecraft, Distant Horizons, and optional Voxy draws into the unified surface data consumed
by later compute passes. Root Iris wrappers select macros/includes; the real entry shaders live under [
`shaders/pass/geometry/`](../../shaders/pass/geometry/).

## Entry points

| Path | Responsibility |
| --- | --- |
| [`GBufferSolid.vert.glsl`](../../shaders/pass/geometry/GBufferSolid.vert.glsl), [`GBufferSolid.frag.glsl`](../../shaders/pass/geometry/GBufferSolid.frag.glsl) | Solid and cutout terrain, entities, hand, blocks, and particles |
| [`GBufferTranslucent.vert.glsl`](../../shaders/pass/geometry/GBufferTranslucent.vert.glsl), [`GBufferTranslucent.frag.glsl`](../../shaders/pass/geometry/GBufferTranslucent.frag.glsl) | Water, translucent particles, and active transparent surfaces; weather is currently NOOP |
| [`ShadowPass.vert/geom/frag.glsl`](../../shaders/pass/geometry/) | Shadow-map geometry; see [Shadows](shadows.md) |
| Root [`gbuffers_*.vsh/.fsh`](../../shaders/) | Thin wrappers from Iris program names to solid, translucent, or explicit NOOP implementations |
| [`dh_terrain.*`](../../shaders/), [`dh_water.*`](../../shaders/), [`dh_shadow.*`](../../shaders/) | Distant Horizons entry points |
| [`voxy_opaque.glsl`](../../shaders/voxy_opaque.glsl), [`voxy_translucent.glsl`](../../shaders/voxy_translucent.glsl), [`voxy.json`](../../shaders/voxy.json) | Implements `voxy_emitFragment`, writes `colortex16/17/18`, then merges in [`VoxyMerge`](../../shaders/pass/composite/VoxyMerge.glsl) |

Do not duplicate material logic in every root wrapper. Add a draw category through the existing macro branches and
shared geometry pass.

## Material path

| File | Contents |
| --- | --- |
| [`shaders/util/Material.glsl`](../../shaders/util/Material.glsl) | Unified material structure and accessors |
| [`MaterialIDConst.glsl`](../../shaders/util/MaterialIDConst.glsl) | Material ID constants used by shaders |
| [`HardcodedPBR.glsl`](../../shaders/util/HardcodedPBR.glsl) | Decodes project material parameters from `usam_pbrLUT0` |
| [`GBufferData.glsl`](../../shaders/util/GBufferData.glsl) | G-buffer packing, unpacking, and shared reads |
| [`shaders/util/BSDF.glsl`](../../shaders/util/BSDF.glsl), [`Fresnel.glsl`](../../shaders/util/Fresnel.glsl); [`shaders/techniques/Lighting.glsl`](../../shaders/techniques/Lighting.glsl) | Shared BSDF and lighting functions for later direct/GI shading |
| [`block.properties`](../../shaders/block.properties) | Minecraft block ID to project material/emissive classification |
| [`item.properties`](../../shaders/item.properties) | Currently only `item.0=air` for held-item/empty-hand ID handling |

The solid fragment samples base texture and available LabPBR normal/specular data, receives light-map coordinates, and
packs raw surface data. Shading paths normally unpack [`GBufferData`](../../shaders/util/GBufferData.glsl) and use
`material_decode` to select the built-in or LabPBR path. Some edge/classification paths read packed fields directly
for performance.

## Main frame resources

Raster first writes three solid G-buffer attachments:

- `colortex8` / `usam_gbufferSolidData1` (RGBA32UI): packed geometry normal/tangent, PBR specular, shading normal,
  light-map coordinates, and material ID.
- `colortex9` / `usam_gbufferSolidData2` (R32UI): packed albedo and hand/bitangent/PBR flags.
- `colortex10` / `usam_gbufferSolidViewZ` (R32F): solid view-Z.

Format-reference comments and clear flags live in [`shaders/base/Configs.glsl`](../../shaders/base/Configs.glsl),
sampler aliases live in [`shaders/base/Textures.glsl`](../../shaders/base/Textures.glsl), and pack/unpack logic lives
in [`shaders/util/GBufferData.glsl`](../../shaders/util/GBufferData.glsl).

Later compute derives these screen tiles from the G-buffer:

- `transient_lmCoord`: light-map coordinates.
- `transient_geomViewNormal`, `transient_viewNormal`, and their histories: geometry and shading normals.
- `transient_solidAlbedo`: working-space albedo and emissive; exposure currently reads its A-channel emissive value.
- `transient_specularPBRData`: R is `sqrt(roughness)` and G is the dielectric flag; `history_roughness` stores temporal
  roughness.
- `history_viewZ`: depth history used by reprojection. `history_avgViewZ` is currently retained only in tile
  configuration; no active pass reads or writes it.

These are compute-derived tiles, not the raster attachments themselves. Their formats and lifetimes are owned by [
`shaders/shadesmith.json`](../../shaders/shadesmith.json); producers and consumers must change with the tile
declaration.

## Pipeline integration

| Order | Stage / pass | Purpose |
| --- | --- | --- |
| 1 | `begin` | Updates current/previous camera, jitter, and global state |
| 2 | Shadow geometry | Prepares direct-light visibility |
| 3 | Solid geometry; [`VoxyMerge`](../../shaders/pass/composite/VoxyMerge.glsl) | Writes surface data and optionally merges Voxy data |
| 4 | Translucent geometry; [`TranslucentSST`](../../shaders/pass/composite/TranslucentSST.glsl) | Writes its G-buffer, transmittance, and near/far depth; later produces reflection/refraction |
| 5 | [`HiZGen`](../../shaders/pass/composite/HiZGen.csh), [`GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl), [`ShadowSampleSetup`](../../shaders/pass/composite/ShadowSampleSetup.comp.glsl), [`GIDenoiserEdgeDilation`](../../shaders/pass/composite/GIDenoiserEdgeDilation.comp.glsl), [`ShadowSampleSSS`](../../shaders/pass/composite/ShadowSampleSSS.comp.glsl), [`ShadowSample`](../../shaders/pass/composite/ShadowSample.comp.glsl), [`DirectLighting`](../../shaders/pass/composite/DirectLighting.glsl), [`GIReSTIRInitalSampleHiZ`](../../shaders/pass/composite/GIReSTIRInitalSampleHiZ.comp.glsl) | Builds Hi-Z and edge/depth layers, sets up/dilates shadow data, filters shadows, evaluates direct lighting, and starts GI |

## Settings

Material settings live in the Terrain screens of [`scripts/options.main.kts`](../../scripts/options.main.kts):

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
- Test block, entity, held-item, and distant paths after [`block.properties`](../../shaders/block.properties)/[
  `item.properties`](../../shaders/item.properties) mapping changes.
- Validate vanilla terrain, modded smooth normals, and LabPBR normal maps after TBN packing changes.

## Leaf highlight turn

Rerun this scene when changing cutout foliage, its direct lighting, material normals or the temporal processing of highlights. The user saw the same leaf faces flash while turning in place. Compare the right canopy through the turn, hold, reversal and final stop. The original failure switches bright interior surfaces off and back on within a few frames. Watch the same surfaces through motion and look for a loss of normal highlight detail. The later fix is a comparison candidate until its appearance is accepted.

From the version checkout run `python scripts/test/leaf_highlight_turn.py <candidate-commit>` and pass the emitted JSON to `vibris_run_actions`. Wait for the job, check its loaded source and 60 actual poses, then compare the sequence with the reference or replay a comparison revision the same way. The [request generator](../../scripts/test/leaf_highlight_turn.py) runs one source per job with 144 warmup frames. The preset `ap-leaf-turn-4605290f` uses the isolated QA copy of Dump Forest, player position (−88.534731, 64, 19.515801), yaw/pitch (2.3°, −67.4°) through (7.5°, −75°), tick 6000, clear weather and the 1920×1080 preset at FOV 45. Match Ultra, SPBR-21, built-in PBR, sRGB and EV −1.06. If rebuilding the fixture, retain the reported canopy geometry and keep the replay save separate from the player's world.

The shared 1.21.11 Vibris instance ran this trajectory on `dfc4bb3d` and `1c485d48` under Iris 1.10.6-vibris.3. A longer pair on the reported Iris build also covers the slow traversal and auto exposure. The original photos, motion, fixed-leaf measurements and reference video are in the project evidence store. An unavailable preset or incomplete sequence leaves the visual check pending.

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"repro.json"} -->
<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"before-after.mp4"} -->
<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"live-audit.json"} -->

## Whole-block biome tint for 2.0 voxel GI

At pack revision `140ddfa8`, generated with Shadesmith `2fbd19f`, initial GI and the radiance cache multiply the voxel atlas RGB by the same terrain-vertex tint before material decoding. This also colours a grass block's dirt sides. The mesh already supplies this colour to visible terrain. Group 0 is neutral. Grass, biome foliage, spruce, birch and dry foliage have separate histories. Palette estimates are local to the observed geometry, so biome boundaries can produce a mixture.

The 1.10.4 Iris diagnostic at 1280 × 720 followed a grass dirt-side texel through both biome settings. The source RGB was (0.7255, 0.5216, 0.3608). Initial GI received (0.4125, 0.3866, 0.1259) on plains and (0.4097, 0.2639, 0.1089) on badlands. The RC and initial-sample readbacks also confirmed foliage multiplication and neutral stone. The sampled visible G-buffer colours matched the baseline byte for byte in both frame phases. Across 748 reduction dispatches the largest colour error against the direct-seed and 0.9/0.1 update was below 9e-8. An empty Nether palette returned white and reseeded after returning to the Overworld. In a second run that stayed in the Overworld, both valid tints held every RGB component and the validity bit across 108 consecutive frames without samples, then resumed updates on returning to the fixture.

Regenerate the 2.0 material inputs with Minecraft 26.2 in a staging directory. Check block IDs, quad data and the two unchanged LUTs before taking the material flags. The branch owns additional optimizations in its production model tracer. In the 720p matched product test, the largest paired delta of the shadow group or palette reduction was +0.0041 ms. Initial GI varied by −0.0015/+0.0404 ms on plains and +0.0102/+0.0082 ms on badlands. The separate old-Iris game actually used WDDM Normal. The downstream timing sentinel differed by −0.0010 ms in both orders. The report preserves the frame timings and background samples.

<!-- vela:evidence {"run":"12381471-b0c9-4a36-9a0b-8a8e945a6899","path":"verified.md"} -->

<!-- vela:evidence {"run":"12381471-b0c9-4a36-9a0b-8a8e945a6899","path":"voxel-biome-tint-evidence.zip"} -->

## Leaf highlights during a camera turn in 2.0

In the Dump Forest report the sun lay on the tangent plane of the affected leaf face. Rotating the camera changed the sign of the rounded view-space geometric cosine from +1.42e-9 to -1.04e-9 and back. The binary SSS-material reflection gate switched the full direct specular and diffuse terms. Across 2,726 shared world-space samples, the specular luminance at the three witness frames was 7.647, 0 and 7.597. The independently computed transmission stayed near 2.26–2.29 and the shadow input stayed near 0.49. The jump was already present in scene-linear direct lighting.

`1c485d48` replaces the binary test with a one-sided smoothstep in the geometric cosine. It suppresses roundoff within 1e-6 and reaches the original reflection weight at 1e-4. In the same witness frames the tangent specular term stayed at zero and the transmitted light tracked the original. The focused readback used Iris 1.10.6-vibris.2, Ultra, built-in PBR and SPBR-21. The running default Vibris instance replayed the fixed-position turn in a separate QA world with 60 identical poses per revision and -1.06 EV, then tested the same surface with the sun away from its tangent plane. The existing game restored the prior scene, settings and source after each job.

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"report.md"} -->

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"fixed-leaves.json"} -->

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"lighting-terms.png"} -->

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"before-after-vibris.mp4"} -->

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"front-vibris.png"} -->

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"live-audit.json"} -->

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"leaf-runtime-evidence.zip"} -->

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"AP20-leaf-grazing-fix.zip"} -->

<!-- vela:evidence {"run":"4605290f-70c4-4d6c-82fb-8c3bd133cfc8","path":"storage.json"} -->
