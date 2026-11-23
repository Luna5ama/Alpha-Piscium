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

## 树叶与草的覆盖率

2.1 原生 TAA 和超分默认保住叶簇、草簇透出背景的比例及主要孔洞的形状。移动时最细的叶缘可以略软。高采样参考沿用正常游玩的材质过滤、alpha 裁切和纹理采样足迹，在同一时刻、同一视角取多个亚像素样本。截图模式会把纹理导数置零，并跳过随 mip 变化的阈值抖动。[地形 cutout 入口](../../shaders/gbuffers_terrain_cutout.fsh) · [材质采样与裁切](../../shaders/pass/geometry/GBufferSolid.frag.glsl#L51-L103)

**现有做法。** 2026 年 1 月 25 日的版本开始按屏幕像素取蓝噪声来扰动裁切阈值。次日短暂试过随帧轮换，最后在 `Leaves tweak` 里固定取第 0 层，并让阈值分布随 mip 层级变化，用来调远处树叶的疏密。2.1 的普通渲染路径把阈值限制在 `alphaTestRef` 与 1 之间，完全不透明的样本能通过。过滤后 alpha 已低于参考阈值的样本会在所有噪声取值下被裁掉。`2.0/fsr3@b72a8b5fd` 仍用旧的加法阈值，在部分 `alphaTestRef` 下会超过 1。对照原型中的亮点时，单独检查这种裁切漏孔和 GI 残噪。[引入抖动](https://github.com/Luna5ama/Alpha-Piscium/commit/0b8b65ba5f543a4f633de1837c443fbca40ac478) · [固定空间图案](https://github.com/Luna5ama/Alpha-Piscium/commit/e800d8751b47e4d7619bea41bdb968403100f4cb) · [修复不透明处的亮点](https://github.com/Luna5ama/Alpha-Piscium/commit/2d76ac8b5cc70a9a64f7064004a1aeb3f8558308) · [FSR 原型的阈值](https://github.com/Luna5ama/Alpha-Piscium/blob/b72a8b5fddeb2e52ea20a2c36eb16e0c30b0661b/shaders/pass/geometry/GBufferSolid.frag.glsl#L92-L101)

**同一屏幕像素上的两层叶片。** 在 `d2c4aed21` 所用的 128×128 R8 蓝噪声第 0 层上，取 `LOD=1`、`alphaTestRef=0.5`、过滤后 `alpha=0.5625`。按 shader 的阈值计算，一层通过 8194/16384 个位置。叠上参数相同的第二层，两层查询相同的屏幕坐标，合起来仍覆盖这 8194 个位置。若两层孔洞相互独立，同样单层覆盖率的期望联合覆盖约为 75%。这个算例检验阈值相关性，真实树冠的疏密以参考画面为准。[噪声读取](../../shaders/util/Rand.glsl#L21-L23) · [纹理与格式](../../scripts/shaders.properties#L38)

**改进候选。** 首先保留单层阈值曲线，只比较随机阈值挂在哪个坐标上。候选使用叶片的局部坐标或贴图坐标，给不同方块、叶面各自一个稳定的种子。噪声网格随像素覆盖的纹理范围选尺度，跨尺度过渡时保持阈值分布。用一层叶面、多层错开的叶面和能看见天空的树冠核对单层疏密、层间孔洞及慢移时图案是否贴着叶面。若单层覆盖率本身已随 mip 偏离参考，再比较保覆盖率的 alpha mip 或阈值校准。[Wyman 与 McGuire 的 Hashed Alpha Testing](https://research.nvidia.com/labs/rtr/publication/wyman2017hashed/) · [Castaño 的 alpha mip 覆盖率校准](https://www.the-witness.net/news/2010/09/computing-alpha-mipmaps/)

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
