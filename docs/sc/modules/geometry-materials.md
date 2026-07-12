# 几何、G-buffer 与材质

语言：简体中文 | [English](../../en/modules/geometry-materials.md)

几何阶段负责把 Minecraft、Distant Horizons 和可选 Voxy 的绘制转成 Alpha Piscium 后续 compute pass 使用的统一表面数据。根部
Iris wrapper 只负责选择宏和 include；真实入口在 [`shaders/pass/geometry/`](../../../shaders/pass/geometry/)。

## 入口

| 路径                                                                                                                                                                                           | 职责                                                                                                                   |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| [`GBufferSolid.vert.glsl`](../../../shaders/pass/geometry/GBufferSolid.vert.glsl), [`GBufferSolid.frag.glsl`](../../../shaders/pass/geometry/GBufferSolid.frag.glsl)                         | terrain、cutout、entity、hand、block、particle 等不透明/遮罩表面                                                                  |
| [`GBufferTranslucent.vert.glsl`](../../../shaders/pass/geometry/GBufferTranslucent.vert.glsl), [`GBufferTranslucent.frag.glsl`](../../../shaders/pass/geometry/GBufferTranslucent.frag.glsl) | water、translucent particles 和活跃透明表面；当前 weather wrapper 是 NOOP                                                        |
| `ShadowPass.vert/geom/frag.glsl`                                                                                                                                                             | shadow-map 几何；见[阴影](shadows.md)                                                                                      |
| 根部 `gbuffers_*.vsh/.fsh`                                                                                                                                                                     | Iris program 名到 solid/translucent 或显式 NOOP 的薄 wrapper                                                                |
| `dh_terrain.*`, `dh_water.*`, `dh_shadow.*`                                                                                                                                                  | Distant Horizons 入口                                                                                                  |
| [`voxy_opaque.glsl`](../../../shaders/voxy_opaque.glsl), [`voxy_translucent.glsl`](../../../shaders/voxy_translucent.glsl), [`voxy.json`](../../../shaders/voxy.json)                        | 直接实现 `voxy_emitFragment`，写 `colortex16/17/18`，再由 [`VoxyMerge`](../../../shaders/pass/composite/VoxyMerge.glsl) merge |

不要在每个根部 wrapper 复制材质逻辑；新增绘制类别时沿用现有宏分支并接入共享 geometry pass。

## 材质路径

| 文件                                                                                                                                                                                                | 内容                                       |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|
| [`shaders/util/Material.glsl`](../../../shaders/util/Material.glsl)                                                                                                                               | 项目统一材质结构和读写                              |
| [`MaterialIDConst.glsl`](../../../shaders/util/MaterialIDConst.glsl)                                                                                                                              | shader 内使用的材质 ID 常量                      |
| [`HardcodedPBR.glsl`](../../../shaders/util/HardcodedPBR.glsl)                                                                                                                                    | 解码 `usam_pbrLUT0` 中的项目材质参数               |
| [`GBufferData.glsl`](../../../shaders/util/GBufferData.glsl)                                                                                                                                      | G-buffer pack/unpack 与公共读取               |
| [`shaders/util/BSDF.glsl`](../../../shaders/util/BSDF.glsl), [`Fresnel.glsl`](../../../shaders/util/Fresnel.glsl)；[`shaders/techniques/Lighting.glsl`](../../../shaders/techniques/Lighting.glsl) | 后续 direct/GI shading 共享的 BSDF 与光照函数      |
| [`block.properties`](../../../shaders/block.properties)                                                                                                                                           | Minecraft block ID 到项目材质/发光分类的映射         |
| [`item.properties`](../../../shaders/item.properties)                                                                                                                                             | 当前仅定义 `item.0=air`，服务 held-item/空手 ID 路径 |

solid fragment 入口采样基础纹理和可用的 LabPBR normal/specular，接收 light-map 坐标，并把原始表面数据写入紧凑 G-buffer。后续
shading 通常读取 [`GBufferData`](../../../shaders/util/GBufferData.glsl) 并由 `material_decode` 选择 built-in/LabPBR
路径；少量 edge/classification 代码为性能直接读取 packed 字段。

## 主要帧资源

Raster 首先写三个真正的 solid G-buffer attachments：

- `colortex8` / `usam_gbufferSolidData1`（RGBA32UI）：geometry normal/tangent、PBR specular、shading normal、light-map 坐标和
  material ID 的 packed 数据。
- `colortex9` / `usam_gbufferSolidData2`（R32UI）：albedo 与 hand/bitangent/PBR flags 的 packed 数据。
- `colortex10` / `usam_gbufferSolidViewZ`（R32F）：solid view-Z。

format 参考注释与 clear flags 位于 [`shaders/base/Configs.glsl`](../../../shaders/base/Configs.glsl)，sampler alias
位于 [`shaders/base/Textures.glsl`](../../../shaders/base/Textures.glsl)，pack/unpack 位于 [
`shaders/util/GBufferData.glsl`](../../../shaders/util/GBufferData.glsl)。

后续 compute 从 G-buffer 派生的 screen tiles 包括：

- `transient_lmCoord`：light-map 坐标。
- `transient_geomViewNormal`、`transient_viewNormal` 及对应 history：几何法线与着色法线。
- `transient_solidAlbedo`：保存工作空间 albedo 与 emissive；当前实际读取点是 exposure 使用 A 通道 emissive。
- `transient_specularPBRData`：R 为 `sqrt(roughness)`，G 为 dielectric 标志；`history_roughness` 保存 temporal roughness。
- `history_viewZ`：重投影使用的深度历史。`history_avgViewZ` 当前只在 tile 配置中保留，现有 pass 没有读写它。

这些是 compute 派生的 tile，而不是 raster attachments 本身。它们的格式和生命周期由 [
`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 管理；必须与 tile 声明一起修改 producer 和 consumer。

## 管线接入

1. `begin` 更新当前/上一帧 camera、jitter 与 global data。
2. shadow geometry 建立直接光照所需的 shadow 数据。
3. solid G-buffer 写入表面数据；可选 Voxy 结果在 [`VoxyMerge`](../../../shaders/pass/composite/VoxyMerge.glsl) merge。
4. translucent geometry 写入 translucent G-buffer、transmittance 和 near/far depth；reflection/refraction 由后面的 [
   `TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl) 生成。
5. [`HiZGen`](../../../shaders/pass/composite/HiZGen.csh) 构建 Hi-Z；[
   `GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl)
   建 edge/depth layers；[`GIDenoiserEdgeDilation`](../../../shaders/pass/composite/GIDenoiserEdgeDilation.comp.glsl) 做
   edge dilation；[`ShadowSampleSetup`](../../../shaders/pass/composite/ShadowSampleSetup.comp.glsl) 做 shadow setup；[
   `ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl)/[
   `ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl) 完成 shadow sampling；[
   `DirectLighting`](../../../shaders/pass/composite/DirectLighting.glsl) 执行 direct lighting；[
   `GIReSTIRInitalSampleRayGenTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayGenTrace.comp.glsl) 开始
   GI。

## 设置

材质相关设置在 [`scripts/options.main.kts`](../../../scripts/options.main.kts) 的 Terrain screens：

- PBR 模式：`SETTING_PBR_MATERIAL`。
- normal mapping：`SETTING_NORMAL_MAPPING`、`SETTING_NORMAL_MAPPING_STRENGTH`、`SETTING_TBN_PACKING`。
- BRDF 限制：minimum F0、solid/water/translucent roughness 和 maximum specular luminance。
- emissive：全局/particle/entity 强度与 emissive 曲线。fire/lava temperature 目前只在 options/UI 中声明，当前 shader
  没有读取这两个宏。
- SSS：sample count、diffuse range、depth range。

这些 Terrain screen 设置涵盖 PBR 模式、normal mapping/TBN packing、BRDF 限制、emissive 响应和 SSS。fire/lava temperature
控件目前仅显示在 UI 中，当前 shader 不会使用这两个宏。

## 维护约束

- G-buffer bit layout、写入端和所有读取端必须同一提交修改。
- geometry normal 与 shading normal 不可混用；denoiser edge/history 明确依赖两者。
- 新材质优先扩展现有 ID/规则，不为单个方块新增一条平行 shading path。
- 修改 [`block.properties`](../../../shaders/block.properties)/[`item.properties`](../../../shaders/item.properties)
  后用对应方块、实体、手持物和远景路径实测。
- 修改 TBN packing 时同时验证 vanilla terrain、modded smooth normals 和 LabPBR normal map。
