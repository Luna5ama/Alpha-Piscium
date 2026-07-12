# 阴影

语言：简体中文 | [English](../../en/modules/shadows.md)

Alpha Piscium 的阴影由 shadow raster、RTWSM warp、屏幕空间 shadow sample 和 direct-light consumption 四部分组成。RTWSM
在一帧末尾分析重要性，为后续 shadow raster 提供自适应映射。

Forward analysis 在 `Shadow VS` 中随 shadow rasterization 完成；backward analysis 在 [
`ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl) 中顺带完成。

## 代码位置

| 路径                                                                                                                                                                                                                                           | 职责                                 |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------|
| [`shaders/pass/geometry/ShadowPass.*.glsl`](../../../shaders/pass/geometry/)                                                                                                                                                                 | shadow vertex/geometry/fragment 入口 |
| 根部 `shadow*`, `dh_shadow*` wrappers                                                                                                                                                                                                          | world、block、cutout、entity、水与 DH 变体 |
| [`shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl`](../../../shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl)                                                                                                                  | shadow 几何后计算水面法线                   |
| [`RTWSM.glsl`](../../../shaders/techniques/rtwsm/RTWSM.glsl), [`Backward.glsl`](../../../shaders/techniques/rtwsm/Backward.glsl)                                                                                                             | warped shadow 坐标与反向映射              |
| [`IMapCollapse.comp.glsl`](../../../shaders/techniques/rtwsm/IMapCollapse.comp.glsl), [`IMapBlur.comp.glsl`](../../../shaders/techniques/rtwsm/IMapBlur.comp.glsl)                                                                           | 屏幕重要性图 collapse 与 blur             |
| [`GetWarp.comp.glsl`](../../../shaders/techniques/rtwsm/GetWarp.comp.glsl), [`Write2DWarp.comp.glsl`](../../../shaders/techniques/rtwsm/Write2DWarp.comp.glsl)                                                                               | 生成并写入下一次使用的 warp                   |
| [`ShadowSampleSetup`](../../../shaders/pass/composite/ShadowSampleSetup.comp.glsl), [`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl), [`ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl) | 屏幕空间 shadow-sample 工作队列与输出         |

## 帧内流程

1. [`ClearRTWSM`](../../../shaders/pass/begin/ClearRTWSM.comp.glsl) 清理本帧 RTWSM accumulation/queue 状态。
2. shadow wrappers 使用现有 RTWSM warp 渲染 shadow color/depth；不同几何类别复用 `ShadowPass`。
3. [`EvaluateShadowWaterNormal`](../../../shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl) 计算 shadow water
   normal。
4. [`ShadowSampleSetup`](../../../shaders/pass/composite/ShadowSampleSetup.comp.glsl) 建立采样任务。
5. [`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl) 从 SSBO 0 offset 32 indirect
   dispatch，处理需要 SSS 的样本。
6. [`ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl) 完成主 shadow 过滤；[
   `DirectLighting`](../../../shaders/pass/composite/DirectLighting.glsl) 消费结果。
7. 半透明/体积完成后，[`IMapCollapse`](../../../shaders/techniques/rtwsm/IMapCollapse.comp.glsl) collapse
   当前屏幕重要性；[`IMapBlur`](../../../shaders/techniques/rtwsm/IMapBlur.comp.glsl) blur，[
   `GetWarp`](../../../shaders/techniques/rtwsm/GetWarp.comp.glsl) 计算 warp，[
   `Write2DWarp`](../../../shaders/techniques/rtwsm/Write2DWarp.comp.glsl) 写回 2D warp。

## 资源

[`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 的固定 RTWSM tiles：

- `persistent_rtwsm_importance2D`：256×256 R32F。
- `persistent_rtwsm_importance1D`、`persistent_rtwsm_importance1DBlurred`：256×2 R32F。
- `persistent_rtwsm_warp`、`persistent_rtwsm_texelSize`：256×2 R32F。
- `persistent_rtwsm_warpTexelSize`：256×256 RGBA16。

这些固定 tile 会跨 raster/composite 边界保存 importance、warp 和 texel-size 数据。[
`scripts/shaders.properties`](../../../scripts/shaders.properties) 也管理 shadow blending；不要把 blend 行为移入 shader
heuristic。

## 设置

- 基础：`SETTING_SHADOW_MAP_RESOLUTION`、`shadowDistance`。
- RTWSM forward/backward：`SETTING_RTWSM_F*`、`SETTING_RTWSM_B*`。
- PCSS：blocker-search count、sample count、blocker/visibility penumbra factors。
- SSS：样本数和深度/扩散范围在材质 screen 中。

Profiles 会缩放 shadow resolution、distance、blocker search 和 sample count。设置声明在 [
`scripts/options.main.kts`](../../../scripts/options.main.kts) 中；项目属性会将 RTWSM 最小值派生为 custom uniforms。

## 远景（Distant Horizons）

[`scripts/shaders.properties`](../../../scripts/shaders.properties) 当前显式设置 `dhShadow.enabled=false`，注释说明
Fabric 上的 DH shadow 仍有问题。`dh_shadow.*` 入口保留用于集成代码，但不要在文档或测试中假定运行时已启用 DH shadow。

## 维护约束

- RTWSM 是跨帧反馈：同时检查当前帧 shadow 稳定性和下一帧 warp 响应。
- importance、warp 与 shadow 坐标的分辨率/方向必须一致。
- 修改采样队列时同步 setup、indirect offset、SSS 和 main sample consumer。
- 实测日月方向切换、快速转身、远近平移、水面、薄 cutout、SSS 和不同 shadow distance。
