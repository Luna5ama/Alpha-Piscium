# 全局光照

语言：简体中文 | [English](../../en/modules/global-illumination.md)

当前 GI 是屏幕空间 ReSTIR/SST 管线，配合 environment probe 和独立的时空降噪。pass 顺序以 [
`scripts/programs.main.kts`](../../../scripts/programs.main.kts) 为准；共享算法集中在 [
`shaders/techniques/gi/`](../../../shaders/techniques/gi/)，入口集中在 [
`shaders/pass/composite/`](../../../shaders/pass/composite/)。

## 代码位置

| 路径                                                                                                                                                                                                                        | 职责                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| [`shaders/techniques/gi/Common.glsl`](../../../shaders/techniques/gi/Common.glsl)                                                                                                                                         | GI 公共数据、坐标与基础访问                  |
| [`InitialSample.glsl`](../../../shaders/techniques/gi/InitialSample.glsl), [`RaySort.glsl`](../../../shaders/techniques/gi/RaySort.glsl), [`FinishTrace.comp.glsl`](../../../shaders/techniques/gi/FinishTrace.comp.glsl) | 初始样本、长 SST 路径的排序与完成              |
| [`Reservoir.glsl`](../../../shaders/techniques/gi/Reservoir.glsl), [`PairwiseMISMetadata.glsl`](../../../shaders/techniques/gi/PairwiseMISMetadata.glsl)                                                                  | reservoir 编码与 pairwise reuse 元数据 |
| [`ResampleMaterial.glsl`](../../../shaders/techniques/gi/ResampleMaterial.glsl)                                                                                                                                           | reuse 时的材质表示                     |
| [`Reproject.glsl`](../../../shaders/techniques/gi/Reproject.glsl), [`ReprojectInfo.glsl`](../../../shaders/techniques/gi/ReprojectInfo.glsl)                                                                              | history 重投影                      |
| [`Irradiance.glsl`](../../../shaders/techniques/gi/Irradiance.glsl)                                                                                                                                                       | GI irradiance/shading 共享计算       |
| [`DenoiserEdgeClassification.glsl`](../../../shaders/techniques/gi/DenoiserEdgeClassification.glsl), [`DenoiseBlur.glsl`](../../../shaders/techniques/gi/DenoiseBlur.glsl)                                                | 降噪边缘与 blur 核心                    |
| [`shaders/techniques/EnvProbe.glsl`](../../../shaders/techniques/EnvProbe.glsl)                                                                                                                                           | environment probe 映射/采样共享代码      |
| [`shaders/techniques/SST2.glsl`](../../../shaders/techniques/SST2.glsl), [`HiZ.glsl`](../../../shaders/techniques/HiZ.glsl), [`HiZCheck.glsl`](../../../shaders/techniques/HiZCheck.glsl)                                 | 屏幕空间 trace 与 Hi-Z 查询             |

## 输入准备

| 顺序 | 阶段 / Pass                                                                                                                                                                                                                                                                                                                                                                                             | 作用                                                        |
|----|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| 1  | Geometry                                                                                                                                                                                                                                                                                                                                                                                              | 写入当前帧的深度、法线、粗糙度、材质与 light-map 数据                          |
| 2  | [`HiZGen`](../../../shaders/pass/composite/HiZGen.csh)、[`GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl)、[`GIDenoiserEdgeDilation`](../../../shaders/pass/composite/GIDenoiserEdgeDilation.comp.glsl)、[`GIDenoiserReproject`](../../../shaders/pass/composite/GIDenoiserReproject.comp.glsl) | 构建 Hi-Z，执行 GI edge classification/dilation，并预先重投影 history |
| 3  | [`DirectLighting`](../../../shaders/pass/composite/DirectLighting.glsl)                                                                                                                                                                                                                                                                                                                               | 完成直接光照；GI 使用同一 G-buffer 与 shadow 结果，避免重复材质解码              |

## 环境探针

probe 为离开当前屏幕的 GI 查询保存低频/历史场景信息。它与 GI 准备交错执行：

| 顺序 | Pass                                                                                                           | 作用                            |
|----|----------------------------------------------------------------------------------------------------------------|-------------------------------|
| 1  | [`EnvProbeUpdate1ReprojectScatter`](../../../shaders/pass/composite/EnvProbeUpdate1ReprojectScatter.comp.glsl) | 重投影并 scatter 旧 probe          |
| 2  | [`EnvProbeUpdate2ReprojectDilate`](../../../shaders/pass/composite/EnvProbeUpdate2ReprojectDilate.comp.glsl)   | 以 `PASS=1`、`PASS=2` 两次填补重投影空洞 |
| 3  | [`EnvProbeUpdate3ReprojectGather`](../../../shaders/pass/composite/EnvProbeUpdate3ReprojectGather.comp.glsl)   | Gather 有效重投影数据                |
| 4  | [`EnvProbeUpdate4ProjectCurrent`](../../../shaders/pass/composite/EnvProbeUpdate4ProjectCurrent.comp.glsl)     | 把当前帧结果投影回 probe               |

运行时资源是 `uimg_envProbe`（在 [`shaders/shaders.properties`](../../../shaders/shaders.properties) 中声明为 1024×768
RGBA32UI）和 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 中的固定 `persistent_envProbeTemp`（1024×768
RGBA16F）。[`ClearEnvProbe`](../../../shaders/pass/begin/ClearEnvProbe.comp.glsl) 在需要时清 probe。

## ReSTIR/SST pass 流程

| 顺序 | Pass                                                                                                                                                                                                                         | 说明                                    |
|----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| 1  | [`GIReSTIRInitalSampleRayGenTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayGenTrace.comp.glsl)                                                                                                               | 生成初始候选并开始 SST                         |
| 2  | [`GIReSTIRInitalSampleRaySort`](../../../shaders/pass/composite/GIReSTIRInitalSampleRaySort.comp.glsl), [`GIReSTIRInitalSampleRayFinishTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayFinishTrace.comp.glsl) | 仅 initial SST steps ≥ 64；排序并完成长路径     |
| 3  | [`GIReSTIRTemporalReuse`](../../../shaders/pass/composite/GIReSTIRTemporalReuse.comp.glsl)                                                                                                                                   | 从上一帧 reservoir、样本、hit normal 与材质重投影   |
| 4  | [`GIReSTIRDuplicationMapDecorrelate`](../../../shaders/pass/composite/GIReSTIRDuplicationMapDecorrelate.comp.glsl)                                                                                                           | 可选 decorrelation                      |
| 5  | [`GIReSTIRPairedSpatialReuse`](../../../shaders/pass/composite/GIReSTIRPairedSpatialReuse.comp.glsl) × 1–4                                                                                                                   | pairwise spatial reuse；每批最多覆盖 7 个基础样本 |
| 6  | [`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl)                                                                                                                         | 对选中样本做 shading                        |
| 7  | [`GIReSTIRSpatialReuseRaySort`](../../../shaders/pass/composite/GIReSTIRSpatialReuseRaySort.comp.glsl)                                                                                                                       | 整理仍需 trace 的 spatial rays             |
| 8  | [`GIReSTIRSpatialReuseTrace`](../../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl)                                                                                                                           | 完成 spatial visibility/SST             |

四个 spatial-reuse pass 的 `PASS_INDEX` 为 0–3，`PASS_BASE_SAMPLE_INDEX` 为 0/7/14/21；它们从 SSBO 0 offset 48 indirect
dispatch。`history_restir_reservoirTemporal`、`history_restir_prevSample`、`history_restir_prevHitNormal` 保存上一帧
输入，`transient_restir_reservoirTemporal`、`transient_restir_spatialInput` 和
`transient_restir_pairwiseMISMetadata` 连接本帧阶段。[
`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl) 会在执行本帧最后一轮读取时，
将当前 temporal reservoir 复制到固定的 history tile。所有 tile 定义见 [
`shaders/shadesmith.json`](../../../shaders/shadesmith.json)。

## GI 降噪

ReSTIR shading 后依次执行：

| 顺序 | Pass                                                                                                                                                              | 作用                                                              |
|----|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| 1  | [`GIDenoiserAccum`](../../../shaders/pass/composite/GIDenoiserAccum.comp.glsl)                                                                                    | 时域 accumulation；更新 `history_gi1`…`history_gi5`                  |
| 2  | [`GIDenoiserAntiFireFly`](../../../shaders/pass/composite/GIDenoiserAntiFireFly.comp.glsl)                                                                        | 可选 anti-firefly pass                                            |
| 3  | [`GIDenoiserGIMip`](../../../shaders/pass/composite/GIDenoiserGIMip.comp.glsl)                                                                                    | 从 SSBO 0 offset 16 indirect dispatch，构建 diffuse/specular mip 输入 |
| 4  | [`GIDenoiserHistoryFix`](../../../shaders/pass/composite/GIDenoiserHistoryFix.comp.glsl)                                                                          | 修复低置信 history                                                   |
| 5  | [`GIDenoiserBlur`](../../../shaders/pass/composite/GIDenoiserBlur.comp.glsl)、[`GIDenoiserPostBlur`](../../../shaders/pass/composite/GIDenoiserPostBlur.comp.glsl) | 可选 blur 与 post-blur pass                                        |

重投影输入还包括 `history_viewZ`、历史/当前 view normal、geometry normal、edge mask、roughness 和 average view-Z。修改 tile
格式或生命周期时必须同步 [`shadesmith.json`](../../../shaders/shadesmith.json)，不能只改 sampler。

## 设置

GI 设置集中在 [`scripts/options.main.kts`](../../../scripts/options.main.kts) 的 GI 与 denoiser screens：

- 追踪：`SETTING_GI_INITIAL_SST_STEPS`、`SETTING_GI_VALIDATE_SST_STEPS`、`SETTING_GI_SST_THICKNESS`。
- probe/sky：`SETTING_GI_PROBE_FADE_START/END`、`SETTING_GI_MC_SKYLIGHT_ATTENUATION`。
- reuse：`SETTING_GI_TEMPORAL_REUSE_LIMIT`、`SETTING_GI_SPATIAL_REUSE`、`SETTING_GI_SPATIAL_REUSE_COUNT`、
  `SETTING_GI_DECORRELATE`。
- denoiser：spatial enable/sample counts、history lengths、fast-history clamping、flicker
  suppression、anti-firefly、history-fix weights。

Profiles 主要缩放 SST steps、spatial reuse count 和 denoiser sample counts。任何新的 `SETTING_*` 都必须先在 options DSL
中注册，再由 GLSL 或 program 条件使用。

## 维护约束

- reservoir 的 pack/unpack、MIS metadata 和 producer/consumer 必须同改。
- temporal tile 必须与当前/上一帧 jitter、camera transform 和 G-buffer 语义一致。
- edge classification/dilation 必须保持在 reprojection 与 accumulation 之前。
- 改 spatial 批大小时，同步 program count thresholds、base sample index 和 indirect 工作队列布局。
- 验证至少覆盖静止收敛、相机移动、disocclusion、屏幕边缘和设置切换后的 history reset。
