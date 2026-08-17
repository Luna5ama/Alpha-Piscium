# 渲染管线总览

语言：简体中文 | [English](../../en/modules/pipeline.md)

Alpha Piscium 的 compute 顺序以 [`scripts/programs.main.kts`](../../../scripts/programs.main.kts) 为唯一源。脚本按 DSL
顺序生成根部 `.csh` 和 [`scripts/programs.shaders.properties`](../../../scripts/programs.shaders.properties)。当前
`PREPARE`、`DEFERRED` 没有条目。

## 一帧的高层顺序

```text
setup（初始化 / 尺寸变化）
  ↓
begin（帧数据、清理、LUT 准备）
  ↓
shadow 几何 → shadowcomp
  ↓
prepare（当前为空）
  ↓
不透明几何 / G-buffer
  ↓
deferred（当前为空）
  ↓
水面与半透明几何
  ↓
场景准备、GI、焦散、云、阴影、光照与体积 pass
  ↓
DOFPrepare、TAAPrepare，随后选择内部关闭/TAA/FSR 3 或外部 SR 准备路径
  ↓
Bloom downsample/upsample
  ↓
PostComposite 显示变换、曝光、下一帧状态与 OverlayComposite
  ↓
可选的外部 Super Resolution，在 OverlayComposite 后执行
  ↓
final（dither 和屏幕输出）
```

这是项目内的数据依赖图，不替代 Iris 对各 program stage 的定义。

## Setup

[`InitGlobalData`](../../../shaders/pass/setup/InitGlobalData.comp.glsl) 初始化 global data；[
`ClearRGBA32UI`](../../../shaders/pass/setup/ClearRGBA32UI.glsl)、[
`ClearRGBA16F`](../../../shaders/pass/setup/ClearRGBA16F.glsl)、[
`ClearRGB10A2`](../../../shaders/pass/setup/ClearRGB10A2.glsl)、[
`ClearRGBA8`](../../../shaders/pass/setup/ClearRGBA8.glsl) 和 [`ClearR32F`](../../../shaders/pass/setup/ClearR32F.glsl)
执行按格式组织的清理。格式级清理避免为每个 tile 建立单独 shader。

## Begin

同一编号的 `_a`、`_b`、`_c` 是 program DSL 中显式分组的并列入口。[
`scripts/shaders.properties`](../../../scripts/shaders.properties) 启用 `allowConcurrentCompute=true`
，因此同组入口必须相互独立，不能依赖后缀字母形成的执行顺序；数字组之间才表达这里记录的管线先后。当前分组：

| 组                                                                                                                                                                                                                                                                                                                                       | 工作                                                                    |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| [`UpdateGlobalData`](../../../shaders/pass/begin/UpdateGlobalData.comp.glsl)、[`ClearRTWSM`](../../../shaders/pass/begin/ClearRTWSM.comp.glsl)、[`GenerateTransmittance`](../../../shaders/techniques/atmospherics/air/lut/GenerateTransmittance.comp.glsl)                                                                               | 更新 global data；清 `persistent_rtwsm_importance2D`；生成 transmittance LUT |
| [`SliceEndPoints`](../../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl)、[`GenerateMultiSctr`](../../../shaders/techniques/atmospherics/air/lut/GenerateMultiSctr.comp.glsl)、[`ClearScreen1`](../../../shaders/pass/begin/ClearScreen1.comp.glsl)、[`ClearScreen2`](../../../shaders/pass/begin/ClearScreen2.comp.glsl) | epipolar slice endpoints；multi-scattering LUT；清屏幕临时资源 1/2             |
| [`Sample`](../../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl)、[`GenerateSkyViewLUT`](../../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl)                                                                                                                                               | cloud ambient LUT sample；sky-view LUT                                 |
| [`Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl)、[`ClearEnvProbe`](../../../shaders/pass/begin/ClearEnvProbe.comp.glsl)、[`InitThreadGroupTilling`](../../../shaders/pass/begin/InitThreadGroupTilling.glsl)                                                                                          | cloud ambient LUT gather；清 environment probe；初始化 thread-group tiling  |
| [`ClearScreen3`](../../../shaders/pass/begin/ClearScreen3.comp.glsl)                                                                                                                                                                                                                                                                    | 仅 `VOXY`：清第三组屏幕资源                                                     |

## Shadow 与 Shadowcomp

帧准备完成后，根部 `shadow*.vsh/.gsh/.fsh` wrapper 将世界、实体、cutout、水和 block shadow 几何接到 [
`shaders/pass/geometry/ShadowPass.*.glsl`](../../../shaders/pass/geometry/)。几何结束后，[
`EvaluateShadowWaterNormal`](../../../shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl)
为后续阴影采样准备水面法线。完整流程见[阴影模块](shadows.md)。

## Geometry 与 G-buffer

几何入口不是由 program DSL 排号。活跃的根部 `gbuffers_*`、`dh_*` wrapper include `GBufferSolid.*.glsl` 或
`GBufferTranslucent.*.glsl`，部分 wrapper 明确接到 NOOP；`voxy_*` 则直接实现 `voxy_emitFragment`。它们写入后续 composite
使用的材质、深度、法线和半透明数据。详见[几何、G-buffer 与材质](geometry-materials.md)和[水与半透明](water-translucency.md)。

## 场景准备

| 顺序                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Pass 流程                                                                                                                |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| [`VoxyMerge`](../../../shaders/pass/composite/VoxyMerge.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | 可选 Voxy merge                                                                                                          |
| [`EnvProbeUpdate1ReprojectScatter`](../../../shaders/pass/composite/EnvProbeUpdate1ReprojectScatter.comp.glsl)、[`HiZGen`](../../../shaders/pass/composite/HiZGen.csh)、[`EnvProbeUpdate2ReprojectDilate`](../../../shaders/pass/composite/EnvProbeUpdate2ReprojectDilate.comp.glsl)、[`GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl)、[`ShadowSampleSetup`](../../../shaders/pass/composite/ShadowSampleSetup.comp.glsl)、[`GIDenoiserEdgeDilation`](../../../shaders/pass/composite/GIDenoiserEdgeDilation.comp.glsl)、[`GIDenoiserReproject`](../../../shaders/pass/composite/GIDenoiserReproject.comp.glsl) | environment-probe reproject/scatter/dilate；Hi-Z；denoiser edge classification/dilation/reprojection；shadow-sample setup |
| [`EvaluateScreenPixelSize`](../../../shaders/pass/composite/EvaluateScreenPixelSize.comp.glsl)、[`CausticsPhotonTrace`](../../../shaders/pass/composite/CausticsPhotonTrace.comp.glsl)、[`CausticsRemap`](../../../shaders/pass/composite/CausticsRemap.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 可选 water-caustics：pixel size → photon trace → remap                                                                    |
| [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl)、[`clouds/ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 可选 cumulus render → temporal/spatial accumulation                                                                      |
| [`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl)、[`ShadowSample`](../../../shaders/pass/composite/ShadowSample.comp.glsl)、[`EnvProbeUpdate3ReprojectGather`](../../../shaders/pass/composite/EnvProbeUpdate3ReprojectGather.comp.glsl)、[`DirectLighting`](../../../shaders/pass/composite/DirectLighting.glsl)                                                                                                                                                                                                                                                                                                                                                                         | SSS shadow samples → main shadow sample，同时完成 environment-probe gather 和 direct lighting                                |
| [`DOFFocus`](../../../shaders/pass/composite/DOFFocus.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | 自动 DOF focus（仅启用 DOF 且非手动 focus）                                                                                       |
| [`EnvProbeUpdate4ProjectCurrent`](../../../shaders/pass/composite/EnvProbeUpdate4ProjectCurrent.comp.glsl)、[`GIReSTIRInitalSampleRayGenTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayGenTrace.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | environment probe project-current 与 GI initial ray generation/trace                                                    |

[`ShadowSampleSSS`](../../../shaders/pass/composite/ShadowSampleSSS.comp.glsl) 使用 SSBO 0 offset 32 的 indirect
dispatch。[`EnvProbeUpdate2ReprojectDilate`](../../../shaders/pass/composite/EnvProbeUpdate2ReprojectDilate.comp.glsl) 通过
`define("PASS", 1/2)` 复用同一入口。

## ReSTIR GI 与降噪

| 顺序 | Pass / 阶段                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 作用                                                                          |
|----|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| 1  | [`GIReSTIRInitalSampleRaySort`](../../../shaders/pass/composite/GIReSTIRInitalSampleRaySort.comp.glsl)、[`GIReSTIRInitalSampleRayFinishTrace`](../../../shaders/pass/composite/GIReSTIRInitalSampleRayFinishTrace.comp.glsl)                                                                                                                                                                                                                                                                                         | 初始 SST 路径较长时可选 sort/finish（`SETTING_GI_INITIAL_SST_STEPS >= 64`）            |
| 2  | [`GIReSTIRTemporalReuse`](../../../shaders/pass/composite/GIReSTIRTemporalReuse.comp.glsl)、[`GIReSTIRDuplicationMapDecorrelate`](../../../shaders/pass/composite/GIReSTIRDuplicationMapDecorrelate.comp.glsl)                                                                                                                                                                                                                                                                                                       | Temporal reuse 与可选 duplication-map decorrelation                            |
| 3  | [`GIReSTIRPairedSpatialReuse`](../../../shaders/pass/composite/GIReSTIRPairedSpatialReuse.comp.glsl) × 1–4                                                                                                                                                                                                                                                                                                                                                                                                          | 最多四个 indirect 批次；阈值为 reuse count `> 0/7/14/21`，从 SSBO 0 offset 48 dispatch  |
| 4  | [`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl)、[`GIReSTIRSpatialReuseRaySort`](../../../shaders/pass/composite/GIReSTIRSpatialReuseRaySort.comp.glsl)、[`GIReSTIRSpatialReuseTrace`](../../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl)                                                                                                                                                                                                      | 对选中样本 shading，整理 spatial ray，并完成 trace                                      |
| 5  | [`GIDenoiserAccum`](../../../shaders/pass/composite/GIDenoiserAccum.comp.glsl)、[`GIDenoiserAntiFireFly`](../../../shaders/pass/composite/GIDenoiserAntiFireFly.comp.glsl)、[`GIDenoiserGIMip`](../../../shaders/pass/composite/GIDenoiserGIMip.comp.glsl)、[`GIDenoiserHistoryFix`](../../../shaders/pass/composite/GIDenoiserHistoryFix.comp.glsl)、[`GIDenoiserBlur`](../../../shaders/pass/composite/GIDenoiserBlur.comp.glsl)、[`GIDenoiserPostBlur`](../../../shaders/pass/composite/GIDenoiserPostBlur.comp.glsl) | Denoiser accumulation、可选 anti-firefly、GI mip、history fix 与可选 blur/post-blur |
| 6  | [`SSTStepDebug`](../../../shaders/pass/composite/SSTStepDebug.comp.glsl)                                                                                                                                                                                                                                                                                                                                                                                                                                            | 可选 SST-step debug                                                           |

详见[全局光照模块](global-illumination.md)。

## 体积与半透明

[`EpipolarScatteringAir`](../../../shaders/pass/composite/EpipolarScatteringAir.comp.glsl)
↓
[`EpipolarScatteringWater`](../../../shaders/pass/composite/EpipolarScatteringWater.comp.glsl)
↓
[`TranslucentBackComposite`](../../../shaders/pass/composite/TranslucentBackComposite.glsl)
↓
[`TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl)
↓
[`TranslucentComposite`](../../../shaders/pass/composite/TranslucentComposite.glsl)
↓
[`IMapCollapse`](../../../shaders/techniques/rtwsm/IMapCollapse.comp.glsl)
↓
可选 [`VolumetricLocalCompositeBreakFix`](../../../shaders/pass/composite/VolumetricLocalCompositeBreakFix.comp.glsl)

最后一项从 SSBO 0 offset 0 indirect dispatch。

## 后处理

| 范围                                                                                                                                                                                                                                                                | 流程                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`DOFPrepare`](../../../shaders/pass/composite/DOFPrepare.comp.glsl)                                                                                                                                                                                              | 可选 DOF prepare                                                                |
| [`TAAPrepare`](../../../shaders/pass/composite/TAAPrepare.comp.glsl) → [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) → [`FXAA`](../../../shaders/pass/composite/FXAA.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | 内部非 FSR3 的时序 AA、可选空间 AA 与锐化                                          |
| [`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) → [`FSR3PrepareInputs`](../../../shaders/pass/composite/FSR3PrepareInputs.comp.glsl) → FSR3 pyramid/reactivity 阶段 → [`FSR3Accumulate`](../../../shaders/pass/composite/FSR3Accumulate.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | 内部 FSR3 时域升采样与公共 RCAS 输出                                               |
| [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl)                                                                                                                                                                                                            | bloom downsample levels 1–10 与 upsample levels 10–2，按 `SETTING_BLOOM_PASS` 截断 |
| [`IMapBlur`](../../../shaders/techniques/rtwsm/IMapBlur.comp.glsl) → [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl)                                                                                                                   | RTWSM importance blur、post composite 与显示变换                                  |
| [`GetWarp`](../../../shaders/techniques/rtwsm/GetWarp.comp.glsl) → [`ExposureMip`](../../../shaders/pass/composite/ExposureMip.comp.glsl)                                                                                                                         | next-frame RTWSM warp 与 exposure mip                                          |
| [`ExposureGather`](../../../shaders/pass/composite/ExposureGather.comp.glsl) → [`Write2DWarp`](../../../shaders/techniques/rtwsm/Write2DWarp.comp.glsl)                                                                                                           | exposure gather 与 RTWSM 2D warp write                                         |
| [`FinalGlobalDataUpdate`](../../../shaders/pass/composite/FinalGlobalDataUpdate.comp.glsl) → [`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl)                                                                                     | final global-data update 与 overlay composite                                  |
| [`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) → [`superresolution.v3.json`](../../../shaders/superresolution.v3.json)，在 [`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) 后触发 | 外部 SR 读取 `colortex31` motion 与完成的 SDR 帧；被禁用的内部 AA/FSR3/RCAS pass 会跳过 |

根部 [`final.fsh`](../../../shaders/final.fsh) include [
`Final.frag.glsl`](../../../shaders/pass/composite/Final.frag.glsl)
，对完成的 `colortex0` 图像执行 dither 并输出屏幕。详见[后处理模块](post-processing.md)。

## 条件与生成

`cond(...)` 会生成 `program.<name>.enabled` 的预处理分支；`indirect(...)` 生成 dispatch buffer/offset；`define(...)` 是普通
define，`constDefine(...)` 放入 Iris 的 const 区块。

修改 program DSL 后运行：

```sh
cd scripts
kotlin programs.main.kts
```

若 program 属性片段有变化，再运行：

```sh
cd scripts
kotlin options.main.kts
```

这会将它汇总进最终 [`shaders/shaders.properties`](../../../shaders/shaders.properties)；options 会再次调用 program 生成器。
