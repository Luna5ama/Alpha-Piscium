# 后处理与显示变换

语言：简体中文 | [English](../../en/modules/post-processing.md)

后处理在 GI、空气/水体体积和半透明合成完成后运行。主链是可选 DOF、时序 resolve、空间抗锯齿/锐化、bloom、后期合成、曝光、overlay
和最终显示变换。

## 代码位置

| 路径                                                                                                                                                                              | 职责                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| [`shaders/techniques/DOF.glsl`](../../../shaders/techniques/DOF.glsl)                                                                                                           | DOF 公共采样与 circle-of-confusion 逻辑 |
| [`DOFFocus.comp.glsl`](../../../shaders/pass/composite/DOFFocus.comp.glsl)、[`DOFPrepare.comp.glsl`](../../../shaders/pass/composite/DOFPrepare.comp.glsl)                       | 自动 focus 与 DOF 输入准备              |
| [`TAAPrepare.comp.glsl`](../../../shaders/pass/composite/TAAPrepare.comp.glsl)、[`TAAResolve.comp.glsl`](../../../shaders/pass/composite/TAAResolve.comp.glsl)                   | 时序 AA 准备与 resolve                |
| [`FSR3MotionVectors.comp.glsl`](../../../shaders/pass/composite/FSR3MotionVectors.comp.glsl)、[`FSR3Accumulate.comp.glsl`](../../../shaders/pass/composite/FSR3Accumulate.comp.glsl) | FSR3 输入、时域升采样与累积                 |
| [`FXAA.comp.glsl`](../../../shaders/pass/composite/FXAA.comp.glsl)                                                                                                              | 空间抗锯齿                            |
| [`RCAS.comp.glsl`](../../../shaders/pass/composite/RCAS.comp.glsl)、[`techniques/ffx/fsr1/`](../../../shaders/techniques/ffx/fsr1/)                                              | RCAS 锐化                          |
| [`techniques/Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl)                                                                                                     | Bloom downsample/upsample 金字塔    |
| [`ExposureMip.comp.glsl`](../../../shaders/pass/composite/ExposureMip.comp.glsl)、[`ExposureGather.comp.glsl`](../../../shaders/pass/composite/ExposureGather.comp.glsl)         | 自动曝光权重、mip 与统计                   |
| [`PostComposite.comp.glsl`](../../../shaders/pass/composite/PostComposite.comp.glsl)、[`OverlayComposite.comp.glsl`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | 主后期合成与 overlay                   |
| [`techniques/displaytransform/`](../../../shaders/techniques/displaytransform/)                                                                                                 | 曝光、DRT 与显示变换                     |
| [`FinalGlobalDataUpdate.comp.glsl`](../../../shaders/pass/composite/FinalGlobalDataUpdate.comp.glsl)、[`Final.frag.glsl`](../../../shaders/pass/composite/Final.frag.glsl)       | 下一帧状态与最终屏幕输出                     |

## DOF

[`DOFFocus`](../../../shaders/pass/composite/DOFFocus.comp.glsl) 仅在启用 DOF 且未启用 manual focus
时运行。透明/体积完成后，[`DOFPrepare`](../../../shaders/pass/composite/DOFPrepare.comp.glsl) 将当前主颜色写入
`transient_dofInput`，供后续后期合成使用。manual focus 只跳过 focus pass，prepare 仍会运行。

设置包括 focal length、f-stop、aperture shape、quality、maximum sample radius、masking heuristic、三段 manual-focus
distance、focus time 和 focus-plane debug。

## 时序 AA 与升采样

[`TAAPrepare`](../../../shaders/pass/composite/TAAPrepare.comp.glsl) 在管线分支前应用公共的 DOF 输入。随后 program list
启用以下路径之一：

| 路径 | Pass 流程 | 作用 |
|------|-----------|------|
| 非 FSR3 | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) → 可选 [`FXAA`](../../../shaders/pass/composite/FXAA.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | Resolve `history_taa`、执行空间抗锯齿，并锐化渲染分辨率输出。关闭 TAA 时，`TAAResolve` 直接写入未滤波结果，并跳过 FXAA/RCAS。 |
| FSR3 | [`FSR3MotionVectors`](../../../shaders/pass/composite/FSR3MotionVectors.comp.glsl) → [`FSR3PrepareInputs`](../../../shaders/pass/composite/FSR3PrepareInputs.comp.glsl) → [`FSR3LumaPyramid`](../../../shaders/pass/composite/FSR3LumaPyramid.comp.glsl) → [`FSR3ShadingChangePyramid`](../../../shaders/pass/composite/FSR3ShadingChangePyramid.comp.glsl) → [`FSR3ShadingChange`](../../../shaders/pass/composite/FSR3ShadingChange.comp.glsl) → [`FSR3PrepareReactivity`](../../../shaders/pass/composite/FSR3PrepareReactivity.comp.glsl) → [`FSR3LumaInstability`](../../../shaders/pass/composite/FSR3LumaInstability.comp.glsl) → [`FSR3Accumulate`](../../../shaders/pass/composite/FSR3Accumulate.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | 构建 motion/reactive 输入、累积全分辨率结果，再通过公共 RCAS pass 执行可选锐化和最终颜色空间转换。 |

TAA 设置包括 enable、jitter、current/history filter 和 CAS sharpness。current/previous jitter custom uniform 在 [
`scripts/shaders.properties`](../../../scripts/shaders.properties) 中由 R2 frame sequence 生成；改变采样序列时，也要同步更新
reprojection 约定。

## Bloom

启用 `SETTING_BLOOM` 时，[`Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl) 通过 `BLOOM_DOWN_SAMPLE` 和
`BLOOM_PASS=1..10` 构建第 1–10 层，再通过 `BLOOM_UP_SAMPLE` 重建第 10–2 层。`SETTING_BLOOM_PASS` 在 program 层禁用未使用的高层。
非 FSR3 路径使用 `transient_bloom`；FSR3 路径在 accumulation 和 RCAS 后复用 `usam_fsr3UpscaleAtlas` 的第三个区域。

## 后期合成与曝光

| 顺序 | Pass                                                                                                                                                                          | 作用                                                  |
|----|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| 1  | [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl)                                                                                                    | 合并主颜色、DOF、bloom 和其他后期输入；RTWSM companion pass 使用独立资源 |
| 2  | [`ExposureMip`](../../../shaders/pass/composite/ExposureMip.comp.glsl)                                                                                                        | 构建亮度/权重层                                            |
| 3  | [`ExposureGather`](../../../shaders/pass/composite/ExposureGather.comp.glsl)                                                                                                  | 汇总曝光统计                                              |
| 4  | [`FinalGlobalDataUpdate`](../../../shaders/pass/composite/FinalGlobalDataUpdate.comp.glsl) 与 [`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | 保存下一帧的 global/exposure state 并合成 overlay            |

曝光 screen 提供 manual EV、最小/最大 EV、emissive/distance/center weighting、average-luminance time 和 target
range、highlight/shadow percentile 及 adaptation 参数。`transient_exposureWeights` 保存逐像素权重。

## 最终显示变换

根部 [`final.fsh`](../../../shaders/final.fsh) include [
`Final.frag.glsl`](../../../shaders/pass/composite/Final.frag.glsl)。该入口使用 [
`Exposure.glsl`](../../../shaders/techniques/displaytransform/Exposure.glsl)、[
`DRT.glsl`](../../../shaders/techniques/displaytransform/DRT.glsl) 和 [
`DisplayTransform.glsl`](../../../shaders/techniques/displaytransform/DisplayTransform.glsl)，将内部 working color space
转为 DRT working space，再应用 tone mapping 和 output color space/transfer function。

tone mapping 后，结果先转换到配置的调色色彩空间；[`PrimaryColorCalibration`](../../../shaders/techniques/displaytransform/PrimaryColorCalibration.glsl) 调整其线性 RGB 原色，再应用配置的调色传递函数，并由 [`HSLColorMixer`](../../../shaders/techniques/displaytransform/HSLColorMixer.glsl) 调整八个色相区间。结果随后被解码并转换到显示器输出空间。两个阶段分别由独立开关控制。

相关设置包括 material transfer/color space、internal working space、DRT working space、调色色彩空间/传递函数、tone-map look/dynamic range/offset/slope/power/saturation，以及显示器 color space/transfer function。

## 调试与特殊模式

[`shaders/techniques/debug/`](../../../shaders/techniques/debug/) 的 debug output 可以查看 TAA、PostFX、tone mapping 或
final 边界；[`SSTStepDebug`](../../../shaders/pass/composite/SSTStepDebug.comp.glsl) 是条件 program。screenshot mode
会调整动画/temporal clamp，video-render mode 会调整长期 temporal effect；它们不构成并行的生产后处理分支。

## 维护约束

- TAA history 的颜色空间、pre-exposure、jitter 和 motion/reprojection 必须一致。
- Bloom pass count 同时影响 program enable 条件和金字塔访问范围。
- exposure state 在帧尾更新；final 不得读取半更新状态。
- 显示变换改动要用数值色卡/渐变验证，并在 SDR、wide-gamut 和 HDR 配置上检查 clipping、NaN 和负值。
- 实测静止细节、运动边缘、disocclusion、高亮 emissive、暗场适应、透明边缘与 UI/overlay。
