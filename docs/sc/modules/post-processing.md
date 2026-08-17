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
| [`GenerateMotionVectors.comp.glsl`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl)、[`FSR3Accumulate.comp.glsl`](../../../shaders/pass/composite/FSR3Accumulate.comp.glsl) | 公共 motion 生成、内部 FSR3 输入与累积          |
| [`FXAA.comp.glsl`](../../../shaders/pass/composite/FXAA.comp.glsl)                                                                                                              | 空间抗锯齿                            |
| [`RCAS.comp.glsl`](../../../shaders/pass/composite/RCAS.comp.glsl)、[`techniques/ffx/fsr1/`](../../../shaders/techniques/ffx/fsr1/)                                              | RCAS 锐化                          |
| [`techniques/Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl)                                                                                                     | Bloom downsample/upsample 金字塔    |
| [`ExposureMip.comp.glsl`](../../../shaders/pass/composite/ExposureMip.comp.glsl)、[`ExposureGather.comp.glsl`](../../../shaders/pass/composite/ExposureGather.comp.glsl)         | 自动曝光权重、mip 与统计                   |
| [`PostComposite.comp.glsl`](../../../shaders/pass/composite/PostComposite.comp.glsl)、[`OverlayComposite.comp.glsl`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | 主后期合成与 overlay                   |
| [`superresolution.v3.json`](../../../shaders/superresolution.v3.json)                                                                                                              | 外部 Super Resolution 接口与触发点        |
| [`techniques/displaytransform/`](../../../shaders/techniques/displaytransform/)                                                                                                 | 曝光、DRT 与显示变换                     |
| [`FinalGlobalDataUpdate.comp.glsl`](../../../shaders/pass/composite/FinalGlobalDataUpdate.comp.glsl)、[`Final.frag.glsl`](../../../shaders/pass/composite/Final.frag.glsl)       | 下一帧状态与最终屏幕输出                     |

## DOF

[`DOFFocus`](../../../shaders/pass/composite/DOFFocus.comp.glsl) 仅在启用 DOF 且未启用 manual focus
时运行。透明/体积完成后，[`DOFPrepare`](../../../shaders/pass/composite/DOFPrepare.comp.glsl) 将当前主颜色写入
`transient_dofInput`，供后续后期合成使用。manual focus 只跳过 focus pass，prepare 仍会运行。

设置包括 focal length、f-stop、aperture shape、quality、maximum sample radius、masking heuristic、三段 manual-focus
distance、focus time 和 focus-plane debug。

## 抗锯齿与超采样

[`TAAPrepare`](../../../shaders/pass/composite/TAAPrepare.comp.glsl) 在管线分支前应用公共的 DOF 输入。随后 program list
启用以下路径之一：

| 路径 | Pass 流程 | 作用 |
|------|-----------|------|
| 关闭 | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) | 直接写入未滤波的当前帧，不执行时域或空间抗锯齿。 |
| TAA | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) → [`FXAA`](../../../shaders/pass/composite/FXAA.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | Resolve `history_taa`、执行空间抗锯齿，并锐化渲染分辨率输出。 |
| FSR 3 | [`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) → [`FSR3PrepareInputs`](../../../shaders/pass/composite/FSR3PrepareInputs.comp.glsl) → [`FSR3LumaPyramid`](../../../shaders/pass/composite/FSR3LumaPyramid.comp.glsl) → [`FSR3ShadingChangePyramid`](../../../shaders/pass/composite/FSR3ShadingChangePyramid.comp.glsl) → [`FSR3ShadingChange`](../../../shaders/pass/composite/FSR3ShadingChange.comp.glsl) → [`FSR3PrepareReactivity`](../../../shaders/pass/composite/FSR3PrepareReactivity.comp.glsl) → [`FSR3LumaInstability`](../../../shaders/pass/composite/FSR3LumaInstability.comp.glsl) → [`FSR3Accumulate`](../../../shaders/pass/composite/FSR3Accumulate.comp.glsl) → [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) | 构建 motion/reactive 输入、累积全分辨率结果，再通过公共 RCAS pass 执行锐化并输出已曝光的线性颜色。 |
| 外部 SR | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) → [`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) → 外部 SR → 常规输出分辨率后处理 | 跳过内部 TAA、FXAA、FSR3 累积和 RCAS；SR 在 Bloom 前立即重建 exposed-linear HDR。 |

抗锯齿 / 超采样 screen 控制模式、渲染分辨率比例、jitter、TAA current/history filter 和公共 RCAS 锐化强度。current/previous jitter custom uniform 在 [
`scripts/shaders.properties`](../../../scripts/shaders.properties) 中由 R2 frame sequence 生成；改变采样序列时，也要同步更新
reprojection 约定。

`SR_ENABLE` 激活时，外部接口通过 `SR_RENDER_SCALE_FACTOR` 与 `SRJitterOffset` 接管渲染比例和 jitter。无论
`SETTING_AA_MODE` 的值为何，内部 TAA、FXAA、FSR3 累积和 RCAS 都会禁用。仅帧生成模式仍通过同一接口保持激活，但
`SR_SHOULD_APPLY_SCALE` 与 `SR_SHOULD_APPLY_JITTER` 为零，因此光影以原生分辨率、无 jitter 渲染。

TAA 与 FSR 3 会把相同的可逆 matrix/log 工作域交给公共 RCAS 实现，并直接使用所选锐化强度。强度为零时会旁路空间滤波并返回
中心样本；此时只保留必需的显示曝光和可逆工作域往返。关闭模式不会调度 RCAS。所有模式都会在输出分辨率 Bloom 和显示变换之前
写出 exposed-linear `main`。

FSR 3 输入渲染分辨率、未曝光的 scene-linear HDR，并以相同的 scene-linear 域保存颜色与亮度历史。它的逐帧重建曝光会等量缩放
当前值和历史值，并在保存前除回；随后公共 RCAS 只应用一次显示曝光。completed-frame marker 与公共时序 reset 状态会在重载、模式
切换或其他时序中断后拒绝陈旧历史。Motion 使用 current-to-previous UV 单位，jitter 使用渲染像素单位。项目自有的 accumulate
入口会用可逆 AgX inset-matrix/log 变换替换 SDK 的 max-channel 重建 tonemap；带偏移的 log 能精确保留黑色且不会硬截暗部 EV，
FSR 专用范围可覆盖 FP16 输入乘以最大逐帧重建曝光。

Reactive/composition mask 覆盖动态或无法跟踪物体变换的 solid 表面与 overlay。半透明 SST 已经合成进输入颜色，并有意沿用下层
solid 的 depth、motion 与 mask；它既不单独写 reactive mask，也不再使用 SST 时域降噪，因为两者会让粗糙反射/折射噪声逐帧闪烁。

[`GenerateMotionVectors`](../../../shaders/pass/composite/GenerateMotionVectors.comp.glsl) 只计算一次公共的
current-to-previous UV 向量。编译期输出宏会将其写入内部 FSR3 history，并附带 reactive/composition mask，或写入供外部
SR 使用的渲染分辨率 `colortex31.rg`。外部路径使用 `colortex0` color、`noTranslucentDepthtex` depth 和
`colortex31` motion，并在被禁用的内部 FSR3 与 RCAS program 之后、Bloom 之前立即触发，因此输入是常量预曝光 `1.0` 的
exposed-linear HDR。SR 把全分辨率结果写回 `colortex0`，随后 Bloom、显示变换、曝光分析与 overlay 合成都在输出分辨率运行。
仅帧生成模式仍读取三项输入，但不会写入 color 输出；此时渲染与后处理都使用原生分辨率。

当前 motion vector 覆盖相机重投影、天空、手部和 solid 表面，但不包含骨骼动画、粒子或任意程序化形变的完整逐物体运动；这些
表面可能产生外部超采样或帧生成伪影。屏幕 overlay 也会沿用下层场景的 motion。

内部 FSR 3 与外部 SR 都在渲染分辨率运行，并在 Bloom 前重建到 view/output 分辨率；后续所有后处理都使用输出分辨率。G-buffer 显式梯度
会把低分辨率 raster derivative 乘以 `0.5 * uval_mainImageScale`，等价于按实际逐轴渲染比例应用 AMD 的
`log2(render/output) - 1` mip bias。

## Bloom

启用 `SETTING_BLOOM` 时，[`Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl) 通过 `BLOOM_DOWN_SAMPLE` 和
`BLOOM_PASS=1..10` 构建第 1–10 层，再通过 `BLOOM_UP_SAMPLE` 重建第 10–2 层。`SETTING_BLOOM_PASS` 在 program 层禁用未使用的高层。
渲染分辨率路径使用 `transient_bloom`；内部 FSR3 在 accumulation 和 RCAS 后复用 `usam_fsr3UpscaleAtlas` 的第三个区域，
外部 SR 使用专用的全分辨率 `usam_superResolutionBloom` image。仅帧生成模式以原生分辨率渲染，因此仍使用 `transient_bloom`。
所有打包金字塔读取都会夹到源 tile 的 texel center 范围内，防止双线性过滤越界混入相邻 tile 或 atlas 陈旧数据。

Bloom 高光压缩是有意设计的有损 Bloom 专用操作。它只对进入第一个 downsample 层的 exposed-linear 主颜色样本应用一次，
并且位于金字塔过滤之前。它不会修改主图像，也不属于 [`AgxInvertible.glsl`](../../../shaders/util/AgxInvertible.glsl)
中的可逆 matrix/log AA 工作变换。

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
