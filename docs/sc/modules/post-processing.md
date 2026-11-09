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
| [`FXAA.comp.glsl`](../../../shaders/pass/composite/FXAA.comp.glsl)                                                                                                              | 空间抗锯齿                            |
| [`RCAS.comp.glsl`](../../../shaders/pass/composite/RCAS.comp.glsl)、[`techniques/ffx/fsr1/`](../../../shaders/techniques/ffx/fsr1/)                                              | RCAS 锐化                          |
| [`techniques/Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl)                                                                                                     | Bloom downsample/upsample 金字塔    |
| [`ExposureMip.comp.glsl`](../../../shaders/pass/composite/ExposureMip.comp.glsl)、[`ExposureGather.comp.glsl`](../../../shaders/pass/composite/ExposureGather.comp.glsl)         | 自动曝光权重、mip 与统计                   |
| [`PostComposite.comp.glsl`](../../../shaders/pass/composite/PostComposite.comp.glsl)、[`OverlayComposite.comp.glsl`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | 主后期合成与 overlay                   |
| [`techniques/displaytransform/`](../../../shaders/techniques/displaytransform/)                                                                                                 | 曝光、DRT 与显示变换                     |
| [`Final`](../../../shaders/pass/composite/Final.frag.glsl) | 最终屏幕输出 |

## DOF

[`DOFFocus`](../../../shaders/pass/composite/DOFFocus.comp.glsl) 仅在启用 DOF 且未启用 manual focus
时运行。透明/体积完成后，[`DOFPrepare`](../../../shaders/pass/composite/DOFPrepare.comp.glsl) 将当前主颜色写入
`transient_dofInput`，供后续后期合成使用。manual focus 只跳过 focus pass，prepare 仍会运行。

设置包括 focal length、f-stop、aperture shape、quality、maximum sample radius、masking heuristic、三段 manual-focus
distance、focus time 和 focus-plane debug。

## 时序与空间 AA

| 顺序 | Pass                                                                 | 作用                                                                       |
|----|----------------------------------------------------------------------|--------------------------------------------------------------------------|
| 1  | [`TAAPrepare`](../../../shaders/pass/composite/TAAPrepare.comp.glsl) | 生成 temporal input、luma difference 和辅助数据                                  |
| 2  | [`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) | 使用 `history_taa`、motion/reprojection data 和当前帧输入写入 `transient_taaOutput` |
| 3  | [`FXAA`](../../../shaders/pass/composite/FXAA.comp.glsl)             | 写入 `transient_fxaaOutput`                                                |
| 4  | [`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl)             | 执行最终空间锐化                                                                 |

四个 pass 都存在于 program list 中；setting 在 shader 内选择行为，而不是启用/禁用不同的 program number。

TAA 设置包括 enable、jitter、current/history filter 和 CAS sharpness。current/previous jitter custom uniform 在 [
`scripts/shaders.properties`](../../../scripts/shaders.properties) 中由 R2 frame sequence 生成；改变采样序列时，也要同步更新
reprojection 约定。

## Bloom

`SETTING_BLOOM_SHAPE` 提供标准（0，默认）、菱形四芒（1）、横向光带（2）、正十字（3）、摄影六芒（4）、
摄影十芒（5）、电影星芒（6）、棱镜光带（7）、67（8）、Trollface（9）与自定义（10）。
标准版以相同权重组合八个尺度，形成集中的核心和长尾。其他预设拟合包含预期光晕与造型的完整 RGB 响应，
并以独立通道权重实现色散。编辑器和导出方式见 [Bloom 核拟合](../development/scripts.md#bloom-核拟合)。

| 顺序 | Pass/阶段 | 作用 |
|---|---|---|
| 1 | [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl)，下采样第 1–8 层 | 每个输出 texel 五次双线性读取。首层读取解码后的线性主图，并对每次读取施加 Bloom 高光压缩。 |
| 2 | [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl)，从第 8 层向第 1 层重建 | 上采样较粗层的结果，累加原始金字塔中同层的采样，再写入合并结果。 |
| 3 | [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl) | 将第 1 层上采样到视口大小，在显示变换前按泛光强度和水下设置累加 Bloom。 |

[`programs.main.kts`](../../../scripts/programs.main.kts) 管理顺序与启用条件。
[`Bloom.glsl`](../../../shaders/techniques/Bloom.glsl) 将读取限制在输入 mip 图块内。
下采样和原始层注入对视口边缘的双线性采样范围做零延拓，重建则钳制到有效 mip。

- `transient_bloomRaw` 保存帧内 RGBA16F 下采样金字塔，mip 图块向上取整并带边距。
- `transient_bloomReconstruction` 保存从粗到细生成的帧内 RGBA16F 结果。
  [`shadesmith.json`](../../../shaders/shadesmith.json) 中的分配复用 FXAA 和 RCAS 用完的 atlas 空间。

高光压缩在 AA 解码后的 Bloom 首次下采样执行，RGB 与亮度模式沿用各自的曲线。
[`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) 和关闭 TAA 时的
[`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) 在解码写回时钳制到 RGBA16F 上限 65504。

### Bloom 验证与限制

CPU 拟合器检查各留出抽取相位的完整脉冲响应。二倍抽取会产生随相位变化的误差。较大的发光面会拓宽方向造型并填平细小间隙。
CPU 成本估算按金字塔各层加权计算双线性读取次数，GPU 成本还受所选采样布局和场景影响。在同一 Iris 配置下对照受影响的重建 pass、
PostComposite 和一个稳定的相邻 pass。

在 RTX 3080 Ti 的 720p 夜景中，Iris 实际运行了九个造型预设、关闭 Bloom 的分支和导出的自定义核。
关闭 Bloom 时，[`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl) 仍正常执行。
标准与自定义采用相同系数的 A/A 中，Bloom 前 HDR 均值差异小于 0.2%。用实际导出的自定义核做 ABBA 时，逐帧累加
Bloom 和 PostComposite 的 compute 耗时再取 p50，标准为 0.145 ms，自定义为 0.623 ms。两份配方的目标核不同，
这些结果反映各自的运行成本。旧增亮样本所在快照漏带了 PostComposite wrapper，该样本无效。
离线检查覆盖四个留出相位，调整版摄影六芒的导出核原点误差最高为 4.0%。编辑器已验证 HTTP 拟合和导出，浏览器实际渲染待验。

## 后期合成与曝光

| 顺序 | Pass                                                                                                                                                                          | 作用                                                  |
|----|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| 1  | [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl)                                                                                                    | 合并主颜色、DOF、bloom 和其他后期输入；RTWSM companion pass 使用独立资源 |
| 2  | [`ExposureMip`](../../../shaders/pass/composite/ExposureMip.comp.glsl)                                                                                                        | 构建亮度/权重层                                            |
| 3  | [`ExposureGather`](../../../shaders/pass/composite/ExposureGather.comp.glsl)                                                                                                  | 汇总曝光统计                                              |
| 4 | [`OverlayComposite`](../../../shaders/pass/composite/OverlayComposite.comp.glsl) | 合成 overlay |

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
- 保持 Bloom 八层、层权重、生成 tap 和 atlas 寻址一致。原始第 1 层载体变化时重新拟合方向形状。
- 保持曝光数据生成、适应和最终显示消费者同步。
- 显示变换改动要用数值色卡/渐变验证，并在 SDR、wide-gamut 和 HDR 配置上检查 clipping、NaN 和负值。
- 实测静止细节、运动边缘、disocclusion、高亮 emissive、暗场适应、透明边缘与 UI/overlay。
