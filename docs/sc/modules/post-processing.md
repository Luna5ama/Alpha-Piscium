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

`SETTING_BLOOM_SHAPE` 选择标准（0，默认）、四向星芒（1）或横向光带（2）。标准版是具有集中核心的八尺度长尾高斯近似，
每个尺度占归一化滤波权重的 1/8，并非单一高斯。固定八层保留宽光晕；此前否决的四层原型丢失了这一远尾。

滤波使用 [`BloomTaps.glsl`](../../../shaders/techniques/BloomTaps.glsl) 中离线拟合的偏移与非负权重。
拟合考虑抽取相位和双线性采样，采用 [DPF 的思路](https://huakaifugui2.github.io/CZ-DPF-Paper/)。运行时为固定 RGB 滤波和
固定深度金字塔。四尺度拟合代理只校准可复用的 tap，不决定运行时层数。

| 顺序 | Pass/阶段 | 作用 |
|---|---|---|
| 1 | [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl)，下采样第 1–8 层 | 每输出 texel 五次双线性读取。首层读取解码后的线性主图，并施加仅用于 Bloom 的高光压缩。 |
| 2 | 可选 [`BloomShape`](../../../shaders/pass/composite/BloomShape.comp.glsl) | 重建前读取原始第 1 层，计算生成的方向核，写入独立半分辨率图像。标准版及关闭 Bloom 时禁用此 pass。 |
| 3 | [`Bloom`](../../../shaders/techniques/Bloom.comp.glsl)，从第 8–2 层重建 | 每输出 texel 八次双线性读取。最深层贡献及每层较细下采样各赋权 1/8；invocation 读取其即将覆盖的较细 texel。 |
| 4 | [`PostComposite`](../../../shaders/pass/composite/PostComposite.comp.glsl) | 融合末级上采样，以中心双线性重建叠加可选形状，再应用泛光强度与水下增强。 |

[`programs.main.kts`](../../../scripts/programs.main.kts) 管理顺序与启用条件，
[`Bloom.glsl`](../../../shaders/techniques/Bloom.glsl) 管理共用寻址和采样。下采样及形状读取对屏幕外的双线性采样覆盖范围
做零延拓，同时把 atlas 读取约束在输入图块内；上采样钳制到有效输入 mip。这避免长光带重复复制边缘亮 texel。

- `transient_bloom`：帧内 RGBA16F atlas，使用向上取整且带 padding 的 mip 图块，从粗到细原地重建。
- `transient_bloomShape`：帧内 RGBA16F 逻辑图像，只读写其半分辨率区域，生命周期从形状 pass 到后期合成。
  [`shadesmith.json`](../../../shaders/shadesmith.json) 中的分配复用 FXAA/RCAS 已用完的槽位，当前物理 atlas 尺寸不增加；
  它在逻辑上与普通泛光金字塔分离。

星芒有四条对角臂；横带组合较亮内段与更长的柔和带。两者保留完整标准核心和长尾，再增加方向性能量，增益分别为 0.375 和 0.5。
生成核分别使用 109 和 72 次双线性读取。从原始第 1 层生成形状，避免在已经模糊的累积 mip 上造型。
参数及自定义 PNG/函数入口由 [`fit-bloom-shape.py`](../../../scripts/fit-bloom-shape.py) 维护，参见[核拟合](../development/scripts.md#bloom-核拟合)。

高光压缩属于 AA 解码后的 Bloom 首次下采样，RGB 与亮度模式保留既有曲线。AA 保留原有对数编码及暗部下限，不应用 Bloom 压缩；
[`RCAS`](../../../shaders/pass/composite/RCAS.comp.glsl) 和关闭 TAA 时的
[`TAAResolve`](../../../shaders/pass/composite/TAAResolve.comp.glsl) 在解码写回处钳制到 RGBA16F 上限 65504。

### Bloom 验证与限制

2026-11-08 接受的 shader 快照 `692e264d…` 在 Minecraft 1.21.11、Iris 1.10.6-vibris.3、RTX 3080 Ti 上检查。
五个 1280×720 场景包括小悬灯夜间、Craft 近灯、室外点光源、Non-cube 建筑与 Frutiger 室内；每场都有标准、星芒、横带和
关闭 Bloom 的完整截图，同组相机、时间及固定曝光一致，预热 192 帧。关闭 TAA 时形状输出有限且非零；关闭 Bloom 时形状 pass 不执行。

小悬灯夜间、固定 EV 9.45，正反顺序配对，每窗预热 128 帧并采样 128 帧：

| 模式 | Bloom 与整个 PostComposite 的 p50 / p95 | 相对配对标准版的 p50 增量 |
|---|---:|---:|
| 标准 | 0.190–0.191 / 0.199–0.200 ms | — |
| 四向星芒 | 0.275 / 0.282 ms | 0.084 ms |
| 横向光带 | 0.279 / 0.286 ms | 0.088 ms |

计时先对同帧 program 样本求和，再取分位数，范围不含整帧。RCAS 和未改动的体积合成对照 pass 中位数稳定。
集成前核实接受的 shader 文件和重新生成的内置权重与实测源完全一致。

独立 CPU 脉冲模型显示标准核接近径向对称：半径 12、24、48、96 像素处，八个相位的平均四重角向分量约为环上亮度的
1.0–1.9%，旧 COD 滤波约为 2.6–3.2%。二倍抽取仍存在像素栅格和相位误差，并非严格旋转对称。
奇数视口、边界能量和相位质心有 CPU 检查；奇数尺寸实机、水下及其他 GPU 尚未测量。
形状拟合误差描述原始第 1 层的新增支路，不代表完整 Bloom 结果。

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
