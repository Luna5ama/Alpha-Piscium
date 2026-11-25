# Post-processing and Display Transform

Post-processing begins after GI, air/water volumes, and translucency are composed. The main chain is optional DOF,
metering, temporal resolve, spatial antialiasing/sharpening, camera exposure, Bloom, the display transform, overlay and final output.

## Code map

| Path | Responsibility |
| --- | --- |
| [`shaders/techniques/DOF.glsl`](../../shaders/techniques/DOF.glsl) | Shared DOF sampling and circle-of-confusion logic |
| [`DOFFocus.comp.glsl`](../../shaders/pass/composite/DOFFocus.comp.glsl), [`DOFPrepare.comp.glsl`](../../shaders/pass/composite/DOFPrepare.comp.glsl) | Automatic focus and DOF input preparation |
| [`TAAPrepare.comp.glsl`](../../shaders/pass/composite/TAAPrepare.comp.glsl), [`TAAResolve.comp.glsl`](../../shaders/pass/composite/TAAResolve.comp.glsl) | Temporal-AA preparation and resolve |
| [`FXAA.comp.glsl`](../../shaders/pass/composite/FXAA.comp.glsl) | Spatial antialiasing |
| [`RCAS.comp.glsl`](../../shaders/pass/composite/RCAS.comp.glsl), [`techniques/ffx/fsr1/`](../../shaders/techniques/ffx/fsr1/) | RCAS sharpening |
| [`techniques/Bloom.comp.glsl`](../../shaders/techniques/Bloom.comp.glsl) | Bloom downsample/upsample pyramid |
| [`Exposure.glsl`](../../shaders/techniques/displaytransform/Exposure.glsl), [`ExposureGather.comp.glsl`](../../shaders/pass/composite/ExposureGather.comp.glsl) | Weighted log-luminance histogram and camera adaptation |
| [`PostComposite.comp.glsl`](../../shaders/pass/composite/PostComposite.comp.glsl), [`OverlayComposite.comp.glsl`](../../shaders/pass/composite/OverlayComposite.comp.glsl) | Main post composition and overlay |
| [`techniques/displaytransform/`](../../shaders/techniques/displaytransform/) | Exposure, DRT, and display transform |
| [`Final`](../../shaders/pass/composite/Final.frag.glsl) | Final screen output |

## DOF

[`DOFFocus`](../../shaders/pass/composite/DOFFocus.comp.glsl) runs only when DOF is enabled without manual focus.
After volumes/translucency, [`DOFPrepare`](../../shaders/pass/composite/DOFPrepare.comp.glsl) writes the current main
color to `transient_dofInput` for later post composition. Manual focus skips only the focus pass; preparation still
runs.

Settings cover focal length, f-stop, aperture shape, quality, maximum sample radius, masking heuristic, three-part
manual-focus distance, focus time, and focus-plane debug.

## Temporal and spatial AA

| Order | Pass | Purpose |
| --- | --- | --- |
| 1 | [`TAAPrepare`](../../shaders/pass/composite/TAAPrepare.comp.glsl) | Meters unexposed post-DOF scene color and encodes the temporal input |
| 2 | [`TAAResolve`](../../shaders/pass/composite/TAAResolve.comp.glsl) | Uses `history_taa`, motion/reprojection data, and current-frame input to write `transient_taaOutput` |
| 3 | [`FXAA`](../../shaders/pass/composite/FXAA.comp.glsl) | Saves the unexposed scene-linear TAA history and writes `transient_fxaaOutput` |
| 4 | [`RCAS`](../../shaders/pass/composite/RCAS.comp.glsl) | Sharpens, decodes to scene linear, and applies the common camera EV |

TAAPrepare and TAAResolve also run with TAA disabled. Iris enables FXAA and RCAS only for TAA. On both frame parities, TAAResolve finishes reading the previous `history_taa` tile before FXAA writes the new scene-linear history to the same tile, at (width, 2 × height) in the RGBA16F image.

TAA settings cover enable, jitter, current/history filters, and CAS sharpness. Current/previous-jitter custom uniforms
are generated from the R2 frame sequence in [`scripts/shaders.properties`](../../scripts/shaders.properties);
changing the sampling sequence requires updating reprojection conventions as well.

### 2.1 时域重建策略对照

2.1 研究 AP 自己的统一时域重建器，可以复用或大改 FSR 3.1 的阶段。原生 TAA 和降比例超分共用同一套历史判断。下表比较公开策略，右栏是待验证的 AP 实验。

**AP 现有画面的三个条件。**

- **静止细节。** 同一表面重投影可靠、画面稳定时，AP 原生 TAA 使用对历史改动最小的邻域颜色 AABB。它在 YCoCg 中取包围邻域颜色的范围；范围内的历史可以保留。用均值和标准差构造的方差盒可能排除真实的少数亮色，连续多帧会把细线和纹理对比压低。实现按相机速度和转向平滑改变裁剪方式，在方差椭球、方差 AABB 和颜色 AABB 间过渡。水面与截图模式另有处理。新重建器以普通静止表面的颜色 AABB 结果作清晰度基准。[裁剪实现](../../shaders/pass/composite/TAAResolve.comp.glsl#L245-L300) · [相机运动与裁剪选择](../../shaders/pass/begin/UpdateGlobalData.comp.glsl#L239-L328)
- **粗糙玻璃。** 半透明屏幕空间反射与折射先合成进场景颜色，最终时域累积负责平均残余随机噪声。`649eaa672` 已同时移除透明表面向 reactive、composition mask 写入的专用覆盖，并撤下当时不可靠的独立 SST 时域降噪。现有原型随底层不透明深度和运动重投影，仍处理实体、overlay 与无效重投影。新方案须保留玻璃的累积机会，同时观察镜头横移时的视差错误和拖尾。[移除专用覆盖的提交](https://github.com/Luna5ama/Alpha-Piscium/commit/649eaa67287b9631d4411ec4cf4a1d8cd25b142c)
- **移动时的 GI 亮点。** 用户报告 FSR 原型在 GI 降噪之后仍会显出零星噪点和 fireflies，镜头运动时尤其干扰观感，这是原型一直未合入的原因之一。代码里有可选的 GI anti-firefly，混合强度随重投影历史长度从 1 到 4 帧逐渐上升。短历史像素是待核对的线索；现有记录还没有把这些亮点逐帧归到具体阶段。[原型 anti-firefly](https://github.com/Luna5ama/Alpha-Piscium/blob/b72a8b5fddeb2e52ea20a2c36eb16e0c30b0661b/shaders/pass/composite/GIDenoiserAntiFireFly.comp.glsl#L89-L125)

| 参考 | 能从原始资料确认的策略 | 在 AP 中怎样检验 |
| --- | --- | --- |
| Unreal TSR | 分析多帧亮度振荡，允许稳定的细密几何保留更多历史。像素动画能触发相同信号，Epic 提供材质标记，让这类像素跳过抗闪烁判断。[Epic TSR](https://dev.epicgames.com/documentation/en-us/unreal-engine/temporal-super-resolution-in-unreal-engine) | 缓慢横移镜头拍远处栅栏，另拍滚动纹理。用几何重投影与深度先排除遮挡变化，对可信表面再试短期振荡判断。记录细线断续与动画拖尾，检验这两个相反目标。 |
| AMD FSR 3.1 | Reactive mask 可降低历史权重，transparency/composition mask 较温和，影响历史锁定和颜色裁剪。AP 原型将两个值都从实体反应标记与 overlay 覆盖生成，重投影无效时只额外提高前者。[AMD 输入说明](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-upscaler/) · [AP 遮罩写入](https://github.com/Luna5ama/Alpha-Piscium/blob/b72a8b5fddeb2e52ea20a2c36eb16e0c30b0661b/shaders/pass/composite/FSR3MotionVectors.comp.glsl#L39-L50) | 把双通道对照限定在有相应信号的实体、粒子或 overlay。粗糙玻璃沿用已移除专用覆盖的接线，核对其历史累积。 |
| Intel XeSS-SR | 支持低分辨率运动向量加深度，也接受已膨胀的输出分辨率向量。可用连续权重的 responsive mask 提示缺少可靠运动的像素，Intel 建议同时检查历史减少引起的闪烁。[Intel 开发指南](https://www.intel.com/content/www/us/en/developer/articles/technical/xess-sr-developer-guide.html) | 固定相机，让实体横穿静态墙。分开观察前景向量是否串入背景、实体自身是否缺运动数据。前者试深度引导的前景速度选择，后者先用连续权重降低历史；速度膨胀无法补出上一帧骨骼姿态。 |
| NVIDIA DLSS Super Resolution | NVIDIA 公开说明 DLSS 4.5 在场景线性空间训练和推理；Streamline 的 SR 接口使用颜色、深度、运动和曝光信息。[NVIDIA 技术说明](https://www.nvidia.com/en-us/geforce/news/dlss-4-5-dynamic-multi-frame-gen-6x-2nd-gen-transformer-super-res/) · [Streamline 接口](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS.md) | AP 原型把历史存为场景线性颜色，累积时临时进入可逆 AgX 域。固定采样、历史权重与锐化，对比在该域混合，以及在线性辐亮度域混合、用压缩亮度估计可信度。检查细小自发光和反射高光的能量、颜色与闪烁。这个实验用普通 shader 即可检验。 |
| Playdead INSIDE TAA | 公开 shader 用当前邻域约束重投影历史，并让亮度差调节历史权重。[Playdead 实现](https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader) | AP 原生 TAA 已有邻域与方差裁剪。分别对照 Playdead 的简化路径和 AP 随运动变化的裁剪，检查静止的颜色 AABB 能保住多少细节。[AP 原生 TAA](../../shaders/pass/composite/TAAResolve.comp.glsl) |
| Qualcomm Snapdragon GSR 2 | 提供二阶段和三阶段的时域超分版本；三阶段版本把激活判定独立出来。利用深度不一致、亮度历史与局部颜色范围控制重投影。[SGSR 2 开发说明](https://github.com/SnapdragonGameStudios/snapdragon-gsr/blob/main/sgsr/v2/README.md) | 作为较轻的累积器对照。保持相同的 GI、玻璃输入与静态颜色 AABB 基线，比较额外一段历史分析能换来多少运动稳定性，再看 GPU 成本。 |
| SMAA T2x | 边缘形态判断与时间方向的子像素采样相结合，公开实现可检查空间边缘恢复的贡献。[SMAA 作者项目](https://www.iryoku.com/smaa/) | 用作原生分辨率的边缘对照。先看远处细线与像素纹理在镜头停稳后的对比度；GI 时间降噪由单独的实验评价。 |
| Adaptive Temporal Antialiasing（ATAA） | 为普通 TAA 难以重建的区域选择性追加光线采样，提高那些位置的观测质量。[ATAA 论文](https://research.nvidia.com/publication/2018-08_Adaptive-Temporal-Antialiasing) | 若短历史区的 GI 输入本身仍有稀疏离群值，可试在少量高风险像素增加 GI 采样。将新增的追踪毫秒计入总成本；原论文的硬件光追路径与 AP 的输入不同。 |
| 联合降噪与超采样（HPG 2022） | 一套低分辨率特征提取器服务多个滤波阶段，联合输出降噪后的高分辨率结果。[Thomas 等人的论文](https://momentsingraphics.de/HPG2022.html) | 待 GI 管线和降噪器迭代后，才评估把方差或有效历史长度传给重建器能带来多少改善。首轮空间与历史实验使用现有的降噪后场景颜色。 |
| 自适应时域滤波 A-SVGF | 对稀疏位置用相同随机序列重算着色，估计真实变化的时间梯度，帮助区分光照变化与采样噪声。[Schied 等人的论文](https://momentsingraphics.de/HPG2018.html) | 留作降噪器迭代后的诊断参考。届时可用固定与变化种子的配对回放，观察疑似 firefly 是否跟着采样跳、真实发光点能否在运动中持续。使用时间梯度需要可重放的着色样本，额外采样计入预算。 |

**建议的实验顺序。**

1. 用同一段静止、缓慢横移、停下的轨迹，保存 GI 降噪前后、最终时域重建输入／输出及锐化后的帧。用高采样的同一场景时刻作为参照，在重投影对齐后记录虚假亮点的数量、亮度偏差与寿命。跟踪真实小灯或镜面高光作对照，记录其峰值与覆盖率。先判断噪点在哪个阶段出现、在哪个阶段变得显眼。
2. 保留粗糙玻璃现有历史路径。在静止玻璃测帧间亮度波动，横移时看玻璃与背景的拖尾；同时以静止的栅栏、远处细线和高反差纹理对照现有颜色 AABB，记录边缘对比、线条覆盖率和停稳后的收敛。抑制移动亮点的候选须同时通过这两组对照。
3. 若亮点主要在重建或锐化时被放大，分别检查当前样本的可信度、历史裁剪与线性 HDR 混合。若现有降噪输出上的残噪和真实小高光难以兼顾，TAA/SR 可以后移，先完成 GI 管线与降噪器，再用相同镜头复查。届时再考虑上游方差或稀疏着色时间梯度。实体和 overlay 的 Z/W 遮罩语义、TSR 式细线保护各自独立对照，组合后再评估累计成本和副作用。

**比较口径。** 对每种算法重放同一段帧序列，固定几何动画、相机、抖动、曝光、DRT 与锐化。原生 1080p 和 720p 输入到 1080p 分别比较；先看重建输出，再看同设置最终画面。用相同场景时刻的高采样参考核对栅栏的可见覆盖率和高光亮度，再逐帧记录遮挡解除后的残影消退帧数、静态区域亮度波动与运动中的细节。另列整条重建链的 GPU 毫秒、图集显存和物理 image 数。在 Iris 1.10.4 上核对原生、超分两种模式及法线贴图开关。以上为实验设计，方案优劣待同段实测。

#### 历史重投影的边缘清晰度

原生 TAA 和超分都要减少有效历史连续重投影时的累积模糊。即使运动向量完全正确，连续半像素平移也会反复插值已经平均过的颜色，使普通物体轮廓、同一平面上的贴图或颜色分界逐帧变软。历史读取应能在 pixel lock 关闭时保住这些边缘的对比度。遮挡解除、运动或着色已经失配的部分仍按历史失效处理。

输出尺寸历史的一种候选表示是同时保存平均颜色与**信号相对像素内位置的一阶矩**。位置与信号按有效面积权重中心化，使均匀色块在采样位置偏向一侧时仍有零位置矩。读取历史时，把目标像素足迹映回旧历史，用均值与矩估计旧像素内部的颜色变化，按相交部分重建目标像素的均值和局部矩。这一步作用于历史采样。已经损失的细节优先用同一表面的可信新观测恢复，当前帧的空间滤核和写入权重单独比较。[Amortized Supersampling](https://hhoppe.com/proj/supersample/)

二维亮度位置矩有两个分量。同亮异色边需要色度信息，逐 RGB 通道存二维矩则有六个分量。最终表示要用普通物体斜边、同深度纹理边、同亮异色边、匀色块及 GI 残噪区检验。固定重投影轨迹，在关闭 lock、关闭锐化的条件下先单测历史读取，再恢复相同的新帧输入。对照输出尺寸历史和宽高各为输出 200% 的历史，记录边缘过渡宽度、过冲、停稳后的恢复，以及实际历史字节数、读写量和 GPU 毫秒。[TSR 的历史分辨率](https://dev.epicgames.com/documentation/en-us/unreal-engine/temporal-super-resolution-in-unreal-engine?application_version=5.7)

**一维黑白分界算例。** 128 格周期数组前半为 0，后半为 1，初始边界对齐。每步向右平移半格，只重采上一帧历史。基线对均值作线性插值。矩的对照在每格用 `A + s·u` 重建亮度，`u` 从 −½ 到 ½，`A` 为格内均值，`s = 12 E[u·Y]`。将斜率幅度收至两个端点均在本格及左右邻格的均值范围内，再对与新格相交的两段精确积分，更新均值和相对新格中心的一阶矩。平移 16 步后，参考分界重新对齐，两侧应为 0 和 1。基线为 0.402 和 0.598，矩的对照为 0.230 和 0.770。两者均保持总亮度。这个标量算例核对的是重采样损失，AP 的二维画质和 GPU 成本待同镜头实测。

#### 细线的默认外观

原生 TAA 和超分默认让远处电线、铁栏的线宽及其相对背景的对比度贴近高采样参考。对照使用相同的场景时刻、视角和输出分辨率。细线可以随像素覆盖率降低而自然变淡，同时应抑制相机静止或缓慢横移时由采样抖动引起的闪烁。

#### Pixel lock 与覆盖率

原生 TAA 和超分共用这项 2.1 设计。Pixel lock 记录重投影后仍是同一条细线的置信度，允许已确认的细节跨过漏采帧。覆盖率 `c` 表示前景占当前**输出像素**的面积比例，另记有效观测权重，以区分尚无观测与已观测为零。普通静止表面沿用从当前原始邻域取得的颜色 AABB。

在渲染分辨率上检测连续的细线及其方向。深度边、材质自发光或另一抖动相位在同一轮廓上的命中可佐证候选，高置信度时才建锁。单帧孤立的 GI 亮点走普通 TAA。横移时锁随轮廓重投影，覆盖统计只继承与新输出像素足迹仍重叠的观测。若无法确定重叠部分，就降低旧统计的权重，从当前轮廓重新估计。按真实抖动位置及其代表的面积更新覆盖率。

| 本帧观测 | 更新 |
| --- | --- |
| 足迹内采到同一根杆 | 记前景命中，用有效的杆色更新前景估计。 |
| 足迹内确实从杆旁采到墙 | 记背景命中，计入覆盖率的分母，并更新墙色。一次穿过缝隙仍可保留锁。只统计见到杆的帧会高估线宽和亮度。 |
| 该相位没有采到此足迹，或无法判定样本归属 | 保留上次的覆盖估计，有效证据随时间变旧，锁逐渐衰减。 |
| 深度和运动证实杆已移走或被遮住 | 当帧清除该位置的旧锁和覆盖历史，从新可见表面重建。 |

首版保留一份已合成的场景线性 HDR 颜色历史。当前邻域能分清杆色 `F` 和墙色 `B` 时，按 `C = B + c × (F − B)` 合成一次。杆为 0.8、墙为 0.2、覆盖四分之一时，结果为 0.35。漏采帧可暂用这份已混好的 0.35，直接作为历史颜色参与融合。灯条熄灭时可以保留几何覆盖，一旦采到可信的新杆色就刷新颜色。

沿同一段静止、慢横移、停稳的铁栏镜头，对照同机位高采样参考，逐帧检查线宽、相对墙面的亮度以及相邻输出像素间的亮度交接。用灯条熄灭、门柱遮住铁栏、墙上单帧 GI 亮点分别检查旧亮度、旧轮廓与离群点的残留帧数。建锁阈值与衰减速度由这组画面校准。

**已确认细线的覆盖与裁剪候选**

- **覆盖的位置矩。** 对能分清前景与背景的细线，可用命中标记的空间矩描述像素内的颜色分布。把前景命中位置的一阶矩除以覆盖率得到前景质心，结合方向重建条带，估算横移后分给相邻像素的面积。两根线可能与一根线有相同面积和质心，交叉或分叉处须检验歧义。[面积与质心用于界面重建的研究](https://doi.org/10.1016/j.jcp.2007.12.029)
- **按覆盖率约束混色。** 当两侧颜色可信时，用覆盖率的可信区间求出允许的前后景混色范围，检查旧颜色是否还能由当前轮廓解释。单侧漏采和照明确已改变分别走上面的历史保留与颜色刷新规则。

#### 当前帧的空间重建

FSR 3.1 原型把当前帧的抖动位置计入采样距离，用 Lanczos 权重估计每个输出像素的颜色，再与重投影历史合成。2.1 按深度和轮廓选取同侧邻居，由空间核从渲染分辨率输入估计当前颜色。历史更新根据实际抖动位置与观测可信度决定这次写入多少。沿边借来的多个邻居能帮助估色，共享同一帧的时间相位。当前帧采用何种滤核、取几个输入样本，可在固定历史重投影、裁剪和锐化的条件下独立比较。历史权重设为零可观察原生 TAA 和超分的单帧结果。[原型的坐标与滤波](https://github.com/Luna5ama/Alpha-Piscium/blob/b72a8b5fddeb2e52ea20a2c36eb16e0c30b0661b/shaders/techniques/ffx/fsr3upscaler/ffx_fsr3upscaler_upsample.glsl#L122-L212)

- **沿轮廓取样。** FSR 1 的 EASU 根据邻域梯度确定滤波方向。原版接收已抗锯齿的画面，AMD 建议在感知域运行。AP 的候选用压缩亮度估计方向，按本帧抖动定位原始样本，以同一组权重重建场景线性 HDR 颜色。[AMD FSR 1](https://gpuopen.com/fidelityfx-superresolution/)
- **选择值得修的边。** SMAA 1x 的斜线、拐角识别和 CMAA2 的保守形状检测都值得单独试。用连续的阶梯轮廓支持定向滤波，方向不明确时退回普通采样。用 FXAA 比较廉价边缘平滑能做到什么，同时保留 AP 当前 TAA 后接 FXAA 的成片对照。[SMAA](https://www.iryoku.com/smaa/) · [Intel CMAA2](https://www.intel.com/content/www/us/en/developer/articles/technical/conservative-morphological-anti-aliasing-20.html) · [NVIDIA FXAA](https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf) · [AP 的 FXAA](../../shaders/pass/composite/FXAA.comp.glsl)

**本帧估色的离线对照。** 从 2.1 设计修订 `99fde776` 出发，28 种灰度图形各取 16 个真实抖动相位，对照原生、1.5× 和 2×，输出为 72×60。低分辨率输入是点采样，参考值是连续图形在每个输出像素内的面积平均。Windows D3D11 WARP 执行原版 SMAA 1x High 与 CMAA2 High。EASU 对照按 AMD 的 12 点公式移植，在压缩亮度上判断方向，按场景线性值估色。下表为关闭历史和锐化时，1.5× 下轮廓邻域按图形等权平均的绝对亮度误差，单位为图形自身的亮暗幅度。

| 图形 | EASU | 只换 3×3 等权结构张量方向 | SMAA 后接 EASU | CMAA2 后接 EASU | 同 12 点拟合单边覆盖，未匹配处用 EASU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 六条斜边 | 0.0545 | 0.0551 | 0.0426 | 0.0440 | 0.0338 |
| 四根斜细杆 | 0.0757 | 0.0757 | 0.0655 | 0.0707 | 0.0743 |
| 三个 T 形接头 | 0.0969 | 0.0960 | 0.1031 | 0.0991 | 0.0964 |

六条斜边的平均法线角误差用 3×3 张量可从 10.6° 降到 4.6°，保留 EASU 原有滤核强度时，颜色误差几乎相同。把单边拟合扩大到 5×5，直接积分输出像素的覆盖，六条斜边误差降至 0.0279，纯轴边却从 0.0845 升到 0.0963。要求两色区域能用一条直线分隔，而且邻域里横向和纵向都有颜色阶跃，同批斜边仍为 0.0279，轴边回到 0.0845。

固定上述条件，在另外 12 个角度、8 个新相位上，直边误差均有下降，平均从 0.06193 降至 0.03745。半径 3.7 和 7.2 输出像素的圆弧却分别从 0.0663、0.0569 升至 0.0708、0.0633。局部点样本能被一条直线分隔，仍可能对应弯曲的真实轮廓。给各方法相同的正确运动向量、双线性历史重投影及 20% 本帧权重，34° 直边在横移期间从 0.0867 降至 0.0761，停稳最后四帧从 0.0411 降至 0.0270。这个固定权重用于隔离空间方法的影响，实际写入量仍取决于历史与当前观测。

细杆有两条边，还要核对整根杆相对背景的亮度贡献。以面积参考为 100%，四根斜杆上 EASU 平均为 107%，SMAA 后接 EASU 为 87%，CMAA2 后接 EASU 为 79%。两种串接都可能在像素误差下降时把整根杆画淡。本机主检出的 `.tmp/taa-sr-design/spatial-estimator-20261125a/` 保存 `report.md`、`results.json`、`holdout.json` 和同一相位的 `same-phase.png`，运行 `run-all.ps1` 可复现。这些对照测的是合成图像，AP 游戏画面与 GPU 毫秒仍待同镜头验证。

历史是否重投影到同一表面、当前邻域能否辨出轮廓，分别判断。门柱移开露出的铁栏立即丢弃门柱历史，由当前样本重建。已有可靠历史的静止铁栏继续累积各抖动相位的样本，颜色 AABB 从空间平滑前的邻域取值，防止逐帧收窄颜色范围。粗糙玻璃仍可累积有效历史。单帧孤立高亮提供的方向证据不足，以运动中的真实小灯与 GI 亮点检验能否避免把两者拉成长边。锐化在历史写入之后进行。

先从已有捕获中找相机、抖动和输入一致的片段，建立原生 TAA 与 FSR 原型的基线。空间候选沿同一条横移轨迹露出细栅栏，再停住。先强制历史权重为零，在锐化前比较原型和候选的首帧轮廓、各抖动相位的断线与闪烁。然后恢复相同的历史策略，逐帧检查遮挡解除、停稳后的纹理对比度、玻璃噪声和 GI 亮点。沿用上面的两档分辨率与整链 GPU 计时，再决定保留哪些空间策略。

#### 当前观测与历史置信度

历史更新分别估计像素平均亮度与跨边对比度的可信度。在本帧真实抖动落点，用对应表面的原始观测减去历史预测，取裁剪、混合之前的残差。若能分清一条边的两侧，令亮侧的面积比例为 `c`，两侧残差为 `r亮`、`r暗`。固定轮廓与覆盖时，观测给出的像素均值残差为 `c·r亮 + (1−c)·r暗`，跨边对比度残差为 `r亮 − r暗`，各按已有证据决定写入量。只命中一侧时可以更新那一侧的估计，另一侧仍缺证据。覆盖率本身不可靠时，保留这项不确定性。用清晰和已变软的两份历史预测同一批新样本，分别核对亮度、对比度与漏采侧的恢复。

将重投影定位误差计入历史预测。在同一表面局部平滑的区域，亮度梯度在位置误差方向上的分量决定错位会引入多大色差。沿直板缝小幅错位对预测影响较小，横穿板缝则更明显。深度和可见性已证实表面更换时当帧丢弃旧历史。材质能确认灯已熄灭时可立即刷新亮度，形状仍可有效。

**持续变化的检测候选。** 在明暗两侧各取得一次观测后，可分别对两侧残差的和、差做带限幅的双向累积和检测（CUSUM），并与两次连续同向超阈值的简化办法对照。以 `99fde776` 设计为起点的离线算例固定历史预测、每帧交替采一侧，每段 24 帧。用 5 万段独立噪声将每种方法的整段误报率校准至约 1%，另用 2.5 万段测试。每段插入一次 10 倍噪声标准差的亮点时，两次同向判据和 CUSUM 的整段误报率分别为 6.4% 与 3.9%。两侧持续变暗 0.75 倍标准差时，在 24 帧内首先正确检出亮度下降的比例分别为 16.9% 与 69.6%。改用一阶相关系数 0.75 的噪声，沿用原阈值，静态输入的 CUSUM 整段误报率升至 35.9%。这些数值描述检测器，在 AP 上还须测实际残差的时间相关、恢复延迟与拖尾。本机从主检出的 `.tmp/taa-sr-design/20261125-history-confidence/` 运行 `python probe.py` 可复现 `results.json`。[CUSUM 方法](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm)

## Bloom

`SETTING_BLOOM_SHAPE` offers Standard (0, default), Diamond Four (1), Horizontal Streak (2), Cross Four (3),
Camera Six (4), Camera Ten (5), Cinema Star (6), Prismatic Streak (7), 67 (8), and Trollface (9).
Standard is an eight-scale, long-tailed Gaussian approximation with a concentrated core. Each scale contributes
1/8 of the normalized filter; it is not a single Gaussian. The fixed eight levels preserve the broad halo that
was lost in the rejected four-level prototype.

The filters use offline-fitted offsets and nonnegative weights in
[`BloomTaps.glsl`](../../shaders/techniques/BloomTaps.glsl). Fitting accounts for decimation phases and bilinear
sampling, following the [DPF approach](https://huakaifugui2.github.io/CZ-DPF-Paper/). Runtime uses fixed RGB filters
and a fixed pyramid depth. The four-scale fitting proxy calibrates reusable taps; it does not set the runtime depth.

| Order | Pass/stage | Purpose |
| --- | --- | --- |
| 1 | [`Bloom`](../../shaders/techniques/Bloom.comp.glsl), downsample levels 1–8 | Five bilinear reads per output texel. The first level reads the decoded linear main image and applies Bloom-only highlight compression. |
| 2 | [`BloomShape`](../../shaders/pass/composite/BloomShape.comp.glsl), optional | Initializes the selected tap table in workgroup shared memory, then samples raw level 1 in a bounded loop and writes a separate half-resolution image. Standard and Bloom Off disable this pass. |
| 3 | [`Bloom`](../../shaders/techniques/Bloom.comp.glsl), reconstruct levels 8–2 | Eight bilinear reads per output texel. The deepest contribution and each finer downsample receive weight 1/8; the invocation reads the finer texel that it then overwrites. |
| 4 | [`PostComposite`](../../shaders/pass/composite/PostComposite.comp.glsl) | Fuses the last upsample, adds the optional shape with centered bilinear reconstruction, and applies Bloom intensity and underwater boost. |

[`programs.main.kts`](../../scripts/programs.main.kts) owns this order and its enable conditions.
[`Bloom.glsl`](../../shaders/techniques/Bloom.glsl) owns common addressing and sampling. Downsample and shape reads
zero-extend their bilinear footprint at screen edges while keeping atlas reads inside the input tile. Upsampling
clamps to the valid input mip. This prevents a long streak from repeatedly copying a bright edge texel.

- `transient_bloom`: frame-local RGBA16F atlas with ceil-sized padded mip tiles; reconstructed in place from coarse
  to fine.
- `transient_bloomShape`: frame-local RGBA16F logical image; only its half-resolution region is written and read.
  It lives from the shape pass through post composition. The allocation in
  [`shadesmith.json`](../../shaders/shadesmith.json) reuses a slot after FXAA/RCAS have finished with it, without
  increasing the current physical atlas dimensions. It remains a separate logical resource from the pyramid.

Each shape adds energy over the full Standard core and tail. Camera and Prismatic shapes spread each input channel
with its own weights. The two novelty kernels align their visual bounds to the light and are most legible around
small emitters. Larger emitting surfaces broaden the shapes and can fill fine gaps. Parameters and custom
PNG/function input are maintained by
[`fit-bloom-shape.py`](../../scripts/fit-bloom-shape.py); see [kernel fitting](../development/offline-tools.md#bloom-kernel-fitting).

Highlight compression belongs to the first Bloom downsample, after AA decoding. RGB and luma modes retain their
existing curves. AA uses fixed log encoding with an exact zero; decoded writes in
[`RCAS`](../../shaders/pass/composite/RCAS.comp.glsl) and the TAA-off path of
[`TAAResolve`](../../shaders/pass/composite/TAAResolve.comp.glsl) clamp to the RGBA16F limit of 65504.

### Bloom validation and limits

Shader snapshot `0ed64baf9a70…` rendered all ten modes and Bloom Off at 1280×720 in Small lamps night (EV 9.45),
outdoor point lights (EV 11.07), and Craft close lights (EV 7.34). Camera and world time were fixed within each
scene, with 192 warmup frames per mode. Camera Six produced finite, nonzero shape output with TAA disabled.
Bloom Off suppressed the first downsample and shape passes with Trollface selected.

On RTX 3080 Ti in Small lamps night, two rounds bracketed the forward and reverse shape orders with Standard.
Each window used 96 warmup frames and 64 sampled frames. GPU times are in milliseconds.

| Mode | Shape pass p50 | Bloom + complete PostComposite p50 / p95 |
| --- | ---: | ---: |
| Standard | — | 0.191 / 0.201 |
| Diamond Four | 0.254 | 0.455 / 0.471 |
| Horizontal Streak | 0.094 | 0.289 / 0.296 |
| Cross Four | 0.193 | 0.386 / 0.396 |
| Camera Six | 0.357 | 0.553 / 1.114 |
| Camera Ten | 0.399 | 0.593 / 1.234 |
| Cinema Star | 0.263 | 0.460 / 0.831 |
| Prismatic Streak | 0.179 | 0.375 / 0.397 |
| 67 | 0.199 | 0.395 / 0.411 |
| Trollface | 0.311 | 0.507 / 0.992 |

The combined column sums the aligned program samples per frame before calculating quantiles. The unchanged
volumetric-composite reference pass had a 0.049 ms median in every mode.

Independent CPU impulse models support a near-radial standard kernel: at radii 12, 24, 48 and 96 pixels, the
eight-phase mean fourfold angular component was about 1.0–1.9% of ring brightness, versus 2.6–3.2% for the previous
COD filter. Dyadic sampling still has pixel-grid and phase error; this is an approximation, not exact rotational
symmetry. Odd viewport sizes, boundary energy and phase centroids have CPU coverage; odd-size runtime, underwater
rendering and other GPUs remain unmeasured. Shape fitting error describes the added raw-level-1 branch, not the
complete Bloom result.

## Camera exposure

[`TAAPrepare`](../../shaders/pass/composite/TAAPrepare.comp.glsl) accumulates a 256-bin weighted histogram of post-DOF scene luminance over 40 stops starting at −24 EV. Distance, emissive and center weights feed workgroup-local bins and then the frame SSBO. [`ExposureGather`](../../shaders/pass/composite/ExposureGather.comp.glsl) trims each tail by cumulative weight, retains partial boundary bins and computes the weighted mean log luminance. The target gray rises from the night setting to the daylight setting. The reducer clamps the resulting EV, updates temporal adaptation, then leaves the histogram empty for the next frame.

Default post-exposure gray is 8/255 (0.031) for the darkest scenes and 40/255 (0.157) for daylight, in scene-linear units before Bloom and the DRT. The camera blends between these targets using metered log luminance from −13 to −2 EV and clamps its gain to −6.5 through +10 EV. The bright target sets the level for the retained log average. Individual sunlit surfaces can be much brighter.

Reprojection reads unexposed scene-linear history from the previous frame. FXAA stores the new history before spatial filtering. RCAS applies the camera multiplier after reconstruction. The TAA-disabled path applies the same multiplier in TAAResolve. Bloom and all four DRTs therefore see the same camera exposure. The histogram debug view shows the retained frame's bin weights, target EV and camera EV.

## Display transform

[`PostComposite`](../../shaders/pass/composite/PostComposite.comp.glsl) adds Bloom and calls [`DRT`](../../shaders/techniques/displaytransform/DRT.glsl). The default is Skibidi v41 with neutral Look and Display P3 target primaries. SDR grading retains the primary calibration and HSL mixer, converts to the chosen output primaries and transfer function, then applies the common Look to encoded RGB. Final output dithers SDR to eight bits.

| DRT | Input and reference | Target gamut |
| --- | --- | --- |
| Skibidi v41 | Linear AP0. The source model and SDR/SDR-style HDR curves come from drt-bench `280aa5e31fdfb412bea24f9746b9417fedfd67d5`. | Native radial allocation in the chosen D65 RGB space. |
| Blender AgX | Blender 5.0.0 `a37564c4df7a9b604809e5192918400e46a0205f` OCIO transforms from ACES2065-1 to sRGB AgX or Rec.2100-HLG AgX HDR 1000 nits. The 57³ LUTs use tetrahedral interpolation. | View-defined primaries. |
| OpenDRT | Standard v1.1.0 from the same drt-bench revision, via XYZ. | Uses the selected display basis for its output rendering. |
| Psychov30 | RenoDX `6d563066956190c8b9a2c7a907398bfac3629a3c` test30, via linear BT.709 with white-point adaptation. | Uses the existing 709 or 2020 operator. Other targets use the native SDR or HDR default. |

For Skibidi, preset D65 and user-entered primary coordinates feed the same gamut matrix. An invalid triangle falls back to Display P3. Rec.709 uses the pinned v41 scalar fields. For other targets [`SkibidiGamut`](../../shaders/pass/setup/SkibidiGamut.comp.glsl) generates the per-hue boundary seeds, red cusp and EO12 fit when Iris runs setup. Its coefficients and weighted bins occupy 670,728 bytes after the 16 KiB global block in SSBO 0. The persistent 145 × 81 tile begins at (0, 256) in the fixed R32F image. Expanding that image from 256 × 264 to 256 × 345 costs 82,944 bytes. One 256-lane workgroup carries the four setup phases. Each workgroup barrier follows a group memory barrier. The fitter reads the newly stored boundary fields through coherent image loads with bilinear interpolation. The steady-state display pass uses the filtered sampler. This allows the existing frame passes to use concurrent compute.

When HDR Mod supplies both presence and enable macros, DRTs read game paper white and peak in nits. The Look uses signed sRGB relative to game paper white, including pixels above white. At handoff AP scales by game paper white divided by UI white, then applies signed sRGB encoding in BT.709 primaries. The mod consumes floating-point output and handles monitor primaries and transfer. The SDR path applies whenever either macro is absent. [HDR Mod shaderpack interface](https://github.com/rrtt217/Minecraft-HDR-Mod/wiki/For-Shaderpack-Devs)

### 2.1 validation record

The fixed-input comparison uses the three ACES sample frames 0017, 0061 and 0067 plus gray, primary and signed probes, at −2, 0 and +2 EV in AP0, neutral Look, sRGB output and Rec.709 target. Across these exposures the largest 8-bit channel difference was one code for Skibidi and Blender AgX, and zero for OpenDRT and Psychov30. Blender uses the user's chosen 5.0 view. Rebuild its textures and shaders with OpenColorIO 2.6.0 and [the AgX generator](../../scripts/generate-agx.py). The Blender color configuration credits Troy Sobotka, Zijun Eary Zhou, Mark Faderbauer and Sakari Kapanen. [Pinned Blender configuration](https://github.com/blender/blender/blob/a37564c4df7a9b604809e5192918400e46a0205f/release/datafiles/colormanagement/config.ocio)

In an isolated GL dispatch, preset P3 and manual (680,320), (265,690), (150,60) in thousandths produced identical tables and SDR/HDR sample values on llvmpipe. The single-workgroup P3 bake differed from the four-dispatch bake by at most 0.00000346 in the EO12 table. All 54 packaged GLSL variants compiled, including manual exposure, TAA off, custom target, and absent, disabled and enabled HDR interfaces. The 65 × 37 weighted-histogram fixture yielded +7.355 EV for the dark uniform scene and the same scene with a small bright emitter, −0.456 EV for bright snow, and +6.668 EV for a dark cave with a bright exit. These values describe the synthetic fixture.

Minecraft 1.21.11, Iris 1.10.6-vibris.3, HDR Mod 2.3.4 and RTX 3080 Ti ran the 1280 × 720 night arcade scene with manual EV 0 and Skibidi targeting Display P3. Both captures used the four-dispatch implementation in candidate `f4213de6508a678531ba8fa0a7da6d30c684d533` with a diagnostic strip inserted before the production DRT. The recovered game paper white, peak and UI white were 203, 1000 and 284 nits respectively. Neutral gray rose monotonically to 998.7 nits. The closest sampled code to paper white decoded to 203.54 nits. In 162 colored highlight samples above paper white the median largest-channel change from neutral to Punchy was 708.3 nits. These are shader handoff values reconstructed from RGBA16F. The SSBO readback verified manual EV 0 in both jobs and a 687,112-byte allocation at binding 0. Against software GL on the same chart and uniform values, the largest differences in signed encoded RGB were 0.000981 for neutral and 0.001961 for Punchy. For every DRT, the installed-but-disabled and missing-mod shader variants gave identical software output for both Looks. The generated pack has the same 16 physical image declarations as its 2.1 base.

On `b384fe2ae24a39329034a8d9c71d6215b32ea819`, Iris dispatched the single-workgroup gamut bake. Its standalone GPU time on the RTX 3080 Ti was 6.66 ms for Display P3 on setup. With the same chart, EV 0, neutral Look and 203/1000 nit scene settings, changing only the mod's UI white from 284 to 160 nits gave 998.125 and 997.500 nits at the brightest gray. The same gray patch near paper white measured 205.469 nits in both window captures. Across 480 gray and colored samples the median largest-channel change was 0.039 nits and the maximum was 1.875 nits. These values come from FP16 Windows Graphics Capture of the dedicated game window on a 10-bit PQ/Rec.2020 DXGI output. They measure the composited signal. The capture and patch coordinate matching are recorded in `compositor-result.json` and `compare-compositor.py`.

The same game also displayed the fixed EV 0 diagnostic chart on the flatroom preset with HDR Mod missing (140 loaded mods) and installed but disabled (142). Both composited windows reached 283.906 nits at the brightest gray. Against the 284-nit SDR reference the 95th-percentile absolute channel errors across the chart were 1.75 and 2.41 nits respectively. The largest difference between the two window captures was 4.22 nits. Both window captures use the HDR measurement's patch coordinates. Vibris timed out waiting for terrain stability on the missing-mod texture dump. With the mod installed and disabled, a settled-frame RGBA16F readback on `c651b6c0` agreed with the software SDR chart to 0.000489 in encoded RGB at manual EV 0. See `sdr-window-results.json`, the retained FP16 windows and both loader records.

In the same Iris version, Skibidi, Blender AgX, OpenDRT, Psychov30 and Skibidi with Bloom intensity raised by four stops all returned the identical 32-bit target EV (+2.6045022) and metered log luminance (−4.1524277). This test fixed the linear input and weights at the histogram entrance. The game loaded each DRT and SSBO 0 recorded the result. In separate unpatched 720p scenes, the torch-free deep wall in Dump Taiga adapted to +6.60 EV (target +6.67), the daylight cave exit to +0.30 EV and the small-lamp gallery to −2.03 EV. The dark wall's central 10th/50th/90th percentiles in encoded SDR luma were 0.210/0.314/0.434. Its surface pattern remained visible.

The default low/high trims are 3%/5% by weight. In the cave exit the weighted mean log luminance was −1.54 without trimming, −1.71 at 3%/5%, and −1.84 at 5%/10%. The same choices gave +0.73, +0.63 and +0.48 in the small-lamp gallery. On the deep wall, 3%/5% changed the untrimmed result by 0.024 stop. In the initial 96/255 daylight calibration, daytime snow reached the −3 EV floor and 29.70% of the frame had encoded luma at or above 0.99. Testing a −5 EV floor allowed snow to settle at −4.18 EV, leaving 0.0054% at or above 0.99. Median sunlit snow was 0.959 and shaded snow measured 0.754. The comparison uses the same camera, sun time and snow patches. See `snow-contrast.json`, the `*-scene.json` histogram readbacks and the matching scene PNGs.

At 96/255 with a −5 EV floor (`c75a8dc61`), daylight surfaces in the 2.1 SDR scene gallery looked washed out even after the camera settled. At 40/255 with a −6.5 EV floor (`0c2a23af3`), the noon courtyard went from −2.07 to −3.34 EV. The area at or above 0.95 encoded luma fell from 15.58% to 0.58%. Median encoded luma was 0.53 previously, 0.35 after tuning, and 0.27 in the earlier default AgX rendering (`792ffe98a`). In the snow preset, sunlit and shaded patches measured 0.887 and 0.570, previously 0.957 and 0.759. The midnight courtyard moved from +7.06 to +6.62 EV.

The six matched 1280 × 720 SDR pairs use Minecraft 1.21.11, Iris 1.10.6-vibris.3, Skibidi, neutral Look, and automatic exposure. They cover day and night in the courtyard, snow, the cave mouth, an indoor mall and a mountain lake. Each pair has the same scene and preset hashes.

The main checkout retains Vela run `c23818a2-c6d0-4fac-99a9-53e6566c3e36` in `builds/exposure-target-20261123/exposure-review.zip` (SHA-256 `f8a19d862694e1c495898411588162f8d58ebf124c835fca7052228763927129`). The archive contains the photographs, effective settings and camera readbacks, four recorded histograms replayed through both shader revisions, and the earlier courtyard reference. Open `comparison.html` from the archive for the six pairs.

<!-- vela:evidence {"run":"c23818a2-c6d0-4fac-99a9-53e6566c3e36","path":"exposure-review.zip"} -->

### GPU timings

On the 720p Dump Taiga snow preset, eight runs compared the former default on `792ffe98a` with the 2.1 default on `c651b6c0`. Each direction included 80 warmup frames and 96 measured frames. Values below are the medians of the per-run medians in milliseconds. Matched PresentMon windows each contained 92 frames after a 40 ms guard at both ends.

| GPU measurement | Earlier default | 2.1 default |
| --- | ---: | ---: |
| TAA preparation, including 2.1 metering | 0.0123 | 0.0174 |
| Display transform and Bloom combine | 0.0369 | 0.0934 |
| Exposure mip or histogram reduction | 0.0174 | 0.0164 |
| Selected post-processing programs, summed per frame | 0.5210 | 0.5978 |
| Terrain solid sentinel | 0.1116 | 0.1111 |
| Whole-frame GPU time | 21.414 | 21.554 |
| Whole-frame GPU busy time | 17.844 | 17.950 |

A separate 16-run forward/reverse matrix used the same frozen shader snapshot at manual EV 0. The display program took 0.0942 ms for Skibidi, 0.0369 for Blender AgX, 0.0389 for OpenDRT and 0.0338 for Psychov30. The program includes the final Bloom combine. Raw case receipts, frozen snapshot identities, PresentMon samples and the reducer are in `native-profiles.zip`. The independent gamut dispatch on the RTX 3080 Ti measured neutral narrow-gamut linear SDR channels from −0.00000003 to 1.00023 after transport and conversion back to the target. Punchy expanded that range to −0.312 to 1.234. See `gpu-geometry.json` for HDR and the preset and custom P3 results.

The AgX choice image, input identifiers and baseline script are in project Vela run `16da9e19-35a7-47c4-a356-a2884c76c57f`. Run `d2481554-868d-4549-9ccc-efb659257cee` contains the numerical comparison, generated tables, histogram tests, game chart, RGBA16F/SSBO readbacks and game receipts. The fit provenance files are `ordered.py` (SHA-256 `c5c571797f8df7293b7617edfc79febaac41f8aa6d5bb51215ff0b3f76bf7268`) and `exposure_curves.py` (`1aa03bb96c213137b76a5785cf00547c57cd81532b58083342376d0ba21ee7ef`). Raw files are under `.vela/local/runs/<run UUID>/` in the main project checkout. The hashed game texture and buffer captures for the exposure tests are in `native-captures.zip`. The disabled-mod chart is in `sdr-verified-strip.npy`.

### Skibidi 色域 LUT 的重载成本

2.1 候选 `b384fe2ae` 在 RTX 3080 Ti、Minecraft 1.21.11、Iris `1.10.6-vibris.3` 上启用默认 Skibidi 和 Display P3。相同街机场景的两次整包 `shader.reload` 用了 21.31 s 和 17.03 s。独立 OpenGL 时间查询在程序编译链接后开始，P3 制表单次派发用 6.66 ms。旧四阶段实验由宿主显式同步，四次派发合计 0.134 ms。AP 开启并发计算，合并阶段的候选用一个 256 线程工作组同步表的生产与读取。重设工作组数量前，需解决阶段间图像和 SSBO 的可见性。原始查询在项目运行 `d2481554-868d-4549-9ccc-efb659257cee/nvidia-{single,fixed}.log`。

候选实验仍按显示器原色在 setup 中制表。192 个色相、每个色相 15 个饱和度各自取 79 个亮度样本，同一方向的样本共用 11 个插值系数。枚举这些方向仅访问 7,203 个网格节点中的 4,168 个，原型在格线附近多留 12 个节点。需要拟合的控制点由 79,233 减至 45,980，227,520 个训练样本全部保留。将四邻点的系数按方向各读一次，源码中的控制点读取由 10,010,880 次减至 126,720 次。P3、Rec.2020、手填 P3 和窄色域在 llvmpipe 上逐项对照，每个目标共 9,417 个有效 LUT 表值，最大绝对差为 `1.67e-6`。`builds/skibidi-setup-20261122/check_fit.py` 可重做软件对照，`software.json` 保存四组结果。

**2.1 保留现行 dense 制表。** 在 RTX 3080 Ti、NVIDIA 617.42、无头 OpenGL 4.6 上，按方向拟合的 rays 将默认 P3 的制表派发从 6.77 ms 缩短到 4.59 ms。两种程序在初始化或重设目标时各派发一次。首次观测的 P3 驱动链接却从 599 ms 增到 755 ms，单次派发节省的约 2 ms 无法抵消这次链接增加的 156 ms。

| 目标 | GLSL 编译，dense / rays（ms） | 首次观测的链接，dense / rays（ms） | 重开进程后的链接，dense / rays（ms） | GPU 派发，dense / rays（ms） |
| --- | ---: | ---: | ---: | ---: |
| Display P3 | 36.91 / 8.62 | 598.93 / 755.25 | 2.39 / 4.36 | 6.77 / 4.59 |
| Rec.2020 | 9.05 / 11.20 | 807.78 / 1070.44 | 3.36 / 2.99 | 7.00 / 4.52 |
| 手填 P3 | 10.21 / 8.69 | 866.97 / 1114.45 | 2.94 / 3.15 | 6.19 / 4.16 |
| 手填窄色域 | 8.79 / 8.79 | 918.85 / 1156.79 | 3.57 / 5.20 | 7.70 / 4.75 |

编译与链接分别计至驱动返回状态。首个 P3 编译包含该进程首次使用编译器的开销。表中派发为两个进程对冻结源码做正反序配对的 GPU 时间中位数，每种实现、每个目标各 18 次，另存了首次派发时间。两个额外进程先预热编译器，再只改图像 uniform 的显式位置，运行时仍绑定同一图像。交替先后顺序后，两个新位置下 rays 的首次链接均比 dense 慢，四个目标的差值为 121–301 ms。以上时间只覆盖 setup 程序。

每个目标读回 9,417 个有效 FP32 表项。对同一目标，dense 和 rays 的结果逐位相同，重复派发和显式位置对照也得到相同结果。另将预设 P3 与手填相同原色单独比较，两种实现在该卡上都出现了最大 `8.29e-6` 的表项差异。原始逐次计时、四目标 LUT、冻结输入及复核脚本在 Vela 运行 `465e2b57-b291-4cb8-a246-e073f2f3db30` 的 `skibidi-setup-ab.zip`。复核时可在解包目录执行 `python summarize.py <f2dbfadcc 检出目录>`。

<!-- vela:evidence {"run":"465e2b57-b291-4cb8-a246-e073f2f3db30","path":"skibidi-setup-ab.zip"} -->

2026-11-22 17:00（洛杉矶）关闭的专用实例运行云实验 `031e136d5` 加 `c66921dd` 补丁。冻结源码中的 `setup23` 清空体素数据，包内没有 Skibidi 程序。Vibris 重试同一快照后，单独记录到重载 64.30 s、应用场景 16.64 s。云分支早先在同卡回放另一快照 `7f787bf60`，五个程序的驱动链接合计 73.79 s。此次中断缺逐程序计时，具体调用仍待定位。源包和重试回执在云检出 `.vibris/jobs/100ff5af-0088-47b8-b646-64ef2fd0c924/`，现场摘录在主检出 `builds/skibidi-setup-20261122/incident.json`。

## Debug and special modes

Debug output in [`shaders/techniques/debug/`](../../shaders/techniques/debug/) can inspect TAA, PostFX, tone-mapping,
or final boundaries; [`SSTStepDebug`](../../shaders/pass/composite/SSTStepDebug.comp.glsl) is a conditional program.
Screenshot mode adjusts animation/temporal clamps, and video-render mode adjusts long-term temporal effects. They do not
define parallel production post-processing branches.

## Maintenance invariants

- Keep TAA history color space, pre-exposure, jitter, and motion/reprojection aligned.
- Keep the eight Bloom levels, layer weights, generated taps and atlas addressing consistent. Refit directional
  shapes if the raw level-1 carrier changes.
- Keep exposure production, adaptation and final display consumers synchronized.
- Validate display-transform changes numerically with color charts/gradients and across SDR, wide-gamut, and HDR
  configurations for clipping, NaN, and negative values.
- Test static detail, moving edges, disocclusion, bright emissives, dark adaptation, translucent edges, and UI/overlays.
