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
| 5  | [`GIReSTIRPairedSpatialReuse`](../../../shaders/pass/composite/GIReSTIRPairedSpatialReuse.comp.glsl) × 1–4                                                                                                                   | pairwise spatial reuse；每批最多覆盖 7 个基础样本；累积 specular BRDF-ratio resolve |
| 6  | [`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl)                                                                                                                         | 对选中样本做 shading，按 roughness 把 specular 混向 ratio resolve，并排队 neighbor visibility ray |
| 7  | [`GIReSTIRSpatialReuseTrace`](../../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl)                                                                                                                           | trace 压缩后的 visibility 队列；被遮挡的样本除 ratio specular 外贡献为零 |

四个 spatial-reuse pass 的 `PASS_INDEX` 为 0–3，`PASS_BASE_SAMPLE_INDEX` 为 0/7/14/21；它们从 SSBO 0 offset 48 indirect
dispatch。`history_restir_reservoirTemporal`、`history_restir_primary`、`history_restir_prevSample` 和
`history_restir_prevHitNormal` 保存上一帧输入，`transient_restir_reservoirTemporal`、`transient_restir_primary`、
`transient_restir_spatialInput` 和 `transient_restir_pairwiseMISMetadata` 连接本帧阶段。[
`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl) 会在执行本帧最后一轮读取时，
将当前 temporal reservoir 与 primary 数据复制到各自固定的 history tile。所有 tile 定义见 [
`shaders/shadesmith.json`](../../../shaders/shadesmith.json)。

Spatial shading 会立即写入临时的 diffuse/specular 结果。需要 voxel visibility 的 neighbor 选择会把 octahedral 编码的
`resultY` 方向、hit distance 以及 ratio 混合参数（ratio specular 与 ReSTIR 权重）存入
`transient_restir_pairwiseMISMetadata`；随后每个 16×16 tile 按 world-direction octant 与 Morton 位置排序，
把自己的 ray 作为一段连续区间追加到 SSBO 1，数量由 `global_restirVisibilityRayCount` 记录（在
[`UpdateGlobalData`](../../../shaders/pass/begin/UpdateGlobalData.comp.glsl) 中重置）。Trace pass 按屏幕大小的队列容量启动，
超出计数的线程直接退出，并且只把未通过 voxel visibility 测试的临时结果替换为零 diffuse 和仅含 ratio 部分的 specular。

Specular 为 `mix(ratioSpec, restirSpec, roughness)`，roughness 为中心像素的线性 GGX roughness。Ratio resolve 覆盖中心像素以及
所有 spatial 批次中同平面 paired 像素的 temporal 样本。每个样本以中心材质计算其自身帧的 specular 估计 `L·W·f_o`，并以样本自身的
target BRDF 为上限，权重为 `min(f_r / f_o, 1)`：`f_r` 是中心像素朝样本 hit 点的 specular BRDF，`f_o` 是同一 BRDF 在样本自身帧中的值。
它不含 visibility 和 Jacobian 项。Paired pass 把加权平均与权重和暂存在 `transient_ssgiSpecOut` 中；在 spatial shading 写入最终结果之前，
该 tile 不被其他 pass 使用。

## GI 降噪

ReSTIR shading 后依次执行：

| 顺序 | Pass                                                                                                                                                              | 作用                                                              |
|----|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| 1  | [`GIDenoiserAccum`](../../../shaders/pass/composite/GIDenoiserAccum.comp.glsl)                                                                                    | 时域 accumulation；更新 `history_gi1`…`history_gi5`                  |
| 2  | [`GIDenoiserAntiFireFly`](../../../shaders/pass/composite/GIDenoiserAntiFireFly.comp.glsl)                                                                        | 可选 anti-firefly pass                                            |
| 3  | [`GIDenoiserGIMip`](../../../shaders/pass/composite/GIDenoiserGIMip.comp.glsl)                                                                                    | 从 SSBO 0 offset 16 indirect dispatch，构建 diffuse/specular mip 输入 |
| 4  | [`GIDenoiserHistoryFix`](../../../shaders/pass/composite/GIDenoiserHistoryFix.comp.glsl)                                                                          | 修复低置信 history                                                   |
| 5  | [`GIDenoiserBlur`](../../../shaders/pass/composite/GIDenoiserBlur.comp.glsl)、[`GIDenoiserPostBlur`](../../../shaders/pass/composite/GIDenoiserPostBlur.comp.glsl) | 可选 blur 与 post-blur pass                                        |

Accum 还会更新 `history_gi6`，RG 通道存储漫反射/镜面反射亮度的二阶矩，使用 fast history 的 alpha。
重投影沿用 fast color 的 bilinear 或边缘感知四点权重；镜面反射使用 virtual point。
一阶矩直接取累积后的 fast color 在工作色彩空间中的亮度。累积完成后，`transient_gi_variance` 的 RG 存储
漫反射/镜面反射方差，BA 存储绝对标准差，计算发生在 anti-firefly、history fix 和 blur 之前。
HistoryFix 使用 5x5 tent 核过滤方差（可分离权重 `[1, 2, 3, 2, 1]`，总权重 81），存入
`transient_gi_filteredVariance`：RG 为过滤后的方差，BA 为其平方根。屏幕边界钳制到最近像素，
非实体像素贡献零。`SETTING_DEBUG_DENOISER` 以灰度显示过滤后的方差或绝对标准差，
使用通用的 debug 曝光和 gamma 控制。M2 history 保持纯时域累积，不参与空间滤波；无效 history 与 fast color 一起重置。

两遍 blur 都以 `1 / (1 + variance / (fastLuminance² + 1e-6))` 分别计算漫反射/镜面反射的稳定度因子。
该因子将有效 history length 从 1 插值到累积值，将 hit-distance 因子从 1 插值到原有值。
方差越高，有效 history 越短，hit-distance 抑制越弱；随后沿用现有的 accumFactor 和核计算。
半径限制、几何权重和镜面反射 roughness 对核形状的控制仍然生效；存储的时域 history length 不变。

重投影输入还包括 `history_viewZ`、历史/当前 view normal、geometry normal、edge mask、roughness 和 average view-Z。修改 tile
格式或生命周期时必须同步 [`shadesmith.json`](../../../shaders/shadesmith.json)，不能只改 sampler。

## 估计器约定

- 体素追踪区分三种结果（[`VoxelTrace.glsl`](../../../shaders/techniques/voxel/VoxelTrace.glsl) 中的
  `voxel_traceExhausted`）：命中；离开网格，按天空着色（网格边界假设）；步数耗尽，对初始候选是合法的零辐射样本，辐射缓存（RC）
  候选和重验证则丢弃它。
- RC face 把零辐射样本计入 M 和控制变量权重，但不会选中它们，因此一个 face 可能有有效的 estimate 却没有选中的样本（[
  `RadianceCache.glsl`](../../../shaders/techniques/gi/RadianceCache.glsl) 中的 `rc_reservoirValid` 与
  `rc_reservoirHasSample`）。
- face 的 estimate 是余弦加权的平均入射辐射 C_n。RC 查询和 RC 反弹都按均匀入射着色：
  `(albedo * dielectric * (1 - F(NoV)) + E_spec(NoV)) * C_n`（[
  `ResampleMaterial.glsl`](../../../shaders/techniques/gi/ResampleMaterial.glsl) 中的
  `resampleMaterial_uniformIncidenceAlbedo`），与 face reservoir 当前的样本方向无关。

## 蒙特卡洛参考

`SETTING_GI_USE_REFERENCE`（Debug 页）用初始候选单样本估计的 FP32 累积均值替代显示的 GI，用于测量采样器的偏差和方差。

| 顺序 | Pass | 作用 |
|----|------|----|
| 1 | [`GIReferenceAccumulate`](../../../shaders/pass/composite/GIReferenceAccumulate.comp.glsl) | 在体素 fallback 之后运行，把与复用输出同样着色的 `L / q` 累加到 `uimg_giReferenceDiff/Spec`（RGBA32F；漫反射 alpha 存亮度平方和，高光 alpha 存样本数）；首帧或视图、投影、相机位置变化时重置 |
| 2 | [`TranslucentBackComposite`](../../../shaders/pass/composite/TranslucentBackComposite.glsl) | 从参考图像读取均值，替代降噪后的 GI |

模式：1 完整 Li；2 冻结 Li，去掉命中点的 RC 项，使被积函数不随时间变化；3 熔炉测试，累加 `cos / (pi q)`，在关闭法线贴图的平面上
均值必须为 1。复用和降噪仍然运行，但不会显示到屏幕。累加假设一天中的时间静止。

## 路径引导

`SETTING_GI_PATH_GUIDING`（默认关闭，需要 RC）把漫反射初始候选和 RC 更新的一部分采样交给按 RC face 学习的两个折叠 von Mises-Fisher
lobe，训练数据来自 ReSTIR 的最终样本。代码在 [`PathGuiding.glsl`](../../../shaders/techniques/gi/PathGuiding.glsl)；采样器和
混合 pdf 在 [`InitialSample.glsl`](../../../shaders/techniques/gi/InitialSample.glsl) 和
[`RadianceCacheUpdate.glsl`](../../../shaders/techniques/gi/RadianceCacheUpdate.glsl)。

RC fresh ray 读取匹配的上一侧 face 统计，在寄存器中执行与
[`GIPathGuidePrepare`](../../../shaders/pass/composite/GIPathGuidePrepare.comp.glsl) 相同的衰减和封顶，然后拟合 lobe。
256 个技术 bin 中最多一半用于引导，其余使用 cosine 采样。fresh CV 估计为 `Li * (cos(theta) / pi) / q`，
fresh RIS 权重也包含相同的 proposal 修正。reservoir 保存未加权的 radiance，因此 temporal CV、cosine 测度下的空间重连接
和 M clamp 保留现有契约。训练仍只来自屏幕空间 ReSTIR 最终样本。没有可用统计的 face 使用 cosine 采样，修正系数为 1。
固定 debug lobe 同样用于 RC 光线。RC 仍丢弃耗尽的追踪；改变 proposal 可能改变拒绝比例，因此这不会使完整 RC 估计器无偏。

| 顺序 | Pass | 作用 |
|----|------|----|
| 1 | [`GIPathGuidePrepare`](../../../shaders/pass/composite/GIPathGuidePrepare.comp.glsl) | 在 RC 更新之后，使用同一份间接 face 列表：把每个 face 的统计从上一侧的槽位移到当前槽位，衰减 0.9，样本数上限 256 |
| 2 | [`GIReSTIRInitalSampleHiZ`](../../../shaders/pass/composite/GIReSTIRInitalSampleHiZ.comp.glsl) | 读取时分别拟合每个 lobe（至少 8 个样本，平均合向量长度至少 0.1，kappa 不超过 32）；拟合成功的 lobe 按样本数分配一半的漫反射采样 bin |
| 3 | [`GIReSTIRPairedSpatialShade`](../../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl)、[`GIReSTIRSpatialReuseTrace`](../../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl) | 每个像素写一条训练记录（最终样本方向，以 f_d / (f_d + f_s) 的概率保留）；trace pass 删除被遮挡邻域样本的记录 |
| 4 | [`GIPathGuideSplat`](../../../shaders/pass/composite/GIPathGuideSplat.comp.glsl) | 四分之一分辨率，每个 4×4 tile 取一个轮换像素，用原子操作累加到对应 face 两个在线方向聚类中较近的一个（与第一个聚类夹角余弦低于 0.5 的方向会填入空的第二个聚类）；结果从下一帧开始使用 |

资源：

- SSBO 6 `pg_stats`：每个 lobe、每个 RC 槽位、每侧一个 uvec4（定点方向和与样本数），大小 32 B × 2 × `SETTING_RC_POOL_SIZE`
  （默认池 16 MiB），只在开启引导时声明。
- `transient_pathGuide_trainRecord`（RG32UI）：与已有 RG32UI atlas 槽位共用，atlas 不变大。
- GlobalData 的 `pg_*Counter` 字段，由 `SETTING_DEBUG_VOXEL_COUNTER` 显示。

`SETTING_DEBUG_PATH_GUIDE` 1–4 用固定的世界向上 lobe 替代学到的 lobe，用于验证采样器；5 和 6 通过 Debug Output 显示份额较大的
学习 lobe 和覆盖情况。

限制：每个 face 两个 lobe 仍无法表示更多光源方向或 face 内部的视差；所在 face 没有 RC 槽位的像素不会被引导。单 lobe 和指向该
face 平均命中点的 lobe 都经过评估后删除：双 lobe 在所有场景中不差于单 lobe，除一个场景外都优于命中点 lobe，开销也比后者低。
仅对屏幕光线使用双 lobe（`composite_total` +1.7%）时，2026-09-30 的同时间误差下降为 `flatroom-hard-lighting-720p` 21%、`gnlxc-normal-1`
13%、`night-gi-1` 11%，其余场景在 ±3% 以内；只有 flatroom 通过困难场景 20% 的门槛，所以该设置默认关闭。

## 设置

GI 设置集中在 [`scripts/options.main.kts`](../../../scripts/options.main.kts) 的 GI 与 denoiser screens：

- 追踪：`SETTING_GI_INITIAL_SST_STEPS`、`SETTING_GI_VALIDATE_SST_STEPS`、`SETTING_GI_SST_THICKNESS`。
- probe/sky：`SETTING_GI_PROBE_FADE_START/END`、`SETTING_GI_MC_SKYLIGHT_ATTENUATION`。
- reuse：`SETTING_GI_TEMPORAL_REUSE_LIMIT`、`SETTING_GI_SPATIAL_REUSE`、`SETTING_GI_SPATIAL_REUSE_COUNT`、
  `SETTING_GI_DECORRELATE`。
- 引导：`SETTING_GI_PATH_GUIDING`。
- Debug 页：`SETTING_GI_USE_REFERENCE`、`SETTING_DEBUG_PATH_GUIDE`、`SETTING_DEBUG_RC_MODE`。
- denoiser：spatial enable/sample counts、history lengths、fast-history clamping、flicker
  suppression、anti-firefly、history-fix weights。

Profiles 主要缩放 SST steps、spatial reuse count 和 denoiser sample counts。任何新的 `SETTING_*` 都必须先在 options DSL
中注册，再由 GLSL 或 program 条件使用。

## 维护约束

- reservoir 的 pack/unpack、MIS metadata 和 producer/consumer 必须同改。
- temporal tile 必须与当前/上一帧 jitter、camera transform 和 G-buffer 语义一致。
- edge classification/dilation 必须保持在 reprojection 与 accumulation 之前。
- 改 spatial 批大小时，同步 program count thresholds、base sample index 和 indirect 工作队列布局。
- Paired pass 中的 ratio resolve producer、spatial shading 中的读取以及 trace pass 的重新混合共享 `transient_ssgiSpecOut`
  与临时记录布局，必须同改。
- 验证至少覆盖静止收敛、相机移动、disocclusion、屏幕边缘和设置切换后的 history reset。
