# 大气、天空与云

语言：简体中文 | [English](../../en/modules/atmosphere-clouds.md)

Alpha Piscium 把低频 LUT、屏幕相关的 sky-view/epipolar 数据、体积云和最终局部合成分开更新。大气及云源项预计算位于 `begin`，云输运位于 `deferred`，云渲染位于
composite 前段，空气/水体体积在 GI 后、透明合成前完成。

## 代码位置

| 路径                                                                                                                                                                                             | 职责                        |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| [`shaders/techniques/atmospherics/air/lut/`](../../../shaders/techniques/atmospherics/air/lut/)                                                                                                | 大气 LUT 生成与 API            |
| [`atmospherics/air/Raymarching*.glsl`](../../../shaders/techniques/atmospherics/air/)                                                                                                                                                           | 空气介质 ray marching         |
| [`SliceEndPoints.comp.glsl`](../../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl)                                                                                                                                                    | Epipolar 切片端点             |
| [`EpipolarScattering.comp.glsl`](../../../shaders/techniques/atmospherics/air/EpipolarScattering.comp.glsl)                                                                                                                                                | 空气 epipolar scattering 核心 |
| [`Cumulus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cumulus.glsl)、[`Cirrus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cirrus.glsl)、[`Mediums.glsl`](../../../shaders/techniques/atmospherics/clouds/Mediums.glsl) | 云密度、介质与相函数                |
| [`RenderVolumetric.comp.glsl`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl)                                                                                                                                               | 低层体积云主渲染                  |
| [`Accum.comp.glsl`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)                                                                                                                                                       | 低分辨率云的时空累积与上采样            |
| [`atmospherics/clouds/amblut/`](../../../shaders/techniques/atmospherics/clouds/amblut/)                                                                                                                                                                  | 云环境光 LUT 的采样与汇聚           |
| [`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl)、[`LocalComposite.glsl`](../../../shaders/techniques/atmospherics/LocalComposite.glsl)                        | 天空与局部体积合成共享代码             |
| [`shaders/util/Celestial.glsl`](../../../shaders/util/Celestial.glsl)                                                                                                                          | 项目中的太阳、月亮、星图和星座逻辑         |

## Begin 阶段：LUT 与帧准备

当前程序列表中的顺序：

| 顺序 | Pass                                                                                                        | 作用                                      |
|----|-------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| 1  | [`GenerateTransmittance`](../../../shaders/techniques/atmospherics/air/lut/GenerateTransmittance.comp.glsl) | 更新 256×64 `persistent_transmittanceLUT` |
| 2  | [`SliceEndPoints`](../../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl)                   | 更新本帧 epipolar 几何                        |
| 3  | [`GenerateMultiSctr`](../../../shaders/techniques/atmospherics/air/lut/GenerateMultiSctr.comp.glsl)         | 更新 32×32 `persistent_multiSctrLUT`      |
| 4  | [`clouds/amblut/Sample`](../../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl)           | 交替使用积云和卷云高度，采样 1024 条全空间大气光线，为卷云单独存储地面的直接反射光 |
| 5  | [`GenerateSkyViewLUT`](../../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl)       | 写入 `uimg_skyViewLUT`                    |
| 6  | [`clouds/amblut/Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl)           | 写入 32×33 `persistent_cloudsAmbLUT`：32×32 的卷云方向 tile，以及最后一行的25 个积云值（余弦加权天空辐照度和 24 个晴空方向）；归约全部 subgroup，并在状态切换时重置未更新区域的历史 |

固定尺寸 LUT 在 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 中声明；屏幕尺寸或可配置图像在 [
`scripts/shaders.properties`](../../../scripts/shaders.properties) 中声明。

`uimg_skyViewLUT` 的宽高都使用 `SETTING_SKYVIEW_RES`，六个切片分别存储完整大气、卷云下方和上方区域的散射与透射率。
`uimg_epipolarData` 的宽使用 `SETTING_EPIPOLAR_SLICES`，高由
`SETTING_SLICE_SAMPLES` 映射到 385、769、1537 或 3073。新增支持的 sample count 时，要同时更新属性尺寸映射和 shader 索引逻辑。

## 云 pass

启用 `SETTING_CLOUDS_CU` 时：

| 顺序 | Pass                                                                                             | 作用                                                                                                               |
|----|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| 1  | [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | 步进对流云塔，结合高分辨率单次散射、不透明度加权光照采样和局部输运 |
| 2  | [`clouds/ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)          | 对低分辨率结果执行历史累积、置信度/方差处理和上采样；该阶段为 [`ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl) |
| 3  | 后续空气/局部合成                                                                                        | 读取累积后的云散射和透射率                                                                                                    |

主要 screen tile 为 `transient_lowCloudRender`、`transient_lowCloudAccumulated` 和 `history_lowCloud`
（RGBA32UI）。前三个字打包有符号 FP16 散射和透射率，最后一个字打包 FP16 历史长度及以大气千米为单位的不透明度加权深度。
手写属性片段声明了自定义 cloud phase LUT、cirrus、cumulus detail 和 curl 纹理。高层 cirrus 由共享 sky/cloud
路径采样，不使用独立 compute program。

### 浓积云

[`Cumulus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cumulus.glsl) 将宽而浅的凝结底部
与三个高度不同的上升云团融合。每个云团由纵向主体和圆润云顶组成。单元种子控制高度、倾斜和侧向分支；
覆盖率控制水平宽度，与高度独立。邻居搜索范围包含云底和倾斜云团，归一化包络使侵蚀尺度与单元半径成比例。

四个 octave 的 Worley 纹理塑造大鼓包，并施加垂直拉伸和 curl 位移，再叠加两级更细的侵蚀。
噪声扰动凝结边界，密度在有限宽度内过渡；液态水密度在云底上方增加。
默认云底高度为 1 千米，云层厚度为 3.5 千米。128³ R8 鼓包纹理使用
[`cumulus_detail1.json`](../../../data/strepitus/cumulus_detail1.json) 中的 Strepitus 配方，第二张细节纹理提供细节侵蚀。

[`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) 使用
[`clouds/Common.glsl`](../../../shaders/techniques/atmospherics/clouds/Common.glsl) 中的球壳求交。
主光线以非均匀距离区间探测整个可见范围，默认使用 128–256 个采样。空采样之后遇到非空采样时，
通过八次二分定位边界。非空区间再按密度上界细分，使每段名义相机光学深度不超过 0.5，
替代在长不透明区间内部一个平均自由程处集中求值的方式。均匀介质的 Beer-Lambert 解析积分使用每段实际长度，
已经积分的区间不会重复计入。密度求积与空区间探测仍是有限分辨率近似。地表交点截断光线；
云内行进距离上限为 60 千米，距相机的最远距离为 120 千米。
当剩余透射率低于 `0.005` 时，用局部光源项补全不透明尾段，并将透射率归零。

灰色云消光为所有波长提供统一的标量相机不透明度 PDF。主步进积分透射率和加权深度，reservoir 只保存选中的距离。
令 `A = 1-T_cloud`、`d alpha_cloud = T_cloud sigma_t ds`，对前景空气积分换序可得
`T_cloud L_clear + integral d alpha_cloud [L_air(0,s) + T_air(0,s) albedo (P B + P S)]`。
此式来自[输运积分方程](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer)。
渲染器仅在选中深度计算方括号内的内容，SS/空气与 MS 使用各自的采样数。
所选求积模型的期望不变，以深度采样方差换取更少的缓存/空气查询；不再保存逐深度缓存预测或 RGB reservoir 权重。
默认使用两个直射光照采样，局部采样预算为四时使用四个。
太阳和月亮分别使用实际密度、大气透射率、地球遮挡及采样日月圆盘方向上的完整相函数。
每个选中深度的天空/地面单次散射使用一个连续相函数重要性方向，PDF 为三个 AP0 相函数密度的等权混合。
精细阴影光线用真实消光积分 16 个指数密度区间，移除比率跟踪噪声和无界空碰撞工作量，
但有限求积仍可能遗漏远处的狭窄遮挡物。积云默认使用 HG+Draine（`SETTING_CLOUDS_CU_PHASE_RATIO = 0`）；
直射、方向采样、相函数矩阵和矩系数统一读取同一个混合设置。

积云环境方向缓存将晴空辐亮度与地面反射分开。在精细散射点，
[`Transport.glsl`](../../../shaders/techniques/atmospherics/clouds/Transport.glsl) 将入射光线与地球求交，
在实际交点计算日月反射，包括地面到光源之间的真实云密度光柱及返回云点的大气透射率，
替代原来的均匀、未受云影遮挡的地面直射贡献。粗边界在相机水平位置和云层中间高度计算此项。
地面漫射照明仍采用晴空近似，但用余弦加权的 `E_sky / pi` 替代未加权半球平均的两倍；
尚未求解云与地面的多次散射耦合。

阴影光柱的指数区间以最小消光平均自由程的四分之一为距离尺度，远处的云层出口不会再将首段阴影采样
拉伸到跨过小表面结构。光柱最多覆盖 12 千米，达到不透明光学厚度（最小通道深度 12）后停止。
远处阴影求积仍然较粗。旧的 extinction octave、逐阶衰减权重及 `0.15` 环境光学深度缩放继续保持移除。
积云使用灰色消光/散射系数 `0.1854716894` 和 `0.1854716020`，取自先前 AP0 系数的平均值。
照明和所选相函数保留 RGB。相机、直射阴影与首段局部飞行使用此真实消光；高阶约化特征线解析包含
同一角向通道内的重复散射，其系数与真实消光以及输运消光 `a1` 均有区别。

空气散射和透射率从相机到每个选中深度积分一次，使用 4–32 个采样。
云不透明度重要性采样统一加权前景辐亮度与云光照；主密度循环不再携带空气光照状态或查询光照缓存。
[`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl) 将结果合成到天空之上，
不再把积云放置在固定的云层中间高度平面。从卷云上方观察时，卷云的前景散射和透射率会作用于积云贡献。
云历史按不透明度加权深度重投影，并计入水平风平流和大气高度比例。
深度差异会降低历史置信度；世界时间变化使用已有光照历史重置，风计时器回绕时拒绝历史。
单个深度仍近似多个重叠云层，竖直细节运动未参与重投影。风同时平移密度场及其细节，截图模式将其冻结。
覆盖率、密度或云层厚度为零时输出空云层。卷云保留独立的照明模型和多重散射半径控制。

### 共享方向输运与双通道校正

[`Transport.glsl`](../../../shaders/techniques/atmospherics/clouds/Transport.glsl) 将云内散射辐亮度
`S = L - B` 存储在 24 个等立体角方向单元中，方向为 `(2,1,1)/sqrt(6)` 的带符号排列。
目标方程是 `A S = sigma_s P B`，使用守恒特征线输运，入射散射光边界为零。
空单元的消光严格为零，保留云间自由传播。相机缓存仅存储 `P S`，采样后仍需由实际高分辨率密度散射；
不会再次叠加 `S` 或未碰撞天空光。四方向正值插值保持常量光场，立方对称性保证各基函数积分权重相等；
空间插值使用实际单元中心。

| 顺序 | Pass / 阶段 | 作用 |
|---|---|---|
| 1，begin | [`CloudTransportPrepare`](../../../shaders/pass/begin/CloudTransportPrepare.comp.glsl) | 构造网格、相函数矩阵和系数层级；四个体积求积点各定义一条含八个密度采样的光源弦线，根据入口透射率与 `1-exp(-tau)` 积分 `rho B` |
| 2，begin | [`CloudTransportPrepare`](../../../shaders/pass/begin/CloudTransportPrepare.comp.glsl)，`CT_HISTORY` | 按相机位移和云平流将上一帧 `(P - f I) S` 重投影到独立暂存区；无效历史提供零初值 |
| 3，begin | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl)、[`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 计算未碰撞天空/地面光，由 `P B` 和重投影后的散射光初值构造当前源项 |
| 4，deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl) | 使用降序共享内存平面求解方向特征线，每工作组最多 48 KiB |
| 5，deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 将真实方向残差限制到两个耦合扩散通道；V-cycle 使用 4³ 块预条件 CG、实际中心延拓和红黑平滑 |
| 6，deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl)、[`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 提升并阻尼校正，保持辐亮度非负；残差或功率误差未达标时才执行后续迭代 |
| 7，deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 记录实际残差/功率，存储单元平均散射光，预卷积 `P S` 供相机和下一帧初值使用 |

空间离散之前先解析合并同一角向通道内的重复散射。令 `f = min_RGB(P_jj)`，定义
`sigma_f = sigma_t - sigma_s f`、`P_f = P - f I`，方程变为
`omega_j.grad S_j + sigma_f S_j = sigma_s (P B + P_f S)_j`。这是所选角向方程的代数重排，
对非均匀密度同样成立。`P_f` 保持非负，吸收仍为 `sigma_t-sigma_s`，等立体角矩阵保留功率收支。
相函数记录的 alpha 保存约化相函数 CDF；RGB 仍保存完整矩阵。PDF 由已存储的 CDF 差值计算，包含浮点舍入。

特征线区分入射光、出射面值和单元平均值。散射光的光学厚度为
`sigma_f / sum(abs(omega)/cell_size)`，未碰撞光仍使用 `sigma_t`。
平均为 `(1-w) L_out + w L_in`，其中 `w = 1/tau - 1/(exp(tau)-1)` 在薄介质使用稳定级数。
散射和吸收使用平均值，共享面输运与逃逸使用出射值；
残差、校正和功率算子保持同样区分。这替代了向下一格传递单元平均值的方式，后者使厚单元未碰撞透射率成为
`1/(1+tau)`。有限横向求积和段内常量源仍是空间近似。

双通道核心使用 `a_l = sigma_t - sigma_s beta_l`、`D0 = 1/(3 a1)`、正的快慢扩散根、
权重 `w, 1-w` 及成对交换 `kappa = w(1-w) sigma_s (1-beta2)`。
共享面的调和导通率与各通道独立的 Marshak 逃逸边界保持核心的守恒对称结构。
拟合的有效 `a3` 改变中等空间尺度响应，但不改变吸收或 `D0`。
这是标量均匀 P3 响应的双通道表示，并非精确的非均匀角向 P3 求解器。
渲染器使用它预条件方向方程，而非替换目标方程或额外贡献一份光照。

[`cloud-transport.py`](../../../scripts/cloud-transport.py) 积分实际过滤后的积云 Mie LUT 和
[HG+Draine 相函数](https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf)。
它在直射求值与矩提取前统一归一化 Mie 相函数，通过方向单元积分构造正、对称、行和为一的矩阵，
并在网格 Nyquist 频带内对 P255 连分式拟合 `a3`。LUT 保存两个逆 CDF、两个相函数矩阵、
相函数矩/归一化，以及不同相函数混合与输运单元尺寸的闭合拟合。
当前 HG+Draine 名义粒径参数为 11.54 µm；未照搬独立示例的 20 µm 拟合值。
低阶误差校正遵循 [synthetic acceleration](https://arxiv.org/html/2402.10488v2) 的思路；
具体 GPU 离散与有限迭代预算仍是近似。

| 资源 | 布局与生命周期 |
|---|---|
| [`cloud_transport.bin`](../../../shaders/textures/cloud_transport.bin)、`usam_cloudTransport` | 静态 1024×26 RGBA32F 相函数/闭合资产，416 KiB |
| SSBO binding 6 | FP32 AP0 方向光场、源项和校正，各层耦合系数/误差/RHS、归约与网格边界；32³ 为 56,240,912 字节，64³ 为 449,858,832 字节；使用当前介质/源项，并以重投影后的上一帧解作为初值 |
| `persistent_cloudsAmbLUT` | 原有 32×33 atlas：卷云保持 32×32；最后一行 x=0 保存余弦加权天空辐照度除以 π，x=1…24 保存各方向单元平均晴空辐亮度 |
| SSBO 6 内的 `ctStats` | 已执行的残差/步长、最终残差与功率、校正后功率误差、实际迭代数、历史有效性/位移和上一帧相机/时间/光源元数据；通过 buffer dump 检查 |

SSBO 使用单个不定长 `vec4[]` 和显式偏移；大型定长 SSBO 数组会使 NVIDIA 编译耗时异常增长。
偏移及索引运算保持无符号整数：实测 NVIDIA 路径在 64³ 下错误处理大有符号偏移，使散射光缓存错位一个 vec4，
并破坏归约地址。这是寻址错误，而非 NaN/Inf；两种分辨率都应核验回读的平均场/源项关系及功率和。
改变打包布局时须同步修改字节分配和回读程序。帧末 `ctRadiance` 保存出射值，`ctLifted` 保存单元平均值。
系数 RGB 保留求解器参数，零层 fast/weight/exchange 的 alpha 分量不再使用，保持零值。
旧边界模型的平均光缓存已移除。

实验网格水平覆盖 ±121 千米。每条水平轴将 75% 的单元均匀分配到相机两侧各三倍云层厚度的范围，分配时厚度下限为 0.1 千米。其余距离采用一阶导数连续的指数外缘，形状参数随所选厚度在编译期确定。
采样逆映射和实际中心插值使用同一套分配。竖直范围覆盖球面云层并延伸到名义云底以下 2.5 千米。
默认层厚 3.5 千米时，32³（默认）与 64³ 网格的内部横向间距分别为 875 米和 437.5 米。
`SETTING_CLOUDS_MS_ITERATIONS` 选择 1、2 或 4（默认），作为迭代上限。上一帧 `(P - f I) S` 只提供当前算子和源项的初值。
帧不连续、历史重置，或日月方向变化超过角半径时拒绝该初值。至少执行一轮完整迭代；
预测相对残差和校正后功率误差都低于 `0.005` 时，后续迭代提前停止。最终诊断重新计算实际残差与功率。
冷启动和突变仍可能耗尽预算而尚未收敛。

此实现尚未证明云图像的角向或空间收敛。24 方向会柔化低阶方向细节，粗单元仍有空间误差
（厚单元的扩散极限不同于物理 `D0`），有限光源弦线求积也可能遗漏狭窄受光区域。
有限网格使用空间均匀环境，其中地面直射项包含代表位置处的云影，局部路径则计算自己的地表交点。
仍未包含云与地面的多次散射反馈，以及晴空环境中的云影。
固定迭代预算留下可测量的残差与功率缺口；核心系数守恒不等于整个插值后的相机图像零误差。
提高网格分辨率成本较高，是精度控制，并不保证上述误差已经收敛。

从 [`scripts/`](../../../scripts/) 重新生成资产并运行独立稀疏矩阵参考检查：

```sh
python cloud-transport.py --output ../builds/cloud-transport/phase.json
python test/cloud-transport-check.py --output ../builds/cloud-transport/numerics.json
```

[`cloud-transport-check.py`](../../../scripts/test/cloud-transport-check.py) 检查 P3 恒等式、收敛后的有限体积正性与
功率平衡、薄密度缩放、真空间隙传播、叠加线性和实际特征线扩散极限。
其 `--gpu-buffer PATH --resolution 32` 模式检查 Iris SSBO 回读，并报告实际有限预算残差与功率缺口。
这些检查补充目标 Iris 编译、buffer 捕获、图像和 pass 耗时，不能替代与参考输运或照片的图像对照。

### 局部散射路径与不透明度采样

[`LocalTransport.glsl`](../../../shaders/techniques/atmospherics/clouds/LocalTransport.glsl) 对相机与二次散射
保留连续方向，随后使用与共享求解器相同的 24 通道方程。高阶飞行解析合并相函数的公共对角质量，
缓存提供未解析的远场。局部着色已移除法线、平面边界块、Marshak 到 HG 的重建、距离混合和密度 AO。

| 阶段 | 操作 |
|---|---|
| 相机积分 | 积分真实云透射率，仅保存按不透明度加权选中的距离 |
| 首段局部飞行 | 使用真实灰色消光及连续 RGB 相函数；选择离散延续通道之前先积分投影后的角向缓存 |
| 高阶飞行 | 沿实际密度使用 `sigma_f`，采样非负剩余相函数 `P - f I`；相机消光与物理阴影光线不受此移项影响 |
| 终段飞行 | 在 16 个指数密度区间上积分透射率；WRS 在已发生散射的条件下选择位置，同时加入加权散射回填与透射后的终点来光 |
| 外部源项 | 每条路径分别用正值 reservoir 选择一个天空事件及一个日/月事件；通过精细阴影计算真实辐亮度，不减去不准确的粗阴影 |
| 相机着色 | 在选中深度计算完整局部 MS、直射和前景空气；根方向采用分层时空蓝噪声，并映射到半开区间的采样格中心 |

`SETTING_CLOUDS_MS_LOCAL_SAMPLES` 可选 0、1（默认）、2、4 条路径；零查询共享 MS 缓存，并保留直射。
`SETTING_CLOUDS_MS_LOCAL_BOUNCES` 可选 1、2、4（默认）、8、16、32 个输运阶段。首阶段使用连续方向，
后续阶段包含任意多次同通道事件，因此不再代表字面上的物理散射次数。更多阶段延伸局部输运，
不会改变艺术化边界权重，也不保证有限样本噪声更低。

终段中，`T` 是积分到终点的透射率，`A` 是各段散射选择权重 `T_i (1-exp(-tau_i))` 之和。
按不透明度加权的 reservoir 选出位置 `X`，估计式为
`T C_j(endpoint) + A (sigma_s/sigma_f) [(P-f I) C + P B]_j(X)`。
条件位置采样保留积分期望，同时始终计算透射终点，替代散射/逃逸二选一，并避免每个精细密度步都查询完整缓存。
真空中 `T=1`、`A=0`；保守介质的恒定平衡光场保持不变。有限区间求积及三个粗单元长度的飞行范围仍有偏差。

在中间高阶事件，`(P-f I) C` 是缓存延续光照的已知条件期望。加上此项，减去采样到的 `C` 后继续追踪，
保持相同的有限深度输运目标。连续相函数 LUT 的有限采样精度仍是近似。UNORM 蓝噪声映射到 `[0,1)` 的采样格中心，避免 RGB 相函数采样索引落到第四个分量。

天空源方向连续；高阶源项先在选中的角向单元内均匀采样方向。光束源使用已积分的日/月方向单元相函数。
正值 proposal 使用未遮挡入射功率及路径/相函数权重。完全遮挡的源严格贡献零，不再靠随机减去错误的
粗受光预测来抵消。源项、投影延续与相机深度采样分别保留各自的概率因子，避免重复添加未碰撞或已散射光。

粗网格、24 通道高阶表示、飞行范围、密度求积和终点光场仍是近似。局部修正没有将净通量反馈给粗网格，
其功率平衡不能证明整张图像守恒。默认历史保持 32 帧，包含深度/风重投影与方差裁剪；
保留带符号辐亮度，历史计数使用正值重建权重。时间/FP16 偏差和有限样本噪声仍然存在。
提高深度是在细化所选模型，不能据此证明照片一致性或完整的角向/空间收敛。

[`cloud-local-transport-check.py`](../../../scripts/test/cloud-local-transport-check.py) 检查灰色飞行采样、
真空/平衡、空气/云耦合积分恒等式、UNORM 采样边界、相机及源项 reservoir、与独立连续平板对照的解析合并、条件延续、强制终段方差、
阴影求积和等质量狭窄通道。`--gpu-history PATH` 检查有限的带符号散射、透射率与深度。
主输运检查还通过 Iris SSBO dump 核验相函数 CDF、帧末 `P S`。

```sh
python test/cloud-local-transport-check.py --output ../builds/cloud-transport/local-numerics.json
```

### CPU 辐亮度参考

`cloud-light-transport-comparison` 工作项沿用冻结规格
`d2f1ef0fe2b1ac9a962a062fe7d2af7f922611ba`，实时基线为 wt9 的
`c09cfdc76e79cb3e4e986d16e29730777283e78d`。首个 CPU 阶段独立于 24 方向缓存及局部重建。
[`cloud-reference.cpp`](../../../scripts/test/cloud-reference.cpp) 在八个恒密度椭球的固定并集中追踪连续方向。
并集弦长、重叠、自由飞行和太阳透射均解析求值。直接采样归一化的 11.54 um HG+Draine 相函数，
没有角度 LUT、空间网格或固定散射阶截断；第 32 次事件起使用俄罗斯轮盘。
它求解[体积输运方程](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer)。

当前参考的延续方向使用重要性分布 `q = 0.8 p + 0.1 p_side + 0.1 p_back`，每次事件以 `p/q`
补偿通量；后两瓣分别朝向两个固定太阳轴。物理相函数、自由飞行分布和逐次太阳连接积分保持相同。
引导权重传零即可使用原来的纯相函数采样。用相函数解析矩核对提议分布加权后的归一化和一阶矩。
这减少罕见的近前向连接，不依赖缓存、相函数对角合并或密度近似。

样本云体尺寸约 150 x 216 x 110 米。固定的 32 x 40 正交像素中心相机位于 Z=-450 m，朝 +Z；
X 范围 [-100,100] m，Y 范围 [-10,240] m。并集内部密度 0.15，消光 0.027820753413 /m，
反照率 0.999999528504。太阳半径 0.266 度、垂直入射辐照度 1，分别从 +Y 侧照或 +Z 逆照。
天空、空气、地面和邻塔均关闭。manifest 记录各椭球、随机种子方案、输入及原始 float64 通道。
这是可复用的样本云体，不是生产中的侵蚀程序化密度。

当前引导参考位于运行 `2b8a366e-740b-4da8-b7b8-09cc28b1c15c`：八组独立流，共 671,088,640 条路径、
839,402,395 次散射事件，单条路径最多 33 次事件。增加预算后，早期样本检查点逐位复现。
末版仍具有有限的 Monte Carlo 不确定性：

| 总路径/像素 | 侧光总辐亮度 RMS 相对标准误 | 逆光总辐亮度 RMS 相对标准误 |
|---|---:|---:|
| 32,768 | 5.571% | 0.3276% |
| 131,072 | 3.270% | 0.1616% |
| 524,288 | 1.453% | 0.08113% |

最后一档中，多次散射分量的 RMS 相对标准误分别为 1.511% / 0.3969%（侧光/逆光）。
侧光薄缘、凹口和云芯的区域均值标准误为 0.0827%、0.2104%、0.1476%；它们最后两个不相交样本块
的差异均在 95% 区间内。整云侧光均值在这两个块间仍变化 0.44%（2.66 个标准误），保留该观察，
不把参考称为无噪真值。与旧采样器对照，各记录区域/分量均值最多相差 2.45 个标准误。
这些是参考收敛诊断，不是生产渲染器验收。

早期运行 `802ff939-d1e4-4654-bdc1-2c41fdb030b3` 已完成 41,943,040 条路径，使用八组独立重复，
逐级累计 512/2,048/8,192/32,768 路径/像素。相函数积分距 1 为 2.11e-9，采样方向均值及解析球的一次
散射均落在两个标准误内；并集及真空间隙检查通过。逆光的像素 RMS 标准误/辐亮度 RMS 从 2.52%
降至 0.318%，侧光仍为 36.2%，其凹口均值标准误为 9.32%。**首轮侧光结果还不是收敛的图像参考。**
少量接近前向峰的连接主导多次散射方差。
[`cloud-reference-report.py`](../../../scripts/test/cloud-reference-report.py) 保留单次/多次分量、独立重复的
置信区间及不相交样本块对照。每种光向的显示使用明确标注的线性范围，同一行各分量共用范围；数值比较使用原始辐亮度。

用开启优化的 C++20 编译器编译程序，传入预分配输出目录、每组最大样本数、独立组数、CPU 线程数
及可选引导权重。本轮使用 Clang 22.1.8、`-O3 -std=c++20`，四线程 Idle 优先级，参数 `65536 8 4 0.2`，
未开 fast-math。再对该目录运行 Python 报告脚本，`--compare` 指向旧运行进行估计器对照。
Vela 登记运行及作业源码。生产云体一致性、邻塔影响、天空入射、
运动和 Iris 成本仍分别待验；本阶段没有操作实机或测量 GPU。

<!-- vela:evidence {"run":"2b8a366e-740b-4da8-b7b8-09cc28b1c15c","path":"reference.json"} -->
<!-- vela:evidence {"run":"2b8a366e-740b-4da8-b7b8-09cc28b1c15c","path":"radiance.bin"} -->
<!-- vela:evidence {"run":"2b8a366e-740b-4da8-b7b8-09cc28b1c15c","path":"convergence.json"} -->
<!-- vela:evidence {"run":"2b8a366e-740b-4da8-b7b8-09cc28b1c15c","path":"reference.png"} -->
<!-- vela:evidence {"run":"2b8a366e-740b-4da8-b7b8-09cc28b1c15c","path":"convergence.png"} -->

<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"reference.json"} -->
<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"radiance.bin"} -->
<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"convergence.json"} -->
<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"reference.png"} -->
<!-- vela:evidence {"run":"802ff939-d1e4-4654-bdc1-2c41fdb030b3","path":"convergence.png"} -->

### 邻塔对照

运行 `046ae71f-2a18-44dd-a029-94bd57f95af6` 将同一组八个椭球沿 X 平移 180 m 作为邻塔。两个包围盒之间的 X 间隙为 27 m，邻塔与整个相机取景范围相隔 4 m。以各包围盒沿光轴的最大跨度计算，从主塔任意一点到整个太阳盘的光线都避开邻塔，侧光和逆光的横向余量分别至少为 25.997 m 和 26.508 m。太阳一次散射的差值严格为零。

[路径追踪器](../../../scripts/test/cloud-reference.cpp) 保留原独云太阳样本。射线逃出主塔、与邻塔相交时，按邻塔不透明度分出一条支路，在发生碰撞的条件下采样位置，扣除被挡住的天光，再在两座塔内继续追踪，允许光路返回主塔。上半球天光为均匀辐亮度，水平面收到的天空辐照度为太阳垂直入射辐照度的 10%，下半球为黑。只统计云散射辐亮度。八组独立流合计 524,288 路径/像素，独云太阳分量的均值、方差和协方差在每档样本量上均逐位复现封存的引导参考。

| 邻塔引起的主塔区域均值变化 | 上部薄缘 | 右侧凹口 | 云芯 |
|---|---:|---:|---:|
| 仅侧光太阳 | +1.020% | +1.664% | +1.195% |
| 侧光太阳与天光 | +0.348% | +0.601% | +0.553% |
| 仅逆光太阳 | +0.0000180% | +0.0000373% | +0.0000868% |
| 逆光太阳与天光 | +0.0000212% | +0.0000411% | +0.0000886% |

百分比以同一照明下独云的区域均值为分母。十二项配对差值各自的 95% 区间均高于零。仅侧光时，凹口增量除以太阳垂直入射辐照度为 8.73 × 10⁻⁵ sr⁻¹，区间为 [8.07, 9.39] × 10⁻⁵。逆光时凹口的绝对增量与侧光接近，独云凹口却已达约 267.38 sr⁻¹。这组几何下，邻塔贡献不足以产生明显的逆光亮边。单独看天光，整云一次散射减少 9.49 × 10⁻⁶ sr⁻¹，多次散射增加 9.23 × 10⁻⁶ sr⁻¹，净变化的区间 [−6.85, +1.63] × 10⁻⁷ sr⁻¹ 跨零。三个指定区域的天光净变化在该上半球边界条件下均为正。

在原参数 `65536 8 4 0.2` 后追加 `neighbor`，再用[报告脚本](../../../scripts/test/cloud-reference-report.py)的 `--compare` 指向引导运行 `2b8a366e-740b-4da8-b7b8-09cc28b1c15c`。报告保留各散射分量的独立重复、配对区间、不相交样本块差值、统一曝光的整图与局部放大差值。这些测量使用固定样本云体。

<!-- vela:evidence {"run":"046ae71f-2a18-44dd-a029-94bd57f95af6","path":"reference.json"} -->
<!-- vela:evidence {"run":"046ae71f-2a18-44dd-a029-94bd57f95af6","path":"radiance.bin"} -->
<!-- vela:evidence {"run":"046ae71f-2a18-44dd-a029-94bd57f95af6","path":"neighbor.json"} -->
<!-- vela:evidence {"run":"046ae71f-2a18-44dd-a029-94bd57f95af6","path":"neighbor.png"} -->
<!-- vela:evidence {"run":"046ae71f-2a18-44dd-a029-94bd57f95af6","path":"neighbor-regions.png"} -->

### 固定云体的 Iris 对照

[CloudTransportPrepare](../../../shaders/pass/begin/CloudTransportPrepare.comp.glsl) 对每格太阳光柱取四个横向位置，各取八个密度样本。固定相机距云塔 450 m，层厚设为 0.3 km。[CPU 采样核对](../../../scripts/test/cloud-grid-fixture.py)显示，在两种光向和原版两档网格上，独云命中的格数均为零。原网格近处格宽分别为 1,000 m 和 500 m。[Transport](../../../shaders/techniques/atmospherics/clouds/Transport.glsl) 的实验分配在这一层厚下将近处格宽缩为 75 m 和 37.5 m，沿用 24 个方向、原密度积分、格数和远边界。

在 1,024² 个像素中心积分解析弦长，单塔体积约为 1,642,924 m³，将该积分分辨率减半，体积相差 9 m³。实验 32³ 网格在两种光向下均命中 12 格，单塔体积误差分别为侧光 −0.38%、逆光 −6.30%。64³ 的两项误差为 −1.12% 和 +0.61%。这项测量只覆盖输入密度。最终辐亮度还取决于角度与空间输运、细级重建和历史积累。网格取舍待 Iris 的同曝光画面与实际 pass 耗时对照。

沿同一组正交射线后移相机只会增加一段真空。在 2、6、12 km 三个距离上，实验 32³ 网格在两种光向下均漏掉整座独云。64³ 在 2 km 只命中侧光采样，在 6 km 只命中逆光采样，到 12 km 两者均漏掉。局部细级光路仍可直接查询密度。

同一组射线上，旧版 300 m 光柱采用四倍太阳角半径抖动，最大横向位移为 5.57 m，两塔间隙为 27 m。只有 50 m 外的覆盖度法线采样可能感知邻塔，在薄缘、凹口与暗芯分别涉及 9/112、1/16 和 0/58 个像素。旧光柱的暗芯对邻塔的太阳增量因此恰为零。在校准密度下，旧算法的局部天空透射为 exp(−27.82075)，默认多次散射反馈增益为 19.9903。即使每个天空样本都取较强 Cornette–Shanks 相函数的峰值，按公式所得的纯天空散射仍小于 1.1 × 10⁻¹⁰ sr⁻¹。八组独立参考在独云薄缘、凹口和暗芯的纯天空散射分别为 0.010078 ± 0.000002、0.010946 ± 0.000005、0.014874 ± 0.000002 sr⁻¹（一个标准误）。用 `cloud-reference-report.py <邻云运行> --column-check <输出.json>` 可复算。

[Iris 对照脚本](../../../scripts/test/cloud-iris-fixture.py)从邻云运行的 `reference.json` 生成补丁。光柱基线 `fb8c36487` 是重写前的直接父提交，相机光照 shader 与规格修订 `d2f1ef0f` 相同（blob `4fa191b0e`）。与它对照的是 `c09cfdc76` 输运和按层厚分配格宽的候选。正交校准使用 `--sun side|back`，按需加 `--neighbor`、`--sky`。`composite9` 将已打包的半精度云辐亮度与透射复制到 `colortex0` 供读取。层厚用 0.3 km，相函数混合比为零，尺度为每千米 1,000 方块，曝光固定。性能和运动对照用 `--world-origin X_M Z_M` 在原透视路径中放入同一密度，沿用游戏光源。分别统计新增输运 pass、云渲染、历史积累、稳定的对照 pass 及整帧。固定样本用于衡量候选在该场景下的开销，新云形接入后还须用改造前同场景整帧复查 1.30 倍约定。

两条路径均接收与参考相同的有限太阳盘。着色器入口使用太阳辐亮度乘立体角，单位法向辐照度在该入口应填写 `2 / (1 + cos(角半径))`。使用运行 `7a42496c-59bb-4bf5-8e91-8f520e82ef83` 内换算后的补丁哈希。光柱的正交校准在天空归约时为 RTX 上 1,024 个调用的 32 个子组预留部分和，原版仅分配了 16 项。原生性能对照保留基线 shader 的天空计算。

在 Vibris 回执中核对实载源码、设置和场景后，运行 `cloud-reference-report.py <邻云运行> --iris-readout <case>.mean.bin <输出.json>`。先对共享一条参考射线的屏幕像素取均值，再对各条射线等权比较。输出包括线性 RGB 误差、参考统计误差、透射误差及真空背景检查。GPU 区域均值标准误的上界允许像素间存在相关性。

用户选择将地形深度合成留在本次光照工作范围之外。[SkyComposite](../../../shaders/techniques/atmospherics/SkyComposite.glsl) 仅在天空深度像素读取累积云结果。云上俯视时，终点为地形的射线会漏掉前方低云。该视角记录可见性限制，无遮挡射线可与物理参考比较。

<!-- vela:evidence {"run":"bb2ea177-6c80-4ec0-bb82-4ca3a1ab350e","path":"grid.json"} -->
<!-- vela:evidence {"run":"bb2ea177-6c80-4ec0-bb82-4ca3a1ab350e","path":"column.json"} -->
<!-- vela:evidence {"run":"7a42496c-59bb-4bf5-8e91-8f520e82ef83","path":"iris-inputs.json"} -->

## 空气、深度层与合成

| 阶段    | Pass/代码                                                                                                                                                    | 作用                                                             |
|-------|------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| GI 阶段 | [`GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl) | 建立体积所需的深度层                                                     |
| 空气体积  | [`EpipolarScatteringAir`](../../../shaders/pass/composite/EpipolarScatteringAir.comp.glsl)                                                                 | GI 降噪后读取 endpoints、epipolar data、LUT、shadow 和深度层，生成空气散射        |
| 水体/透明 | [`LocalComposite.glsl`](../../../shaders/techniques/atmospherics/LocalComposite.glsl) 等共享代码                                                                | 合并 water scattering、透明 back composite/SST/composite 的天空、云和局部介质 |
| 可选修正  | [`VolumetricLocalCompositeBreakFix`](../../../shaders/pass/composite/VolumetricLocalCompositeBreakFix.comp.glsl)                                           | 在最后修正深度断裂                                                      |

## 天空与天体渲染

项目属性片段会关闭原版 clouds、sun、moon、sky 和 stars。Alpha Piscium 通过自己的 sky composite 和 [
`Celestial.glsl`](../../../shaders/util/Celestial.glsl) 渲染它们：仅在 star intensity 非零时绑定 `usam_starmap`，仅在对应
setting 启用时绑定 `usam_constellations`。

## 设置

| 类别      | 设置                                                                              |
|---------|---------------------------------------------------------------------------------|
| 大气比例与地面 | 高度、密度比例和地面反照率                                                                   |
| 空气      | epipolar slices/samples；Mie turbidity/time curve；Mie/Rayleigh/ozone multipliers |
| 天空与光柱   | sky-view 分辨率、sky samples、shaft samples/shadow samples、深度断裂修正和柔和度                |
| 低云      | 上采样比例；历史长度/置信度/方差；最小/最大步数；MS 网格分辨率/迭代数及局部采样数/散射深度；高度/厚度/密度/覆盖率/相函数；风和形状频率 |
| 高云      | cirrus 高度、密度、覆盖率和相函数                                                            |
| 天体      | sun/moon 半径、距离、温度/颜色/反照率；star-map 强度/gamma/bright-star boost；以及星座               |

所有设置均在 [`scripts/options.main.kts`](../../../scripts/options.main.kts) 中声明；profile 主要缩放 epipolar
分辨率、云上采样、历史长度和 march steps。

## 维护约束

- 修改 LUT 坐标约定时，同时修改所有 API consumer。
- epipolar count 必须保持在声明的尺寸映射范围内。
- cloud history 的渲染分辨率、上采样比例、jitter 和置信度逻辑必须一致。
- 修改二进制资产时，同时修改 `customTexture.*` 声明和 sampler。
- 实测完整昼夜循环、天气、水体切换、快速转向、云层上下两侧和 history reset。
