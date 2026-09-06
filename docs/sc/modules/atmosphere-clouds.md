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
| 1  | [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | 步进对流云塔，结合高分辨率单次散射、缓存 MS 预测和稀疏局部输运残差 |
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

每个非空采样积分便宜的缓存多次散射预测和前景空气。昂贵的单次散射在不透明度加权的 reservoir 深度处求值，
使用完全相同的光谱相机权重。默认使用两个直射光照采样，局部采样预算为四时使用四个。
太阳和月亮分别使用实际密度、大气透射率、地球遮挡及采样日月圆盘方向上的完整相函数。
每个选中深度的天空/地面单次散射使用两个连续相函数重要性方向，PDF 为三个 AP0 相函数密度的等权混合。
精细阴影光线在前八个上界光学深度内使用比率跟踪，远端尾段使用 16 点指数光柱。
这能解析近场遮挡物，避免对稀疏密度积分直接取指数；远端求积仍有近似误差。
近场透射率低于 `1e-6` 时使用带存活补偿的俄罗斯轮盘，避免继续追踪极小的不透明尾段。
这会改变方差，但不会丢弃其期望贡献。

积云环境方向缓存将晴空辐亮度与地面反射分开。在精细散射点，
[`Transport.glsl`](../../../shaders/techniques/atmospherics/clouds/Transport.glsl) 将入射光线与地球求交，
在实际交点计算日月反射，包括地面到光源之间的真实云密度光柱及返回云点的大气透射率，
替代原来的均匀、未受云影遮挡的地面直射贡献。粗边界在相机水平位置和云层中间高度计算此项。
地面漫射照明仍采用晴空近似，但用余弦加权的 `E_sky / pi` 替代未加权半球平均的两倍；
尚未求解云与地面的多次散射耦合。

阴影光柱的指数区间以最小消光平均自由程的四分之一为距离尺度，远处的云层出口不会再将首段阴影采样
拉伸到跨过小表面结构。光柱最多覆盖 12 千米，达到不透明光学厚度（最小通道深度 12）后停止。
远处阴影求积仍然较粗。旧的 extinction octave、逐阶衰减权重及 `0.15` 环境光学深度缩放继续保持移除。
相机与局部光路始终使用真实消光，不使用输运消光或 MS 亮度缩放。

空气散射和透射率逐段积分到非空云采样位置，每段使用 4–32 个采样，再以对应距离的云不透明度加权晴空辐亮度。
[`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl) 将结果合成到天空之上，
不再把积云放置在固定的云层中间高度平面。从卷云上方观察时，卷云的前景散射和透射率会作用于积云贡献。
云历史按不透明度加权深度重投影，并计入水平风平流和大气高度比例。
深度差异会降低历史置信度；世界时间变化使用已有光照历史重置，风计时器回绕时拒绝历史。
单个深度仍近似多个重叠云层，竖直细节运动未参与重投影。风同时平移密度场及其细节，截图模式将其冻结。
覆盖率、密度或云层厚度为零时输出空云层。卷云保留独立的照明模型和多重散射半径控制。

### 共享方向输运与双通道校正

[`Transport.glsl`](../../../shaders/techniques/atmospherics/clouds/Transport.glsl) 将云内散射辐亮度
`S = L - B` 存储在 24 个等立体角方向单元中，方向为 `(2,1,1)/sqrt(6)` 的带符号排列。
目标方程是 `A S = sigma_s P B`，使用有限体积迎风输运，入射散射光边界为零。
空单元的消光严格为零，保留云间自由传播。相机缓存仅存储 `P S`，采样后仍需由实际高分辨率密度散射；
不会再次叠加 `S` 或未碰撞天空光。四方向正值插值保持常量光场，立方对称性保证各基函数积分权重相等；
空间插值使用实际单元中心。

| 顺序 | Pass / 阶段 | 作用 |
|---|---|---|
| 1，begin | [`CloudTransportPrepare`](../../../shaders/pass/begin/CloudTransportPrepare.comp.glsl) | 构造以相机为中心、水平拉伸的网格边界、相函数矩阵和积分后的太阳/月亮注入；每单元四个密度采样保留采样点密度与高分辨率直射透射率的相关性，再构造系数层级 |
| 2，begin | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl)、[`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 将未碰撞天空/地面边界光扫过网格，构造其散射源，并将 `S` 初始化为零 |
| 3，deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl) | 通过三角波前精确求解所选迎风算子，执行一次方向源迭代 |
| 4，deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 计算真实方向残差，将标量部分限制到两个耦合扩散通道；多重网格 V-cycle 使用体积积分残差、4³ 块预条件 CG、实际中心延拓及成对红黑平滑 |
| 5，deferred | [`CloudTransportSweep`](../../../shaders/pass/begin/CloudTransportSweep.comp.glsl)、[`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 通过方向输运提升标量校正，在保持辐亮度非负的条件下沿校正方向最小化真实残差，再更新 `S`；按设置预算重复阶段 3–5 |
| 6，deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 记录最终残差及注入/逃逸/吸收功率，随后预卷积 `P S` 供相机查询 |

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
| SSBO binding 6 | FP32 AP0 方向光场、源项和校正，各层耦合系数/误差/RHS、归约与网格边界；32³ 为 56,240,912 字节，64³ 为 449,858,832 字节；每帧根据当前密度和光照重新求解 |
| `persistent_cloudsAmbLUT` | 原有 32×33 atlas：卷云保持 32×32；最后一行 x=0 保存余弦加权天空辐照度除以 π，x=1…24 保存各方向单元平均晴空辐亮度 |
| SSBO 6 内的 `ctStats` | 四轮校正前/后残差及接受步长、最终相对残差、注入与逃逸加吸收功率、粗层 CG 残差及迭代数；通过 buffer dump 检查 |

SSBO 使用单个不定长 `vec4[]` 和显式偏移；大型定长 SSBO 数组会使 NVIDIA 编译耗时异常增长。
改变此打包布局时，必须同步修改字节分配大小和 buffer 回读程序。

网格水平覆盖 ±121 千米。每条水平轴将 75% 的单元均匀分配到 ±12 千米范围，其余距离采用一阶导数连续的指数外缘。
采样逆映射和实际中心插值使用同一套分配。竖直范围覆盖球面云层并延伸到名义云底以下 2.5 千米。
`SETTING_CLOUDS_MS_RESOLUTION` 选择 32³（默认）或 64³，内部横向间距分别为 1 千米和 500 米；
`SETTING_CLOUDS_MS_ITERATIONS` 选择 1、2 或 4（默认）。
每帧重新求解光场；时域云累积只位于相机积分之后，因此风、相机或光照变化不会复用旧输运解。

此实现尚未证明云图像的角向或空间收敛。24 方向会柔化低阶方向细节，粗迎风单元会引入数值扩散
（厚单元的扩散极限不同于物理 `D0`），每单元四个源项采样也可能遗漏狭窄受光区域。
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
功率平衡、薄密度缩放、真空间隙传播、叠加线性和实际迎风扩散极限。
其 `--gpu-buffer PATH --resolution 32` 模式检查 Iris SSBO 回读，并报告实际有限预算残差与功率缺口。
这些检查补充目标 Iris 编译、buffer 捕获、图像和 pass 耗时，不能替代与参考输运或照片的图像对照。

### 局部散射路径与稀疏残差

[`LocalTransport.glsl`](../../../shaders/techniques/atmospherics/clouds/LocalTransport.glsl) 在查询远场缓存之前，
先穿过真实密度追踪多个散射事件，替代一次冻结源项更新及固定指数局部区间。
粗网格双通道求解器及其 24 方向输运场保留原空间算子；局部光路使用连续 Mie/HG+Draine 方向及连续直射相函数求值。

| [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) 内部阶段 | 操作 |
|---|---|
| 相机步进 | 积分缓存 `P S_c`；按 RGB 散射权重绝对值的最大分量更新独立深度 reservoir，包含前景空气和不透明尾段；保存选中深度处的准确相机预测值 |
| 方向选择 | 混合 RGB 相函数提议与可见太阳/月亮相函数提议，使用完整 PDF；光源概率从 1/16 开始，后续事件按 `1 / (16 * event)` 递减，限制路径权重方差 |
| 局部自由飞行 | 使用光谱空碰撞跟踪穿过真实密度，飞行距离为一个上游单元尺度 `1 / sum(abs(omega) / cell_size)`；地表截断光路，逃逸时查询端点处正值插值的散射辐亮度 `S_c` |
| 散射事件 | 累积缓存中的未碰撞源项预测，并更新两个独立源项 reservoir；继续采样连续方向，达到深度预算时，在最后一次相函数卷积之后用方向性 `S_c` 闭合 |
| 源项重建 | 在两个选中的路径事件处，穿过精细密度重新计算日月及天空/地面光，减去源项预测，并应用事件 PDF 权重 |
| 修正 | 将相函数加权局部估计减去保存的相机预测值，再施加深度 PDF 修正；保留有符号残差和原相机透射率 |

路径跟踪上界为 `1.5 * max(sigma_t) * CU_DENSITY_MAX`，当前密度函数满足
`CU_DENSITY_MAX = 1.35 * CU_DENSITY_SCALE` 的上界。令 `sigma_ref = mean(sigma_t)`，对 AP0 通道取平均。
真实事件概率为 `sigma_ref * density / majorant`，事件权重为 `sigma_s / sigma_ref`；空事件权重为
`(majorant - density * sigma_t) / (majorant - density * sigma_ref)`。
提高提议上界使空事件权重保持正值，避免紧上界配合最小消光真实事件产生稀少但很大的光谱路径权重。
这只改变采样方差，不改变消光或吸收。改变密度最大值时必须同步此上界。
空区域保持自由传播；算法包含散射回填并保持无吸收各向同性平衡光场。
算法采用[空碰撞输运](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/Volume_Scattering_Integrators)。

`SETTING_CLOUDS_MS_LOCAL_SAMPLES` 选择每条相机光线 0、1、2（默认）或 4 条局部路径；
零关闭 MS 修正，仍保留采样直射光照。`SETTING_CLOUDS_MS_LOCAL_BOUNCES` 选择 1、4、8（默认）、16 或 32 个散射事件，
替代已移除的局部区间步数设置。不增加屏幕图像、体积分配或 pass；粗网格 SSBO 仍使用无尺寸 `vec4[]`，
没有引入大型定长数组。

每条采样局部路径使用 `ctSky` 和密度相关光束缓存预测未碰撞源项。事件提议为
`max(throughput * (prediction + (E_sun + E_moon) / (4*pi)))`。
正的各向同性项使粗网格遗漏的单元仍可被选中；它只改变提议概率，不会作为光照加入结果。
两个独立 reservoir 修正源项预测之和，因此精细阴影工作量不再乘以整个散射深度。
给定已经采样的路径，整个源项预测在期望中抵消；提前飞行逃逸时也同样成立。

修正减去整个保存的相机预测值，而不是某个采样离散方向的缓存值。因此，粗角向矩阵无需与连续局部相函数一致：
在相同深度权重下，原预测器在期望中抵消。单次散射和有符号 MS 修正都使用这些权重，
不会重复添加缓存 `S_c` 或未碰撞光。额外局部事件解析小结构之间的散射，而不只是衰减平滑缓存值。

空碰撞跟踪消除了固定步长局部密度求积，但有限飞行半径、终点缓存、相函数 LUT、阴影光柱和相机求积仍留下近似误差。
局部路径不将净通量反馈到粗网格；更多事件减少过早替代为缓存的情况，并不证明完整输运收敛。
稀疏光路仍有噪声，尖锐前向散射尤其明显；有符号 FP16 范围饱和、空间重建和时间方差裁剪仍然有偏。
保留已有时间滤波，没有加入额外大范围模糊；默认历史长度为 32 帧，用于稳定稀疏修正。
历史是否有效仍由重投影和解除遮挡检查控制。

[`cloud-local-transport-check.py`](../../../scripts/test/cloud-local-transport-check.py) 用非均匀平板解析式检验光谱空碰撞跟踪，
检查灰色介质的严格平衡与真空透射、粗细角向算子不同时保存预测值的抵消、光谱 WRS 期望/方差，
以及 RGB 光源混合 PDF 与 HG 解析卷积的一致性。另用二方向平板解析解检查局部散射阶数收敛，
检查光学薄相机段积分，并将相同粗密度的狭窄通道与连续冻结源项 ODE 对照。
额外检查两个路径源项 reservoir、固定散射阶数的光谱权重方差、近场阴影跟踪与高斯光学厚度积分的一致性、朗伯天空归一化，
以及地面直接反射路径两段的衰减。
这些是模型测试，不是完整云图像或全局功率证明。其 `--gpu-history PATH` 模式检查帧末 RGBA32UI atlas 中
有符号散射的有限性、透射率范围、历史长度和深度；帧末只有持久历史 tile 仍含有效云数据。

```sh
python test/cloud-local-transport-check.py --output ../builds/cloud-transport/local-numerics.json
```

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
