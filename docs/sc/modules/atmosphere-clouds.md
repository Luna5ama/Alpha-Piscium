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
| 4  | [`clouds/amblut/Sample`](../../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl)           | 交替使用积云和卷云高度，采样 1024 条全空间大气光线，并单独存储地面的直接反射光 |
| 5  | [`GenerateSkyViewLUT`](../../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl)       | 写入 `uimg_skyViewLUT`                    |
| 6  | [`clouds/amblut/Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl)           | 写入 32×33 `persistent_cloudsAmbLUT`：32×32 的卷云方向 tile，以及最后一行的25 个积云值（上半球天空积分和 24 个入射方向）；归约全部 subgroup，并在状态切换时重置未更新区域的历史 |

固定尺寸 LUT 在 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 中声明；屏幕尺寸或可配置图像在 [
`scripts/shaders.properties`](../../../scripts/shaders.properties) 中声明。

`uimg_skyViewLUT` 的宽高都使用 `SETTING_SKYVIEW_RES`，六个切片分别存储完整大气、卷云下方和上方区域的散射与透射率。
`uimg_epipolarData` 的宽使用 `SETTING_EPIPOLAR_SLICES`，高由
`SETTING_SLICE_SAMPLES` 映射到 385、769、1537 或 3073。新增支持的 sample count 时，要同时更新属性尺寸映射和 shader 索引逻辑。

## 云 pass

启用 `SETTING_CLOUDS_CU` 时：

| 顺序 | Pass                                                                                             | 作用                                                                                                               |
|----|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| 1  | [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | 步进对流云塔，结合高分辨率单次散射和共享多重散射光场                                                           |
| 2  | [`clouds/ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)          | 对低分辨率结果执行历史累积、置信度/方差处理和上采样；该阶段为 [`ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl) |
| 3  | 后续空气/局部合成                                                                                        | 读取累积后的云散射和透射率                                                                                                    |

主要 screen tile 为 `transient_lowCloudRender`、`transient_lowCloudAccumulated` 和 `history_lowCloud`
（RGBA32UI）。手写属性片段声明了自定义 cloud phase LUT、cirrus、cumulus detail 和 curl 纹理。高层 cirrus 由共享 sky/cloud
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
主光线以非均匀距离区间覆盖整个可见范围，默认使用 128–256 个采样。空采样之后遇到非空采样时，
通过四次二分定位边界，并将照明求值位置移到距边界一个名义平均自由程以内。
均匀介质的 Beer-Lambert 解析积分使用被云占据的区间长度。
这种表面采样是有偏近似，用于在有限采样数下保留受光鼓包。地表交点截断光线；
云内行进距离上限为 60 千米，距相机的最远距离为 120 千米。
当剩余透射率低于 `0.005` 时，用局部光源项补全不透明尾段，并将透射率归零，
避免浓云内部出现晴空漏光和提前终止造成的色带。

每个非空采样把单次散射与缓存的散射光场分开计算。太阳和月亮分别使用实际密度上的 16 点光柱、
大气透射率、地球遮挡及完整的配置相函数。天空/地面单次散射采用两个相函数重要性采样方向，
各积分 12 点密度光柱；PDF 是三个 AP0 相函数密度的等权混合，背景环境来自方向环境光 LUT。
旧的 extinction octave、逐阶衰减权重及 `0.15` 环境光学深度缩放已移除。
相机积分始终使用真实消光，不使用输运系数或拟合的 MS 缩放。

空气散射和透射率逐段积分到非空云采样位置，每段使用 4–32 个采样，再以对应距离的云不透明度加权晴空辐亮度。
[`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl) 将结果合成到天空之上，
不再把积云放置在固定的云层中间高度平面。从卷云上方观察时，卷云的前景散射和透射率会作用于积云贡献。
云历史打包和累积继续作为集成接口。风同时平移密度场及其细节，截图模式将其冻结。
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
| 4，deferred | [`CloudTransportCorrect`](../../../shaders/pass/begin/CloudTransportCorrect.comp.glsl) | 计算真实方向残差，将标量部分限制到两个耦合扩散通道；多重网格 V-cycle 使用体积积分残差、4³（32³/64³ 网格）或 6³（96³ 网格）块预条件 CG、实际中心延拓及成对红黑平滑 |
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
| SSBO binding 6 | FP32 AP0 方向光场、源项和校正，各层耦合系数/误差/RHS、归约与网格边界；32³ 为 56,240,912 字节，64³ 为 449,858,832 字节，96³ 为 1,518,246,672 字节；每帧根据当前密度和光照重新求解 |
| `persistent_cloudsAmbLUT` | 原有 32×33 atlas：卷云保持 32×32；最后一行 x=0 保存上半球天空积分除以 4π，x=1…24 保存各方向单元平均入射辐亮度 |
| SSBO 6 内的 `ctStats` | 四轮校正前/后残差及接受步长、最终相对残差、注入与逃逸加吸收功率、粗层 CG 残差及迭代数；通过 buffer dump 检查 |

SSBO 使用单个不定长 `vec4[]` 和显式偏移；大型定长 SSBO 数组会使 NVIDIA 编译耗时异常增长。
改变此打包布局时，必须同步修改字节分配大小和 buffer 回读程序。

网格水平覆盖 ±121 千米。每条水平轴将 75% 的单元均匀分配到 ±12 千米范围，其余距离采用一阶导数连续的指数外缘。
采样逆映射和实际中心插值使用同一套分配。竖直范围覆盖球面云层并延伸到名义云底以下 2.5 千米。
`SETTING_CLOUDS_MS_RESOLUTION` 选择 32³、64³ 或 96³（默认），内部横向间距分别为 1 千米、500 米和 333 米；
`SETTING_CLOUDS_MS_ITERATIONS` 选择 1、2 或 4（默认）。
每帧重新求解光场；时域云累积只位于相机积分之后，因此风、相机或光照变化不会复用旧输运解。

此实现尚未证明云图像的角向或空间收敛。24 方向会柔化低阶方向细节，粗迎风单元会引入数值扩散
（厚单元的扩散极限不同于物理 `D0`），每单元四个源项采样也可能遗漏狭窄受光区域。
有限网格使用空间均匀的天空/地面环境，未包含云影反馈到地面反射的耦合。
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
| 低云      | 上采样比例；历史长度/置信度/方差；最小/最大步数；MS 网格分辨率/迭代数；高度/厚度/密度/覆盖率/相函数；风和形状频率                 |
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
