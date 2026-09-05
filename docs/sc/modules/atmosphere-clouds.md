# 大气、天空与云

语言：简体中文 | [English](../../en/modules/atmosphere-clouds.md)

Alpha Piscium 把低频 LUT、屏幕相关的 sky-view/epipolar 数据、体积云和最终局部合成分开更新。大气预计算位于 `begin`，云渲染位于
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
| 6  | [`clouds/amblut/Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl)           | 写入 32×33 `persistent_cloudsAmbLUT`：32×32 的卷云方向 tile，以及最后一行的三个积云值（上半球大气、下半球大气、地面直接反射）；归约全部 subgroup，并在状态切换时重置未更新区域的历史 |

固定尺寸 LUT 在 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 中声明；屏幕尺寸或可配置图像在 [
`scripts/shaders.properties`](../../../scripts/shaders.properties) 中声明。

`uimg_skyViewLUT` 的宽高都使用 `SETTING_SKYVIEW_RES`，六个切片分别存储完整大气、卷云下方和上方区域的散射与透射率。
`uimg_epipolarData` 的宽使用 `SETTING_EPIPOLAR_SLICES`，高由
`SETTING_SLICE_SAMPLES` 映射到 385、769、1537 或 3073。新增支持的 sample count 时，要同时更新属性尺寸映射和 shader 索引逻辑。

## 云 pass

启用 `SETTING_CLOUDS_CU` 时：

| 顺序 | Pass                                                                                             | 作用                                                                                                               |
|----|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| 1  | [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | 步进对流云塔，积分太阳光和环境光遮挡，并计算有限阶多重散射近似                                                           |
| 2  | [`clouds/ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)          | 对低分辨率结果执行历史累积、置信度/方差处理和上采样；该阶段为 [`ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl) |
| 3  | 后续空气/局部合成                                                                                        | 读取累积后的云散射和透射率                                                                                                    |

主要 screen tile 为 `transient_lowCloudRender`、`transient_lowCloudAccumulated` 和 `history_lowCloud`
（RGBA32UI）。手写属性片段声明了自定义 cloud phase LUT、cirrus、cumulus detail 和 curl 纹理。高层 cirrus 由共享 sky/cloud
路径采样，不使用独立 compute program。

### 浓积云

[`Cumulus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cumulus.glsl) 用三个高度不同、相互重叠的竖直热泡
构造每个单元，在共同的凝结高度处截断。单元半径随高度缩放，倾斜和侧向分支随单元变化。
经过 curl 扭曲的三维蓬松噪声沿竖直方向拉伸，两级更细的侵蚀塑造表面。侵蚀幅度相对于单元半径，
使较小单元也保留连贯的内部体积。液态水密度在狭窄的凝结过渡区上方逐渐增加。
默认云底高度为 1 千米，云层厚度为 3.5 千米。

[`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) 使用
[`clouds/Common.glsl`](../../../shaders/techniques/atmospherics/clouds/Common.glsl) 中的球壳求交。
主光线以非均匀距离区间覆盖整个可见范围，每个区间内按距离均匀抖动一个采样。积分使用区间实际长度及均匀介质的
Beer-Lambert 解析积分。地表交点截断光线；云内行进距离上限为 60 千米，距相机的最远距离为 120 千米。

每个非空采样积分包含 16 个采样的太阳光柱，以及分别包含 6 个和 4 个采样的天空、地面方向光柱。
三者使用相同密度场和指数距离区间，在表面附近分配更多采样。下半球均匀方向采样地面覆盖区域，
再用包含 10 个采样的云光柱遮挡其直接反射光。下半球大气贡献保持独立，地面的漫反射天光使用配置的地面反照率和上半球积分。
太阳光从设置指定的积云相函数开始，再叠加四阶散射近似：各阶振幅为
`0.6^n`，光学深度缩放为 `0.45^n`，Henyey-Greenstein 相函数的方向性逐阶减弱。
环境光可见性使用两个垂直光柱，输运光学深度缩放为 `0.15`。这些有限阶近似提供内部补光和云团自阴影，
并非完整的多重散射解。

空气散射和透射率逐段积分到非空云采样位置，每段使用 4–32 个采样，再以对应距离的云不透明度加权晴空辐亮度。
[`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl) 将结果合成到天空之上，
不再把积云放置在固定的云层中间高度平面。从卷云上方观察时，卷云的前景散射和透射率会作用于积云贡献。
云历史打包和累积继续作为集成接口。风同时平移密度场及其细节，截图模式将其冻结。
覆盖率、密度或云层厚度为零时输出空云层。卷云保留独立的照明模型和多重散射半径控制。

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
| 低云      | 上采样比例；历史长度/置信度/方差；最小/最大步数；高度/厚度/密度/覆盖率/相函数；风和形状频率                 |
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
