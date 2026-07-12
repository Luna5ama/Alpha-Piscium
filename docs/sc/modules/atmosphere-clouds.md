# 大气、天空与云

语言：简体中文 | [English](../../en/modules/atmosphere-clouds.md)

Alpha Piscium 把低频 LUT、屏幕相关的 sky-view/epipolar 数据、体积云和最终局部合成分开更新。大气预计算位于 `begin`，云渲染位于
composite 前段，空气/水体体积在 GI 后、透明合成前完成。

## 代码位置

| 路径                                                                                                                                                                                             | 职责                        |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| [`shaders/techniques/atmospherics/air/lut/`](../../../shaders/techniques/atmospherics/air/lut/)                                                                                                | 大气 LUT 生成与 API            |
| `atmospherics/air/Raymarching*.glsl`                                                                                                                                                           | 空气介质 ray marching         |
| `atmospherics/air/SliceEndPoints.comp.glsl`                                                                                                                                                    | Epipolar 切片端点             |
| `atmospherics/air/EpipolarScattering.comp.glsl`                                                                                                                                                | 空气 epipolar scattering 核心 |
| `atmospherics/clouds/Cumulus.glsl`、[`Cirrus.glsl`](../../../shaders/techniques/atmospherics/clouds/Cirrus.glsl)、[`Mediums.glsl`](../../../shaders/techniques/atmospherics/clouds/Mediums.glsl) | 云密度、介质与相函数                |
| `atmospherics/clouds/RenderVolumetric.comp.glsl`                                                                                                                                               | 低层体积云主渲染                  |
| `atmospherics/clouds/ss/Accum.comp.glsl`                                                                                                                                                       | 低分辨率云的时空累积与上采样            |
| `atmospherics/clouds/amblut/`                                                                                                                                                                  | 云环境光 LUT 的采样与汇聚           |
| [`SkyComposite.glsl`](../../../shaders/techniques/atmospherics/SkyComposite.glsl)、[`LocalComposite.glsl`](../../../shaders/techniques/atmospherics/LocalComposite.glsl)                        | 天空与局部体积合成共享代码             |
| [`shaders/util/Celestial.glsl`](../../../shaders/util/Celestial.glsl)                                                                                                                          | 项目中的太阳、月亮、星图和星座逻辑         |

## Begin 阶段：LUT 与帧准备

当前程序列表中的顺序：

| 顺序 | Pass                                                                                                        | 作用                                      |
|----|-------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| 1  | [`GenerateTransmittance`](../../../shaders/techniques/atmospherics/air/lut/GenerateTransmittance.comp.glsl) | 更新 256×64 `persistent_transmittanceLUT` |
| 2  | [`SliceEndPoints`](../../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl)                   | 更新本帧 epipolar 几何                        |
| 3  | [`GenerateMultiSctr`](../../../shaders/techniques/atmospherics/air/lut/GenerateMultiSctr.comp.glsl)         | 更新 32×32 `persistent_multiSctrLUT`      |
| 4  | [`clouds/amblut/Sample`](../../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl)           | 采样旧/当前云环境信息                             |
| 5  | [`GenerateSkyViewLUT`](../../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl)       | 写入 `uimg_skyViewLUT`                    |
| 6  | [`clouds/amblut/Gather`](../../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl)           | 写入 32×192 `persistent_cloudsAmbLUT`     |

固定尺寸 LUT 在 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 中声明；屏幕尺寸或可配置图像在 [
`scripts/shaders.properties`](../../../scripts/shaders.properties) 中声明。

`uimg_skyViewLUT` 的宽高都使用 `SETTING_SKYVIEW_RES`。`uimg_epipolarData` 的宽使用 `SETTING_EPIPOLAR_SLICES`，高由
`SETTING_SLICE_SAMPLES` 映射到 385、769、1537 或 3073。新增支持的 sample count 时，要同时更新属性尺寸映射和 shader 索引逻辑。

## 云 pass

启用 `SETTING_CLOUDS_CU` 时：

| 顺序 | Pass                                                                                             | 作用                                                     |
|----|--------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| 1  | [`RenderVolumetric`](../../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | 使用环境 LUT、大气 LUT、阴影/光照和项目 cloud noise ray march cumulus |
| 2  | [`clouds/ss/Accum`](../../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl)          | 对低分辨率结果执行历史累积、置信度/方差处理和上采样；该阶段为 `ss/Accum`             |
| 3  | 后续空气/局部合成                                                                                        | 读取累积后的云散射和透射率                                          |

主要 screen tile 为 `transient_lowCloudRender`、`transient_lowCloudAccumulated` 和 `history_lowCloud`
（RGBA32UI）。手写属性片段声明了自定义 cloud phase LUT、cirrus、cumulus base/detail 和 curl 纹理。高层 cirrus 由共享 sky/cloud
路径采样，不使用独立 compute program。

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
| 低云      | 上采样比例；历史长度/置信度/方差；最小/最大步数；高度/厚度/密度/覆盖率/相函数；风和形状频率                               |
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
