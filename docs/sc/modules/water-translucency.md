# 水、焦散与半透明

语言：简体中文 | [English](../../en/modules/water-translucency.md)

Alpha Piscium 将透明几何的 raster、反射/折射 trace、水体散射和最终 composite 分开。这样水/玻璃可以使用 solid G-buffer
和已经完成的光照，同时保持多层深度与体积结果。

## 代码位置

| 路径                                                                                                                                                                                                                                                      | 职责                             |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| `shaders/pass/geometry/GBufferTranslucent.*.glsl`                                                                                                                                                                                                       | 水、透明表面、weather/particle raster |
| `gbuffers_water.*`、`gbuffers_hand_water.*`、`dh_water.*`                                                                                                                                                                                                 | 水路径 wrapper                    |
| [`shaders/techniques/WaterWave.glsl`](../../../shaders/techniques/WaterWave.glsl)                                                                                                                                                                       | 水波法线/位移共享计算                    |
| [`shaders/util/Translucent.glsl`](../../../shaders/util/Translucent.glsl)                                                                                                                                                                               | 半透明数据和 composite 共享逻辑          |
| [`CausticsPhotonTrace`](../../../shaders/pass/composite/CausticsPhotonTrace.comp.glsl)、[`CausticsRemap`](../../../shaders/pass/composite/CausticsRemap.comp.glsl)                                                                                       | 屏幕空间焦散生成                       |
| [`EpipolarScatteringWater`](../../../shaders/pass/composite/EpipolarScatteringWater.comp.glsl)                                                                                                                                                          | 水体 epipolar scattering         |
| [`TranslucentBackComposite`](../../../shaders/pass/composite/TranslucentBackComposite.glsl)、[`TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl)、[`TranslucentComposite`](../../../shaders/pass/composite/TranslucentComposite.glsl) | 背层合成、透明 trace 和最终合成            |
| `techniques/atmospherics/water/`                                                                                                                                                                                                                        | 水下散射常量与 epipolar 实现            |

## 几何与资源

透明 geometry 将 packed G-buffer 写入 `colortex11/12`、将 transmittance 写入 `colortex14`、将 CSR R32F atlas 的 near/far
depth 写入对应深度资源，而不是立即替换 solid lighting。[
`TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl) 随后写入供最终合成读取的 reflection/refraction
tile；tile 的格式和生命周期由 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 管理：

- `transient_translucentReflection` 和 `transient_translucentRefraction`。
- `transient_translucentZLayer1/2/3`。
- `transient_lmCoord` 和 `transient_shadow`。
- `transient_caustics_input`、`transient_caustics_final` 和 `transient_screenPixelSize`。

水波使用 [`scripts/shaders.properties`](../../../scripts/shaders.properties) 声明的自定义 `usam_waveNoise`（2048×2048
R16）和 `usam_waveHFCurl`（512×512 RG16_SNORM）纹理。water shadow variant 与 [
`EvaluateShadowWaterNormal`](../../../shaders/pass/shadowcomp/EvaluateShadowWaterNormal.glsl) 共同提供后续 water-shadow
normal。

## 焦散

启用 `SETTING_WATER_CAUSTICS` 时：

| 顺序 | Pass                                                                                           | 作用                                                               |
|----|------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| 1  | [`EvaluateScreenPixelSize`](../../../shaders/pass/composite/EvaluateScreenPixelSize.comp.glsl) | 估计 trace footprint                                               |
| 2  | [`CausticsPhotonTrace`](../../../shaders/pass/composite/CausticsPhotonTrace.comp.glsl)         | 从水面和光照条件生成 photon contribution                                   |
| 3  | [`CausticsRemap`](../../../shaders/pass/composite/CausticsRemap.comp.glsl)                     | 将输入重映射到 `transient_caustics_final`，供后续 lighting 和 volume pass 使用 |

关闭该设置时，三个 program 会作为一组同时禁用。

## 体积与最终合成

GI 降噪后按以下顺序执行：

| 顺序 | Pass                                                                                           | 作用                     |
|----|------------------------------------------------------------------------------------------------|------------------------|
| 1  | [`EpipolarScatteringAir`](../../../shaders/pass/composite/EpipolarScatteringAir.comp.glsl)     | 完成空气体积                 |
| 2  | [`EpipolarScatteringWater`](../../../shaders/pass/composite/EpipolarScatteringWater.comp.glsl) | 计算水下和穿水路径              |
| 3  | [`TranslucentBackComposite`](../../../shaders/pass/composite/TranslucentBackComposite.glsl)    | 捕获透明表面背后的已着色场景和体积      |
| 4  | [`TranslucentSST`](../../../shaders/pass/composite/TranslucentSST.glsl)                        | 执行屏幕空间反射/折射 trace      |
| 5  | [`TranslucentComposite`](../../../shaders/pass/composite/TranslucentComposite.glsl)            | 使用多层 Z、材质吸收、反射和折射更新主颜色 |

## 设置

| 类别                  | 设置                                                                                            |
|---------------------|-----------------------------------------------------------------------------------------------|
| 表面                  | water roughness、wave frequency/speed 和 normal scale                                           |
| Parallax            | enable、strength、linear/secant steps                                                           |
| Refraction/caustics | refraction approximation 和 `SETTING_WATER_CAUSTICS`                                           |
| 介质                  | water scattering RGB/multiplier、absorption RGB/multiplier 和 refraction-approximation contrast |
| Shadow/volume       | light-shaft softness、water-shadow samples 和 sample-pool size                                  |
| 通用透明材质              | roughness reduction/min/max 和 absorption saturation/gamma/alpha curve/multiplier              |

这些设置在 [`scripts/options.main.kts`](../../../scripts/options.main.kts) 的 Terrain 和 Atmospherics water screen 中声明。

## 维护约束

- 多层 Z 的排序、clear 值和读取必须一起修改。
- reflection/refraction tile 的 alpha 和 packing 语义必须在 geometry 与 composition 间保持一致。
- 水体 absorption 和 scattering 的路径长度单位必须一致，不要用 clamp 隐藏错误。
- 焦散 program 是条件组；其启用/禁用必须作为一个 feature 保持一致。
- 验证水上/水下、看向水面两侧、手持水、DH water、玻璃重叠和相机穿越介质边界。
