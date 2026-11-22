# Atmosphere, Sky, and Clouds

Alpha Piscium separates low-frequency LUTs, view-dependent sky-view/epipolar data, volumetric clouds, and final local
composition. Atmosphere preparation runs in `begin` and clouds render early in composite. Air/water resolve after
GI history fix, alongside spatial blur passes when enabled, and finish before translucency.

## Code map

| Path | Responsibility |
| --- | --- |
| [`shaders/techniques/atmospherics/air/lut/`](../../shaders/techniques/atmospherics/air/lut/) | Atmosphere LUT generation and API |
| [`atmospherics/air/Raymarching*.glsl`](../../shaders/techniques/atmospherics/air/) | Air-medium ray marching |
| [`SliceEndPoints.comp.glsl`](../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl) | Epipolar slice endpoints |
| [`EpipolarScattering.comp.glsl`](../../shaders/techniques/atmospherics/air/EpipolarScattering.comp.glsl) | Air epipolar-scattering kernel |
| [`Cumulus.glsl`](../../shaders/techniques/atmospherics/clouds/Cumulus.glsl), [`Cirrus.glsl`](../../shaders/techniques/atmospherics/clouds/Cirrus.glsl), [`Mediums.glsl`](../../shaders/techniques/atmospherics/clouds/Mediums.glsl) | Cloud density, media, and phase functions |
| [`RenderVolumetric.comp.glsl`](../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | Main low-cloud render |
| [`Accum.comp.glsl`](../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl) | Temporal/spatial cloud accumulation and upscale |
| [`atmospherics/clouds/amblut/`](../../shaders/techniques/atmospherics/clouds/amblut/) | Cloud ambient-light LUT sampling and gathering |
| [`SkyComposite.glsl`](../../shaders/techniques/atmospherics/SkyComposite.glsl), [`LocalComposite.glsl`](../../shaders/techniques/atmospherics/LocalComposite.glsl) | Shared sky and local-volume composition |
| [`shaders/util/Celestial.glsl`](../../shaders/util/Celestial.glsl) | Project sun, moon, star-map, and constellation logic |

## Begin: LUT and frame preparation

The LUT update sequence follows these dependencies.

| Order | Pass | Purpose |
| --- | --- | --- |
| 1 | [`GenerateTransmittance`](../../shaders/techniques/atmospherics/air/lut/GenerateTransmittance.comp.glsl) | Updates the 256×64 `persistent_transmittanceLUT` |
| 2 | [`SliceEndPoints`](../../shaders/techniques/atmospherics/air/SliceEndPoints.comp.glsl) | Updates epipolar geometry for the frame |
| 3 | [`GenerateMultiSctr`](../../shaders/techniques/atmospherics/air/lut/GenerateMultiSctr.comp.glsl) | Updates the 32×32 `persistent_multiSctrLUT` |
| 4 | [`clouds/amblut/Sample`](../../shaders/techniques/atmospherics/clouds/amblut/Sample.comp.glsl) | Samples previous/current cloud ambient information |
| 5 | [`GenerateSkyViewLUT`](../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl) | Writes `uimg_skyViewLUT` |
| 6 | [`clouds/amblut/Gather`](../../shaders/techniques/atmospherics/clouds/amblut/Gather.comp.glsl) | Writes the 32×192 `persistent_cloudsAmbLUT` |

Fixed LUT dimensions are declared in [`shaders/shadesmith.json`](../../shaders/shadesmith.json);
screen-sized/configurable images are declared in [`scripts/shaders.properties`](../../scripts/shaders.properties).

Both `uimg_skyViewLUT` dimensions use `SETTING_SKYVIEW_RES`. The fixed `persistent_epipolarData` tile is 2048×3073.
`SETTING_EPIPOLAR_SLICES` selects the active width, and `SETTING_SLICE_SAMPLES` selects a logical height of
385, 769, 1537, or 3073 for three layers plus the endpoints row. When adding a sample count,
check the tile capacity, dispatch, and indexing in [the epipolar code](../../shaders/techniques/atmospherics/air/).

## Cloud passes

The [first procedural cloud-modeling revision](#首版建模交付) defines the geometry and its integration into the current renderer.
The [cloud-lighting experiment](#云光照的物理对照) uses the same density for its real-time candidate and transport reference.

When `SETTING_CLOUDS_CU` is enabled:

| Order | Pass | Purpose |
| --- | --- | --- |
| 1 | [`RenderVolumetric`](../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) | Ray marches cumulus using the ambient LUT, atmosphere LUTs, shadows/lights, and project cloud noise |
| 2 | [`clouds/ss/Accum`](../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl) | Applies history accumulation, confidence/variance processing, and upscaling to the low-resolution result; this is the [`ss/Accum`](../../shaders/techniques/atmospherics/clouds/ss/Accum.comp.glsl) stage |
| 3 | Later air/local composition | Reads the accumulated cloud scattering and transmittance |

The main screen tiles are `transient_lowCloudRender`, `transient_lowCloudAccumulated`, and `history_lowCloud` (
RGBA32UI). The hand-maintained property fragment declares custom cloud phase-LUT, cirrus, cumulus base/detail, and curl
textures. High cirrus is sampled by the shared sky/cloud path rather than a separate compute program.

### Cumulus isotropic multiple scattering

The cumulus renderer reuses the existing ordered eight-sample sun-light column. For each source sample jittered uniformly in distance within its quadratic-distance bin,
let `U_i` be the prefix optical depth from the start of the light column to that source position, `sigma_s` the scattering coefficient,
`sigma_tr` the transport coefficient, `ds` the sample length, and `r` the source radius. This diffusion approximation
uses `sigma_tr ≈ sigma_t` in the implementation. With dimensionless absorption fraction (albedo deficit) `a = 0.001`
multiplying optical depth, per-channel asymmetry `g`, and `k = sqrt(3a)`, the direct upstream-prefix estimator is

$$
W_i=\frac{(\sigma_s\,ds)\sigma_{tr}}{r},\qquad
\Phi=\sum_{i=1}^{8}W_i e^{-aU_i}
     \left(1-e^{-(1-g)U_i}\right)e^{-kU_i}.
$$

The implementation clamps the build rate `1 - g` to a nonnegative value so transformed color spaces cannot turn
scattering buildup into amplification.

Intensity is applied before a fixed soft compression:

$$
\Phi_{\mathrm{mapped}}=1-e^{-\max(\mathrm{intensity}\,\Phi,0)}.
$$

This bounds each mapped channel below `1`. `SETTING_CLOUDS_CU_ISOTROPIC_MS_INTENSITY` supplies `intensity`; `0`
disables the contribution, the default `1.0` is the current artistic gain, and `0.25` approximately matches the
omitted `3/(4π)` normalization reference. The result is added independently of the existing WDT22
multiple-scattering term; it does not replace or modify that term.

The accumulated `phi_fwd` field is isotropic and follows the prefix estimator, but final view-path readout intentionally
uses `msPhase = mix(UNIFORM_PHASE, layerParam.medium.phase, 0.7)` to retain controlled directional structure. This
rendering choice is layered after the isotropic field, not part of its transport recurrence.

The receiver-local boundary weight is

$$
H(x,z)=\mathrm{thickness}\;\mathrm{saturate}(\mathrm{baseCoverage}_{raw}(x,z)),\qquad
\Delta=\mathrm{clamp}(0.05/\_LOW\_BASE\_FREQ,0.025,0.2),
$$

$$
\partial_xH=\frac{H(x+\Delta,z)-H(x-\Delta,z)}{2\Delta},\qquad
\partial_zH=\frac{H(x,z+\Delta)-H(x,z-\Delta)}{2\Delta},\qquad
N=\mathrm{normalize}(-\partial_xH,1,-\partial_zH),
$$

$$
C_{top}^{raw}=\mathrm{saturate}\!\left(\frac{N\cdot\mathrm{renderParams.lightDir}+0.5}{1.5}\right),\qquad
C_{top}=\mathrm{mix}(1,C_{top}^{raw},0.25),\qquad
C_{bottom}=1-\exp\!\left(-\frac{\max(h_{local},0)}{0.1\,\mathrm{mix}(1,4,h_{column})}\right),\qquad
B_{eff}=C_{top}C_{bottom}.
$$

Here `baseCoverage_raw` is the existing pre-height coverage value, `h_column = saturate(baseCoverage_raw)` reuses the
receiver density lookup, and `h_local` is the normalized receiver height (`0` at the cumulus-layer base and `1` at its
top). The constants correspond to `b = 0`, `p = 1`, and `H_bottom = 0.1`, matching the density model's existing normalized
`0.1` bottom scale. `C_top` uses the actual `renderParams.lightDir`, not the cone-jittered light-march direction. The
`0.25` confidence strength keeps the top factor in `[0.75,1]`; full-strength use of the coarse coverage normal exposes
vertical bands in the cloud body. The gate is evaluated once per occupied receiver sample and multiplies every source
weight before accumulation, intensity, and compression. It costs four extra coverage-proxy evaluations with no extra
center lookup, and adds no pass, resource, texture resource, or density march.

The estimator is adapted from AshenOneArt's [HanPi Volume Cloud implementation](https://github.com/AshenOneArt/HPVolumeCloud/blob/27e799914493de9fa527179312ed72a39d08e225/VolumetricClouds.hlsl)
and [forward-flux derivation](https://github.com/AshenOneArt/HPVolumeCloud/blob/27e799914493de9fa527179312ed72a39d08e225/Docs/PhiFwd_FromRTE.md).

## Air, depth layers, and composition

| Stage | Pass/code | Purpose |
| --- | --- | --- |
| GI phase | [`GIDenoiserEdgeClassificationAndVolumetricsDepthLayers`](../../shaders/pass/composite/GIDenoiserEdgeClassificationAndVolumetricsDepthLayers.comp.glsl) | Builds the depth layers required by volumetrics |
| Air volume | [`EpipolarScatteringAir`](../../shaders/pass/composite/EpipolarScatteringAir.comp.glsl) | Consumes endpoints, epipolar data, LUTs, shadows, and depth layers to produce air scattering |
| Water/translucency | [`LocalComposite.glsl`](../../shaders/techniques/atmospherics/LocalComposite.glsl) | Reconstructs one air or water interval from epipolar samples for the translucent back and final composites |
| Optional correction | [`VolumetricLocalCompositeBreakFix`](../../shaders/pass/composite/VolumetricLocalCompositeBreakFix.comp.glsl) | Corrects depth breaks at the end |

### 天空、雾与水的分段合成

主检出 `9e82f4565` 有两套含义不同的“层”。Sky View 的第 0 组从阴影距离之后积分晴空，供 GI 和部分反射、折射查询。第 1–3 组按积云中高和卷云高度分割空气，主天空在这些组之间插入整段积云和卷云。积云历史只存散射、透射和历史长度，实体像素走不到这条天空合成路径。另一套由水面、其他透明表面和实体深度生成，远到近的三个槽位分别交给空气、水、空气。两套层没有共同的沿射线距离边界。[Sky View 生成](../../shaders/techniques/atmospherics/air/lut/GenerateSkyViewLUT.comp.glsl) · [天空合成](../../shaders/techniques/atmospherics/SkyComposite.glsl) · [透明深度分段](../../shaders/techniques/atmospherics/VolumetricsDepthLayers.glsl)

推荐保留现有 `ScatteringResult`。在给定起点、方向和世界米制区间内，返回段内朝观察者散入的 RGB 辉度 $S$ 和 RGB 透射 $T$。近段和远段串接得到 $S_n+T_n S_f$ 与 $T_nT_f$，最后给背景乘总透射。天空以行星或大气边界为终点，远山以实体深度为终点。日月盘与星空作为边界辐亮度。先将视深转为沿该射线的实长，再把端点变换到大气坐标。现有水平距离和高度缩放可以不同。[现有合成运算](../../shaders/techniques/atmospherics/Common.glsl) · [大气坐标换算](../../shaders/techniques/atmospherics/air/Common.glsl)

**空气和雾。** 主检出的空气包含 Rayleigh、Mie 和臭氧，近处体积光另查主光阴影。在一条无遮挡的天空射线上，高度层可从眼点开始，epipolar 又会计算到阴影距离的近段。两边都输出了这段空气的散射与透射。建议近场由 epipolar 或实际区间积分给出一次，远场从同一个交接点算到地形或天空。`shadowDistance` 只约束地形阴影采样。远处仍有空气透视，云隙光用所在位置到太阳的云透射。若加入局部地雾，则在重叠区间把额外雾的消光和带自身光照的散射源加到空气里，用共同透射积分。同一批气溶胶的密度只计一次。[近场空气](../../shaders/techniques/atmospherics/air/RaymarchScreenViewAtmosphere.glsl) · [EGSR 2020 的天空与空气透视](https://diglib.eg.org/items/8a3e5350-18b3-46bd-9274-3add5af88c75)

**云和空气共处的距离。** 浓云采用标量云消光，保留彩色入射光与云前空气透视。已有实验给出了省去每个细步长都查大气的做法。对单个空气区间，以近端为 0、长度为 $D$，记 $S_a(s),T_a(s)$ 为该段空气和雾累计到 $s$ 的结果。它们可包含云投在空气上的太阳阴影，观察路径上的云消光则由 $T_c(s)$ 单独处理。$Q_c$ 是云散射源除以云消光，在有云的点取值。按 $d\alpha_c=T_c(s)\sigma_{t,c}(s)\,ds$ 加权可得

$$
S_{ac}=T_c(D) S_a(D)+\int_0^D [S_a(s)+T_a(s)Q_c(s)]\,d\alpha_c(s),\qquad T_{ac}=T_a(D)T_c(D).
$$

云细步本来就算出了权重 $T_{c,\mathrm{near}}(1-e^{-\Delta\tau_c})$。可以按它选少量散射深度，再查到这些位置的空气。每个被水面、玻璃或实体截断的区间单独累计。薄云、擦边或隔着净空的两团云应保留分开的深度贡献。举例，近、远云透射分别为 0.2 和 0.8，交换次序后整段云透射都是 0.16。夹在中间的雾若向相机散入 0.4，其可见贡献却是 0.08 或 0.32。单一平均云深可以帮助历史重投影，但不能恢复这个差别。已有的推导、均匀介质校验和 GLSL 取样实现分别见 `c09cfdc76:docs/en/modules/atmosphere-clouds.md`、`c09cfdc76:scripts/test/cloud-local-transport-check.py` 的 `air_cloud_integral` 和该修订的 `RenderVolumetric.comp.glsl`。[体积输运方程](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer)

**如何供应距离查询。** 旧云实验返回给背景的是上式的积分项和 $T_c$，预期背景已经带有该段空气。建议在交给通用合成前就补入 $T_c S_a(D)$，并将该段透射写成 $T_a(D)T_c(D)$，让后面的水面和地形直接消费完整区间。第一版可只给实际端点和少量云深提供空气结果。沿相机射线缓存累计值时，只在同一条直线、同一介质范围且近端透射仍可分辨时恢复子段。透射接近零时，从可见的新起点直接求段，或在前景已遮蔽背景时终止。低分辨率的全球方向查询继续给 GI 和水面提供远方天空。在近云旁的反射或折射要从表面起点沿新方向查询局部段，再接远场天空。每次拼接只接入尚未覆盖的距离。

**水面和玻璃。** 首先求从界面看出去的反射与折射分支，各按自己的方向和介质累积。用现有 Fresnel、表面材质和高光合成，最后应用眼点到界面的前景段一次。岸上看湖时，反射分支走空气和云雾，透进湖底的分支走水。水下向上时，折射出水的分支查询空气与天空，全反射的分支留在水里，两者都经过同一段眼前水体。现有 SST 命中读的是从另一个屏幕像素看过去、已带一部分体积的背景。可共用的后景段只取一次，路径差异明显时应以命中点的表面辐亮度为边界补算界面到命中的介质。水的 RGB 消光、散射与焦散沿用现有求值。最前层的深度断裂修正也应重算原来失败的区间，现行补算使用眼点到实体的空气段。[水体输运](../../shaders/techniques/atmospherics/water/WaterEpipolarScattering.comp.glsl) · [SST](../../shaders/pass/composite/TranslucentSST.glsl) · [最终表面合成](../../shaders/pass/composite/TranslucentComposite.glsl) · [深度断裂修正](../../shaders/pass/composite/VolumetricLocalCompositeBreakFix.comp.glsl)

用户确认首版覆盖一段连续水体，包括从岸上看入水中、穿过水面和从水下抬头看云。沿用现有近、远水深和眼点介质标记。水下起段取眼点，终点取出水界面或先命中的实体。视线上隔着空气的第二段水体留给需要额外出入界面的后续版本。

接入时先用无云天空与远山检查近远交接处，再在两团云之间放一段可见薄雾并移动地形截断深度。水面在同一天空下由上、下两侧观察，连续跨越水线并比较屏内命中和屏外方向。首版所有体积改动沿用同场景整帧 1.30 倍上限。

## Sky and celestial rendering

Vanilla clouds, sun, moon, sky, and stars are disabled in the project property fragment. Alpha Piscium renders them
through its own sky composite and [`Celestial.glsl`](../../shaders/util/Celestial.glsl), conditionally binding
`usam_starmap` only when star intensity is nonzero and `usam_constellations` only when its setting is enabled.

## Settings

| Category | Settings |
| --- | --- |
| Atmosphere scale/ground | Altitude, density scale, and ground albedo |
| Air | Epipolar slices/samples; Mie turbidity/time curve; Mie/Rayleigh/ozone multipliers |
| Sky/light shafts | Sky-view resolution, sky samples, shaft samples/shadow samples, depth-break correction, and softness |
| Low cloud | Upscale factor; history length/confidence/variance; minimum/maximum steps; height/thickness/density/coverage/phase; isotropic multiple-scattering intensity; wind; and shape frequencies |
| High cloud | Cirrus height, density, coverage, and phase |
| Celestial | Sun/moon radius, distance, temperature/color/albedo; star-map intensity/gamma/bright-star boost; and constellations |

All settings are declared in [`scripts/options.main.kts`](../../scripts/options.main.kts); profiles mainly scale
epipolar resolution, cloud upscale, history length, and march steps.

## Maintenance invariants

- Change LUT coordinate conventions and every API consumer together.
- Keep epipolar counts within the declared size mapping.
- Keep cloud-history render resolution, upscale factor, jitter, and confidence logic aligned.
- Change binary assets, `customTexture.*` declarations, and samplers together.
- Test the full day cycle, weather, water transitions, fast turns, both sides of cloud layers, and history resets.

## 程序化云的水平分布、垂直结构与保守边界

实拍编号对应主检出本地的 `agent_inputs/cloud-references/index.html`。地面全景适合判断云区交界和遮挡次序，侧视适合判断同一云体的底宽与塔冠，斜俯视适合判断连通性和云隙。绝对高度和间距按世界尺度试摆。

### 首版建模交付

本次授权在正常 Iris 云渲染路径中实现程序化云形。优先让淡、中、浓积云出现在同一片可飞入的天空，保留现有高云的天空合成。世界 XZ 的云区决定成团方向、蓝隙和凝结底高，同一 XZ 可以有多段云和层间净空。形状按 1 方块 = 1 米求值。相机、太阳透射和屏幕外采样在同一位置和游戏时刻应读到同一云场。继续使用现有光照完成形状接入。新光照的[输运对照](#云光照的物理对照)可独立推进。游戏晴雨驱动与云团生成消散留待后续。

每一层细节经过形变、侵蚀与过滤后仍须包含在当前父包络内。整团随风运动，局部可升腾、偏移。重载光影或重进存档时，相同世界位置、游戏时刻与设置应得到相同云形。底宽、塔高、瓣数、噪声与 LOD 细节交给实现者按实拍试调。用户选中的动画构图是 `anime-sky-01` 和 `anime-sky-03`，实拍是 `SKY-031`、`SKY-032`、`SKY-106` 和 `Cu-ref-02/03/05/07/09/10`。整幅图可用于观察主次、蓝隙、宽底和偏塔，具体云团的位置、高矮和数目允许变化。跨检出查看时从主检出的 `agent_inputs/cloud-references/` 打开图库。

交付以可运行的 shader 和同场景对照为准。用开阔地环视、绕塔飞近、穿云及云上俯视检查三种积云的轮廓与遮挡。用透射图核对塔间和层间的空隙，在动画与远近 LOD 交替处核对保守边界。在固定游戏时刻重载，比较云形。按相同场景、分辨率和设置对照改动前后的整帧 GPU 时间，首版上限为原来的 1.30 倍，同时记录云相关 pass 以便定位开销。实现、实机对照和可恢复的修正可连续推进，遇到影响美术目标、支持范围或帧时间上限的取舍时交回设计讨论。

### XZ 分布与垂直叠置

| 照片 | 可见特征 | 对生成器的约束 |
| --- | --- | --- |
| SKY-098、SKY-122 | 航拍中小塔断续成列，列内仍有缺口；另一些区域云块汇合，开口可以看见地面 | 分开控制云团中心的疏密、沿某方向的聚集倾向、单团足迹大小。三者共同决定覆盖率。 |
| SKY-033、SKY-200 | 全景里厚团、宽蓝隙和细碎云片相邻；另一张中数条云带近平行，其中一条由许多小块串成 | 给云斑、云带有限的水平范围和逐渐散开的边缘。允许带内断裂、同一片天空局部换形态。 |
| SKY-106、SKY-127 | 窗外的孤立塔高出较低云面；另一次航拍中，飞机上方是灰底，下方是能透见地面的云块 | 同一 XZ 可以有多个不连通的有云区间，层间净空必须可查询。塔基有时被前景挡住，不能据照片把它焊到前景云面。 |
| SKY-164、Cu-ref-02、Cu-ref-03 | 近处暗云挡住远方亮塔；另一组照片中几个塔头共用宽底、彼此高低错落 | 分别生成视线叠置和同体分枝。只凭亮度或图上的上下次序无法推断前后两块属于同一云体。 |

首版可在世界 XZ 上按种子重建有范围的云区。对流区由局部凝结底高、可发展厚度、聚集强度和带向控制。先用低频分布确定哪里聚团、哪里留空，再在区域内对带扰动的候选中心进行确定性筛选；沿带与横带使用不同相关长度即可得到断续云列。每个中心有独立的底宽与发展程度，邻近中心共享缓变的底高。高覆盖可以来自许多浅云，也可以来自少数连片的宽底。云塔可以孤立出现，也可局部聚成一列。厚度上限限制对流发展，实际塔顶由每簇上升枝决定。云区可叠置，不把一个地面坐标锁给一个云种。

默认水平和垂直尺度均为 1 方块 = 1 米。开阔平地转一圈，中远处至少要有一处能认出浓积云的轮廓，典型构图可读出一至三处，对流旺盛的方向也可以更多。一座孤塔、同一宽底上的几座塔头或远近搭接的两团都可以构成一处视觉重心。各处的大小和间隔应有差别，单独一个朝向可以留出很大一片蓝天。世界 XZ 上对流区的间距、范围与局部发展强度决定这幅构图。玩家的位置和朝向只用于观察。

本地图库 `agent_inputs/cloud-references/composition/index.html` 将四张动画截图中的六个局部与实拍并列。实拍 SKY-033 可看左侧厚团和右侧云片之间成片的蓝天，SKY-164 可看近处暗云的开口怎样框住远处宽塔和一串矮云头，SKY-183 可看几座细高云塔在开口中各自显出轮廓。首幅构图试取一组距海平面观察者水平 5 千米、底高 1 千米、顶高 3 千米的云塔，正侧面仰角约从 11° 到 31°。这些尺寸用于试摆，其他云区仍有各自的底高和可发展厚度。[当前高度与距离设置](../../scripts/options.main.kts)

层云、高层云和雨层云可先用有缺口的宽片体表达，以局部起伏的上下表面包住有厚度的云幕。层积云、高积云和卷积云用有限片区内的云卷或胞状团块，分别调节连通性、单元尺度、排列和层厚；细粒的视大小同时受距离影响。卷层云可用薄幕，卷云可用沿曲线展开、逐渐变细的纤维束。积雨云另需表达高处横展的砧部。对流塔遇稳定层而横向展开时，可将展开部分接在同一塔群上；独立过境的云幕则保留自己的底高、厚度与水平运动。[WMO 积云发展与稳定层](https://cloudatlas.wmo.int/en/explanatory-remarks-and-special-clouds-cumulus.html)

### 三种积云共用的几何

![三种积云在相同底宽和世界尺度下的剖面与俯视](figures/cumulus-support-sections.png)

图中三列以相同底宽比较塔高、分枝和侧向偏移。每列上下两图来自同一组三维体，黄线给出这一姿态的粗轮廓。子瓣可以在里面鼓起或侧倾，贴近包络边缘时密度逐渐降到零。实际云场中底宽独立变化，底部由若干横向搭接、边缘渐薄的云团构成，只在局部接近同一凝结高度。图中水平截面和简单塔冠用于比较几何，云底的密度过渡、花椰菜的小瓣和光照另行计算。

| 形态 | 粗级轮廓 | 中级细化 |
| --- | --- | --- |
| 淡积云 | 浅厚度预算，云肩横向展开，个体之间可留下大片净空 | 少量宽而低的云瓣，侵蚀集中在薄边，保住连续可读的浅云底。 |
| 中积云 | 某个上升枝形成可辨的主丘，较矮的云肩继续撑住原有底宽 | 把圆顶分成相互搭接的瓣，在主丘与侧肩之间留下凹口。 |
| 浓积云 | 由同一宽底长出偏心主塔、侧塔和矮枝。高处允许侧移及局部悬出根部 | 大云头朝顶侧分出一簇小瓣，保留大瓣与小瓣相接的颈部和塔间真实的凹口。远处应先读出主侧塔的阶梯。 |

种子固定各上升枝的位置、倾向和发育次序，发展程度连续改变各枝的高度、半径及冠部可见程度。底宽、主枝优势和可达高度分别控制。允许孤立浓积云，也允许同一个云场内多数云矮、少数云拔高。历史浓积云尝试中的宽底和主侧塔构造值得复用，旧归一化密度值需要另配世界长度单位的距离下界。

浓积云的形状对照沿用早期选中的 A 图宽底、B 图受光塔冠上的成簇小瓣，两图见本地图库的 `previews/model-pair.jpg`。Cu-ref-09 的塔尖轮廓清楚，同一团右上缘却已碎散。可把次级凸瓣集中在上升枝冠部，允许部分云肩逐渐碎散，保留一段宽而完整的底。云芯有多厚可用穿过塔身的透射图检查，另取一条穿过两塔开口的视线核对是否确实留空。

美术上先按实拍建立形体，把动画参考中的主次和留白带入布局。亮边、背光凹槽与日落色交给同一密度场在不同照明下呈现。主塔应夸张到什么程度留给原型比较。可固定世界位置、镜头、光照、曝光和底宽，只改变主塔相对侧塔的高度，对照全景中的辨识度和飞近后的轮廓。

#### 从实机画面校准积云的轮廓与疏密

建模候选 `ad10830fe` 的 Iris 镜位 `ground-180` 和 `ground-0` 能找到用户圈出的横条和斜塔。四张带标记的原图存于主检出本地图库的 `agent_inputs/cloud-references/experiments/2026-11-cumulus/`。以下看点对应这一轮候选。[建模实验聊天](codex://threads/01a200bd-9912-7992-b976-45bd87b0e457)

- 图 1、2 里，几处近云和远塔的塔身沿一个方向伸出很长一段，宽度变化小，末端收成圆头。实拍也有窄高、侧倾的塔。可对照 Cu-ref-03 的阶梯状侧肩和 Cu-ref-09 的不对称塔冠，先修根部、塔腰和冠部的宽窄变化。图 3 里矮云横过远塔，换方位观察视差能辨别哪些轮廓属于同一云体。[WMO 浓积云](https://cloudatlas.wmo.int/en/species-cumulus-congestus-cu-con.html)
- 图 4 的矮云有几道近乎平行的底边，上方圆瓣的大小和间距相近。实拍 Cu-hum-01 的浅底横向展开，到一侧渐细并断开。Cu-hum-03 的主体则可以很饱满，上缘清楚，下缘局部疏松。保留近似共同的凝结底高，让外围的厚薄、凹口和可透见背景的部分沿周缘变化。正在上涌的局部可有成簇凸瓣，混入周围干空气的薄肩可有顺风且断续的碎絮。优先在保守包络内塑出这些中尺度轮廓，细噪声只补近看可见的边缘。[WMO 淡积云](https://cloudatlas.wmo.int/en/species-cumulus-humilis-cu-hum.html) · [WMO 碎积云](https://cloudatlas.wmo.int/en/species-cumulus-fractus-cu-fra.html)
- 用云本身的透射率分辨形体与光照。`ad10830fe` 的离线单朵淡积云样本（`humilis-lit.bin`，15° 仰角）中，绿色透射 $T<0.95$ 的 24,218 条视线里有 13,667 条 $T<0.05$。该方向有过半的云投影遮掉了至少 95% 的背景光。实拍无法反推相同路径的消光系数，这个比例只用作该样本的前后对照。固定云与镜位查看云芯、薄肩、云底周缘和云隙的透射变化，再以侧视与下视对照路径长度的影响。如果细节在透射原图里清楚、在实机累积后消失，先查采样与重建。

### 保守步进与密度

首版由当前帧的父 SDF 直接裁边。以世界空间同一采样点为准，$P$ 是父包络，$C$ 是子瓣当前姿态的几何场，均取内负外正。几何层级逐级取交，沿每条父子路径只在最终密度上做一次边缘淡出。$\epsilon>0$ 是很小的数值余量，$w>0$ 是内侧过渡宽度。

$$
d=\max(P+\epsilon,C),\qquad
\rho=\rho_{\rm shape}\,\operatorname{clamp}(-d/w,0,1).
$$

$\rho_{\rm shape}$ 取局部形变、噪声侵蚀和密度过滤后的值。父场始终在原采样点求值。父包络外可以在读取细节前跳空，内侧窄带才压低密度。即使塔冠摆得太远，裁过的子瓣连同边界也落在 $P\le-\epsilon$ 内。每个子瓣先受自己的父瓣约束，再合并同级结果。若云边显出过于规整的轮廓，可复用塑形时已采到的非负噪声，让淡出起点局部向内退。现有积云可复用 [detailNoiseB 的采样结果](../../shaders/techniques/atmospherics/clouds/Cumulus.glsl)。

整片云的形状和细节共用一个由游戏时间确定的 XZ 风偏移，云瓣布局由区域格号和稳定种子重建。现有细节风取 `frameTimeCounter`，新云形应从 `worldDay` 和 `worldTime` 求风位移与形变相位，跨午夜连续，同一游戏时刻重载可重建。相机、太阳和屏幕外查询使用同一时刻的几何。空间格按本帧父包络纳入跨格云瓣。只掌握本格候选时，跳空最多走到本格出口。[现有风输入](../../scripts/shaders.properties) · [游戏时间输入](../../shaders/base/Uniforms.glsl)

积云塔群、片状云和纤维状卷云可以各用合适的形状表达。每类形状都给出射线上可能有云的范围或安全跳空距离，并能在给定世界位置求标量密度。相机、太阳透射和屏幕外查询可按各自的采样精度读取同一云场，远级仍须保住可辨的塔形和贯通云隙。当前低云视线与天空合成都使用固定积云层高。新场接入时应沿射线按实际深度处理近云、层间净空和远塔，在候选区间重叠处按局部介质积分一次。可用相机位于两层之间、分别向上和向下看作为接入对照。[低云视线](../../shaders/techniques/atmospherics/clouds/RenderVolumetric.comp.glsl) · [天空合成](../../shaders/techniques/atmospherics/SkyComposite.glsl)

步长取有距离下界的几何场。椭球可用 $r_{\min}(\|(p-c)/r\|_2-1)$，这类 1-Lipschitz 场的 $\min$、$\max$ 与反号仍保留该界，塔头之间用几何挖出的凹口也可跳过。薄片先裁射线的上下区间。两个 LOD 做密度混合时，各自先裁边，用包住两者的粗级跳空。[Hart，Sphere tracing](https://graphics.stanford.edu/courses/cs348b-20-spring-content/uploads/hart.pdf)

首版形状原型有两个可单独核对的细节。

- 同一云团内，底座、主塔和子瓣共同描述一份密度。可先对各瓣裁边后的标量密度取最大值。原样复制一个云瓣后，同一视线的透射应相同。若直接累加，原本 50% 的透射会在完全重复时降到 25%。
- 同底宽的三种积云里，淡积云较薄，固定米数的淡出容易吃掉宽底。对单个椭球，可先让 $w$ 取其最短半轴的一小部分，使过渡占各轴相近的比例。用淡积云核对云底是否仍连贯，再看浓积云冠部的小瓣是否仍分明。

第一轮几何对照可固定体场，从几个开阔地面位置环视，检查远塔是否可辨、主次和蓝隙是否自然。挑一座塔飞近并绕到侧面，看宽底、侧塔和层间空隙的关系怎样随视角变化。再以俯视覆盖、侧面轮廓和透射图检查云列间隙、分级塔冠和层间净空。穿过 SKY-116 一类的近云缘，在同一体场换太阳方向，看背光凹槽是否来自真实密度。远级密度近似需要另查列光学厚度，直接填满粗包络会抹掉细级云隙。首版整帧预算仍为同场景现状的 1.30 倍。示意图只检查了几何包含关系，实际求值次数、掠射步数及整帧耗时待 Iris 对照。

## 云光照的物理对照

用户重新看过早期逆光实验，认为末版背光云瓣边缘过于夸张。现在按相同密度、光源和相函数下的参考辐亮度判断云瓣对比度。参考里融入阴面的云瓣可以平滑，薄缘也可以随光向出现或消失。动画图继续提供构图参考。旧实验的四张反馈图保留在主检出 `agent_inputs/cloud-references/experiments/2026-09-congestus/`。

### SDF 能为光照提供什么

相机、太阳和二次散射射线共用形状的保守距离下界。若一个格心到所有候选云体的距离下界超过该格子的半对角线，整格的云散射源和云消光均为零，可省去细密度求值。光仍能跨过这段净空，从一座塔到达另一座塔。存在空气时沿用该段的空气输运。父包络内可能同时包含浓厚云瓣、稀薄边缘和侵蚀出的孔洞，太阳透射要用实际标量消光积分。相同的逐点太阳透射可供云自身直射和空气云隙光使用。[体积输运方程](https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer)

首个候选对太阳一次散射保留液滴相函数和细密度，云水采用标量消光和散射系数，太阳与天空保留各自的入射颜色。比较一个能传递侧面来光的较粗光场，用来估计经过多次散射的辐亮度。粗格跨过小云隙或薄缘时，可在接收点附近沿选定方向补算一段细密度的透射与沿途散射，并用它替换该段的粗估。每条光路的能量只计一次。SDF 法线与内侧距离不提供入射辐亮度，也无法单独证明云芯已进入扩散区。缓存方向数、分辨率以及哪些散射阶值得单独计算都由同成本对照决定。旧版 24 方向输运和双通道收敛加速器可从 `c09cfdc76e79cb3e4e986d16e29730777283e78d:shaders/techniques/atmospherics/clouds/Transport.glsl` 取回。[按散射阶研究云光照](https://www.icare3d.org/research-cat/publications/interactive-multiple-anisotropic-scattering-in-clouds.html)

### 改变邻云的离线对照

两团均匀球云半径均为 1，球心沿 X 相距 2.5，真空中只有平行太阳光。取 $\sigma_t R=2$、单次散射反照率 0.999 和 Henyey–Greenstein 相函数 $g=0.8$ 作为可复算的输入。相机在主云的 −Z 一侧。沿 +Z 或 +Y 望向太阳时，两球在光源投影和相机投影上均分开。加入邻云后，主云的一次散射完全相同，新增的二次散射只来自先在邻云、再在主云散射的光。两球间的透明间隙让这部分光正常通过。

下表取主云光学厚度为 1 的同一条边缘视线。$L_1$ 为主云的一次散射，$\Delta L_2$ 为加邻云引入的二次散射增量。二者均除以太阳垂直入射辐照度，单位为 $\mathrm{sr}^{-1}$。只统计散射光。

| 太阳方向 | $L_1/E_\odot$ | $\Delta L_2/E_\odot$ | $\Delta L_2/L_1$ |
| --- | ---: | ---: | ---: |
| 正逆光（+Z） | 1.31605 | 0.00023994 | 0.0182% |
| 侧光（+Y） | 0.00587884 | 0.00012563 | 2.14% |

正逆光时，两段透射的乘积为 $e^{-\tau}$，沿视线积分得到 $L_1/E_\odot=\omega_0 p(1)\tau e^{-\tau}$。在这个照明条件下，一次散射在 $\tau=1$ 达到峰值。侧光下邻云送来的二次散射绝对量也更小，但主云的一次散射降得更多。这个比值不能外推为所有阶的图像亮度变化。运行 `python scripts/cloud_neighbor_scattering.py` 可重算云芯和薄缘的两组结果。每个积分维度由 32 加密到 64 时，四个二次散射增量的相对变化均小于 0.4%。[积分脚本](../../scripts/cloud_neighbor_scattering.py)

### 学习多次散射的候选

实验分支 `10373f918` 的单塔侧光对照中，32³ 输运候选的整云红通道均值比同密度的路径追踪参考高 27.7%。取该场景参考与候选均值之比 0.7834，对表中总辐亮度作一次事后归一，上部薄缘仍高 10.4%，凹口高 11.2%，云芯低 2.3%。这是一张图上的诊断，说明还存在随位置变化的误差。已有光照实验也给出一次散射的上界，值得先定位局部多次散射与缓存取光的偏差。数据和口径见该提交的 `docs/sc/modules/atmosphere-clouds.md`「单塔侧光，仅太阳入射」。

已有跨云体测试。Deep Scattering 在 75 个云体上生成训练和验证数据，另留 5 个完整云体测试。MRPNN 以多尺度密度、向光源的透射和相函数特征估计高阶散射，留出 6 个云形测试体，作者也报告了碎体积经 mip 降采样后的亮度偏差。MRBNN 再将部分空间信息烘焙到网格中，用小解码器支持换光与换视角。程序化云形随风演变时，取特征、更新体积与屏幕外查询的总成本需要在 Iris 单独核对。[Deep Scattering](https://tom94.net/data/publications/kallweit17deep/kallweit17deep.pdf) · [MRPNN](https://sites.cs.ucsb.edu/~lingqi/publications/paper_mrpnn.pdf) · [MRBNN 的表示与实现](https://extra-creativity.github.io/projects/MRBNN/)

候选目标是在世界空间受光点预测往给定观察方向的二次及以上散射。保留实际密度积分出的视线透射与向光源的遮挡，并用介质相函数计算一次散射。在中性云水假设下，可学习单位太阳和少数天空入射分布的响应，再以当天各方向的 RGB 入射线性组合。便宜输入优先复用已有太阳光学厚度、邻域疏密，以及来自侧面或上方的粗尺度光照。父 SDF 内距相同、直达光路相同的两个位置，仍可能因为旁边多了一座受光塔而得到不同的间接光。学习器能分辨的情形受这些输入约束。

淡、中、浓积云在采用同一水滴光学模型时可共享一套估计器。测试按整朵云和种子留出，并用一种完整形态检验外推，覆盖薄肩、塔间凹口、厚云芯、云内视角和运动。水平延伸很远的层云及冰晶卷云要分别检查大尺度边界与相函数的变化。若做原型，用相同输入和路径追踪目标比较简单拟合或小 LUT 与小型神经网络，统计线性辐亮度的区域误差、参考不确定度与含特征更新的整帧时间。原型采用哪种表示仍待对照。[WMO 卷云的微物理组成](https://cloudatlas.wmo.int/en/physical-constitution-cirrus.html)

### 实机对照与交付

本次授权比较现有云光照和一个物理输运候选。使用固定的偏塔密度，以独立、增加样本后趋于稳定的离线体积路径追踪作参照。先用固定样本云体或旧实验云形即可开工。记录同一介质的世界尺度、相函数、太阳角径与能量、天空入射和相机，使用线性辐亮度比较。侧光和逆光各取一例，让新增邻塔避开目标像素的视线，以及从该段云体望向整个太阳盘的光路。仅开太阳时能隔离邻塔的间接贡献，开启天空入射后还要计入邻塔遮住的天光。对照受光薄缘、塔间凹口和背光云芯，在相同曝光下展示整图及局部，按需分出一次与多次散射并报告参考噪声。若参考本身看不出差异，保留该结果。双球算例验证的二阶项和历史 24 方向解都可用于诊断，全阶参照覆盖有限云体及其边界。

通过 Iris 的正常云路径交付同场景对照，记录云内和云上视角、逆光运动时的噪声及有关 pass 耗时。保持云形相同并切换光照以衡量新增开销。新形状接入后的整体上限仍是改造前同场景整帧 GPU 时间的 1.30 倍。执行者可根据误差与成本选更便宜的近似，持续修正可恢复的问题。实测仍难以兼顾画质与预算时，将同一画面对照带回设计讨论。
