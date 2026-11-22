# Alpha Piscium 版本路线图

## 2.0 发布进度

截至 2026-11-22，`2.0/dev` 已有方块体素化和不透明表面世界空间 GI，沿用普通 Bloom。以下两个候选各自基于 `5ba8718d4`，可以分别检出。它们的实机结果和开发分支的合入状态分开记录。

- **旧版 Iris 绑定**。体素、辐亮度缓存和路径引导使用 Iris 1.10.4 支持的 SSBO 编号，保留 Voxy 的 `colortex16/17/18`。官方 Iris 1.10.4 配 Sodium 0.8.2 和现用 Iris 都通过三组设置的加载与渲染。[目标版本、设置与 GPU 读回](rendering/iris-binding-acceptance.md)
- **屏幕空间与体素混合 GI**。普通场景已验证近处遮挡、离网后的取光、运动和历史复用。原生 Voxy 远景距离和端点精度仍待实测。720p 镜面房间中，体素与屏幕追踪、有限重连两个 pass 合计增加约 1.92 ms。[混合 GI 设计规格](rendering/hybrid-gi.md) · [场景与 A/B/B/A 计时](rendering/hybrid-gi-acceptance.md)

2.1 开发分支已有拟合和相机造型 Bloom。完整 DPF 生成器与 User 8 拟合继续在实验线验证。可在[实验分支的剖面编辑器说明](https://github.com/Luna5ama/Alpha-Piscium/blob/0e95a48f4744e2a1b7a15a8a171121d9e8eedabe/docs/development/offline-tools.md#interactive-bloom-line-profile)查到目标核与拟合结果的同线对照方法。云和统一体积渲染按 2.7–2.9 排期，实验可提前开展。

## 2.1 曝光与 DRT

**曝光与 DRT 的设计已确认，可在 `2.1/dev` 开工。** 默认使用 **Skibidi DRT + 中性 Look**，保留 Skibidi 自身的饱和度。玩家也可选择 AgX、OpenDRT 或 Psychov30（RenoDX test30）。Skibidi DRT 在上游收尾，移植时固定所采用的修订。AgX 先用固定输入、相同 EV、中性 Look 和同一 SDR 显示设备对照当前实现与 Blender 的 AgX，再决定是否升级。[Blender AgX 参考](https://docs.blender.org/manual/en/4.5/render/color_management.html)

**自动曝光测光。** 四种 DRT 共用一套相机曝光。在应用相机 EV、Bloom 和 DRT 之前，按场景线性亮度建立加权直方图。按累计测光权重裁掉最暗和最亮的两端，从保留区间求平均，驱动自动曝光。固定场景切换 DRT 或改变 Bloom 强度时，目标 EV 应一致。四种 DRT 共用一条随场景亮度变化的测光目标曲线，暗场目标较低，亮场目标较高。充分适应后，无火把洞穴里的石壁仍可辨认且明显偏暗，日照雪地仍显明亮。

建议以 `log2(Y)` 分桶，对保留部分求加权平均，再用目标中灰与测得亮度的档差计算目标 EV。百分位落在桶内时按重叠权重计入。最后限制 EV 范围并做时间适应。低端与高端裁剪比例分别用暗洞出口、雪地和小面积强光对照后选定。

**时域重建与相机曝光。** 原生 TAA 与超分共用一条时域重建路径。重建使用的曝光独立于相机 EV。历史保存未施加相机 EV 的场景线性颜色，重建后应用一次相机 EV，再进入 Bloom。只改变显示 DRT 或 Look 时可沿用重建历史。现有 FSR 原型在重建与锐化内部使用可逆 AgX 变换。[原型的接线](https://github.com/Luna5ama/Alpha-Piscium/blob/b72a8b5fddeb2e52ea20a2c36eb16e0c30b0661b/shaders/techniques/ffx/fsr3upscaler/README.md#integration-contract)

**共享 Look。** 四种 DRT 共用 AgX 风格的 Look 及其参数（RGB 斜率、偏移、幂和饱和度）。2.1 的 SDR 接线方案在已经过输出传递函数编码的显示 RGB 上应用一次 Look，再量化输出；以 sRGB 为基准验证。默认的中性 Look 使用单位斜率、零偏移、单位幂和单位饱和度，应能复现各 DRT 的无 Look 基线。可参考 bench 的[共享运算](https://github.com/Luna5ama/drt-bench/blob/84325c18c548e4a49e87b8ef845c4ffd79164265/shaders/ap/techniques/drt/Common.glsl)和 [OpenDRT 的编码后接法](https://github.com/Luna5ama/drt-bench/blob/84325c18c548e4a49e87b8ef845c4ffd79164265/shaders/ap/techniques/drt/OpenDRT.glsl#L233-L262)。四路共用这一位置是 AP 的整合方案。

参考 `drt-bench`，先在场景线性光上应用一次曝光，再按所选 DRT 的输入色彩空间转换。基准中 Skibidi 直接接收线性 ACES AP0，AgX 与 OpenDRT 经 XYZ 接入。Psychov30 的独立对照从 AP0 经 D60→D65 白点适配转为线性 BT.709，输出再做 sRGB 编码。移植时用相同输入和 EV 核对各自的 SDR 输出，避免重复施加曝光或输出编码。[AP 基准入口](https://github.com/Luna5ama/drt-bench/blob/a910d635b0d1b44ba193ec9eea023358dc841ac9/shaders/ap/main.glsl) · [Psychov30 对照入口](https://github.com/Luna5ama/drt-bench/blob/a910d635b0d1b44ba193ec9eea023358dc841ac9/tools/drt_compare_psychov.py)

### 显示色域缩限

Skibidi DRT 沿用自身的色域压缩，选项只设置目标色域。压缩从哪里开始、怎样向边界过渡，都沿用 Skibidi 的曲线。SDR 与 HDR 均可选择 sRGB／Rec.709、Display P3（默认）、Rec.2020 或自定义色域。自定义模式采用 D65 白点，玩家按显示器数据输入红、绿、蓝三组 CIE 1931 xy 坐标。缩限只约束 Skibidi 本身的输出。后续共享 Look 可以把颜色推到所选色域之外，随后按显示器或 HDR Mod 接口要求的原色与传递函数交付。AgX、OpenDRT 和 Psychov30 分别核对各自的色域压缩。已有压缩可接收目标色域时才接入这个选项，缺少对应机制的 DRT 跳过此项。

用灰阶、饱和色渐变和有颜色的高光核对缩限行为。分别在普通 SDR 和 HDR 输出中，核对同一组 D65 原色的预设与手填 xy 结果一致。以固定曝光对照预设和更窄的显示器色域，检查中性灰、色相连续性和靠近边界的颜色层次。用中性 Look 检查 Skibidi 输出的色域边界，再提高 Look 饱和度，验证最终画面保留由 Look 产生的越界颜色，测试时选用能承载该颜色的输出格式。

### HDR Mod 输出

2.1 提供 HDR Mod 的实际输出支持。相机测光和曝光沿用上面的设计。HDR 的峰值亮度和参考白从模组读取，随模组设置生效。rrtt217 的 HDR Mod 分别提供 `HdrGamePeakBrightness` 和 `HdrGamePaperWhiteBrightness`（尼特）。[模组的 shader 接口](https://github.com/rrtt217/Minecraft-HDR-Mod/wiki/For-Shaderpack-Devs)。DRT 按目标色域、参考白和峰值亮度映射场景颜色，输出级按目标模组的接口约定交付颜色，明确由哪一侧编码以及输出精度。关闭或未安装模组时使用现有 SDR 输出。

在启用目标模组的游戏和 HDR 显示器上，用固定 EV 的灰阶和高光梯度检查参考白，以及参考白以上直到目标峰值的亮度层次。记录模组版本、输出格式和实测的 DRT。关闭模组后用同一输入核对 SDR 基线。

## 2.1 原生 TAA 与超分辨率

原生 TAA 和超分都须支持 Iris 1.10.4。`2.0/fsr3` 的 FSR 3.1 原型作为对照，允许大改其算法，也可自研统一的时域重建器。选型时在同一段相机运动下比较细线闪烁、遮挡解除和运动拖影，并记录相同输入、输出尺寸下的 GPU 耗时和显存占用。

首版重建用的 Shadesmith tile 统一按输出分辨率分配。渲染分辨率的数据只读写 tile 内的有效区域，邻域采样和过滤也以有效区域为界。输出分辨率的颜色历史使用完整 tile，跨帧轮换。优先复用已有的同格式图集，减少物理 custom image 声明；沿用原型时，重建深度可接入现有 `R32UI` 图集，颜色历史可接入 `RGBA16F` 图集。

在原生和降比例两种模式下，分别开关法线贴图，核对生成的物理 image 数（Iris 1.10.4 上限 16）、最大图集尺寸和显存占用，再用目标 Iris 加载并验证历史和边缘采样。

## 版本范围

| 版本 | 范围 |
| --- | --- |
| 1.10 | 已完成陡峭与平滑视差映射、视差阴影和 GI，以及调色功能扩充。 |
| 1.10.x | 修复同一资源包混用不同纹理分辨率时的视差映射。 |
| 2.0 | 方块体素化和不透明表面世界空间 GI。旧 Iris 绑定和屏幕空间混合 GI 已发布独立候选，合入与实测范围见上文。 |
| 2.0.x | 优化 GI 管线，收回部分性能回退。 |
| 2.1 | [曝光与 DRT](#21-曝光与-drt) 重构，默认使用 Skibidi DRT + 中性 Look。[统一原生 TAA 和超分辨率](#21-原生-taa-与超分辨率)，加入 DPF Bloom 和 [HDR Mod 输出](#hdr-mod-输出)。 |
| 2.2 | 为世界空间 GI 加入镂空材质和生物群系染色材质，例如草与树叶。 |
| 2.3 | 为半透明表面加入体素追踪的世界空间反射与折射。 |
| 2.4 | 改善 ReSTIR GI 采样与降噪的画质和时域稳定性。 |
| 2.5 | 支持 Super Resolution 模组，并通过该模组支持 DLSS Ray Reconstruction。 |
| 2.6 | 待定。 |
| 2.7 | 重写积云形状，加入层积云和雨层云，改善低云光照。 |
| 2.8 | 将云影投射到地形与体积，加入体积屏幕空间阴影，让反射与 GI 的照明考虑云层。 |
| 2.9 | 重写卷云，加入高积云与地雾。 |
