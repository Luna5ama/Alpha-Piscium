# Alpha Piscium 版本路线图

## 2.0 发布进度

截至 2026-11-22，`2.0/dev` 已有方块体素化和不透明表面世界空间 GI，沿用普通 Bloom。以下两个候选各自基于 `5ba8718d4`，可以分别检出。它们的实机结果和开发分支的合入状态分开记录。

- **旧版 Iris 绑定**。体素、辐亮度缓存和路径引导使用 Iris 1.10.4 支持的 SSBO 编号，保留 Voxy 的 `colortex16/17/18`。官方 Iris 1.10.4 配 Sodium 0.8.2 和现用 Iris 都通过三组设置的加载与渲染。[目标版本、设置与 GPU 读回](rendering/iris-binding-acceptance.md)
- **屏幕空间与体素混合 GI**。普通场景已验证近处遮挡、离网后的取光、运动和历史复用。原生 Voxy 远景距离和端点精度仍待实测。720p 镜面房间中，体素与屏幕追踪、有限重连两个 pass 合计增加约 1.92 ms。[混合 GI 设计规格](rendering/hybrid-gi.md) · [场景与 A/B/B/A 计时](rendering/hybrid-gi-acceptance.md)

2.1 开发分支已有拟合和相机造型 Bloom。完整 DPF 生成器与 User 8 拟合继续在实验线验证。可在[实验分支的剖面编辑器说明](https://github.com/Luna5ama/Alpha-Piscium/blob/0e95a48f4744e2a1b7a15a8a171121d9e8eedabe/docs/development/offline-tools.md#interactive-bloom-line-profile)查到目标核与拟合结果的同线对照方法。云和统一体积渲染按 2.7–2.9 排期，实验可提前开展。

## 版本范围

| 版本 | 范围 |
| --- | --- |
| 1.10 | 已完成陡峭与平滑视差映射、视差阴影和 GI，以及调色功能扩充。 |
| 1.10.x | 修复同一资源包混用不同纹理分辨率时的视差映射。 |
| 2.0 | 方块体素化和不透明表面世界空间 GI。旧 Iris 绑定和屏幕空间混合 GI 已发布独立候选，合入与实测范围见上文。 |
| 2.0.x | 优化 GI 管线，收回部分性能回退。 |
| 2.1 | 重构 DRT 与曝光管线，用 FSR 3.1 统一 TAA 和超分辨率，加入 DPF Bloom。 |
| 2.2 | 为世界空间 GI 加入镂空材质和生物群系染色材质，例如草与树叶。 |
| 2.3 | 为半透明表面加入体素追踪的世界空间反射与折射。 |
| 2.4 | 改善 ReSTIR GI 采样与降噪的画质和时域稳定性。 |
| 2.5 | 支持 Super Resolution 模组，并通过该模组支持 DLSS Ray Reconstruction。 |
| 2.6 | 加入 Skibidi DRT、OpenDRT 和 HDR 模组支持。 |
| 2.7 | 重写积云形状，加入层积云和雨层云，改善低云光照。 |
| 2.8 | 将云影投射到地形与体积，加入体积屏幕空间阴影，让反射与 GI 的照明考虑云层。 |
| 2.9 | 重写卷云，加入高积云与地雾。 |
