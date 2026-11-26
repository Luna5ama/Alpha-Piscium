# 2.0 屏幕空间与体素混合 GI

**2.0 开发版已接入。** 普通场景和真实 Voxy 远景已分别完成实机验收。[实测记录](hybrid-gi-acceptance.md) 实现基线为 `2.0/dev@5ba8718d4`。交付范围是不透明主表面的 ReSTIR GI，包括初始射线求交、次级命中取光、历史复用和空间重连可见性。玩家使用一条混合路径。独立的 1.10 模式和透明表面的体素反射、折射均在本项范围外。[版本路线图](../roadmap.md) · [GI 管线](global-illumination.md)

## 一条射线上的先后关系

接收点为 `P`，单位方向为 `D`，沿用 2.0 采样器产生的同一个方向和 PDF。以下距离 `t` 一律从原始 `P` 沿 `D` 以方块计。与本帧体素网格包围盒求交得到正向区间 `[tEnter, tExit]`，起点在盒内时令 `tEnter = 0`。网格位置决定体素覆盖，普通渲染距离和 Voxy 像素身份都不能替代这个求交。Voxy 的不透明表面在 Hi-Z 构建前已经并入当前 G-buffer。[体素入射与离网](../../shaders/techniques/voxel/VoxelTrace.glsl) · [Voxy 合并](../../shaders/pass/composite/VoxyMerge.glsl) · [pass 顺序](../../scripts/programs.main.kts)

| 射线经过的范围 | 首次命中的搜索顺序 |
| --- | --- |
| 起点在盒内 | 在近处同时考虑可信的屏幕命中与体素遮挡。继续沿体素追踪到首个命中或盒出口。确认离网后，从出口到视锥边界再查一段 SST。 |
| 起点在盒外并射入盒内 | 盒前可见段全长查 SST。随后查盒内体素，若入盒时仍在原始起点的近段内，该重叠部分也纳入屏幕候选。穿出盒后再查可见尾段。屏幕前段离开视锥后，后续与盒相交的部分仍可追体素。 |
| 不与盒相交 | 对射线在视锥内的部分做 SST。全程没有表面命中时，按下文的环境规则结束。 |

近段长度只限定**盒内额外搜索屏幕几何**的范围。首轮可用 4 格作实现起点，再用实体遮挡场景和 720p 耗时选定内部常量。它不截短盒外的前段或尾段，也不会在远处入盒时重新起算。2.0 接受中段只在屏幕上有表示的实体可能漏掉，这一取舍限制了每条射线重复追踪的长度。

重叠段中，设最近的可信 SST 候选在 `tS`。体素必须从盒入口查到 `tS`，包括 SST 已经走过的部分。更近的体素墙面遮住实体。体素已查通到实体才收下实体。若两者位置、面朝向和材质归属都指向同一表面，选屏幕的材质细节。若只有深度接近，按细化后的距离取较近命中。归属不明的屏幕表面只使用自身材质和有效的屏幕历史。SST 没给出候选时从盒入口照常追体素。遍历后的同一段无需从头再追。

## 厚度、续追与结束条件

- 每个 SST 区间先按当前视锥和近裁面裁剪，再用同一帧的投影与 jitter 约定构造屏幕线段。Hi-Z 厚度只产生候选。用 mip 0 实际深度重建表面，结合几何法线细化射线交点，检查交点落在该像素覆盖及原区间内。轮廓、近切线和仅擦过厚度带的结果需要这项确认。跳过失败的候选后向前推进。自交排除只覆盖起点自身。拼接处的首个有效像素仍可能是命中。[SST 区间、起步与终止](../../shaders/techniques/SST2.glsl)
- 屏幕步进的 `currT` 是投影线段的参数。通过逆投影把交点还原到同一坐标系，取 `dot(hit - P, D)` 比较先后和写回距离。跨 pass 可由原始射线和包围盒重建尾段起点。现有 `sstray_recoverOrigin` 从接收像素深度恢复原点，直接拿它恢复出口段会从错误的位置重追。
- 正常离网后才进入尾段。盒内预算耗尽且尚未查通到下一个候选，沿用 2.0 的零辐亮度样本，保留原 PDF。盒内 SST 预算耗尽可由覆盖该段的体素继续。只有屏幕覆盖的可见段若用完预算，保守地返回零贡献。离开视锥则结束该屏幕段，继续检查射线上剩余的体素覆盖。单层深度看不到的几何仍是屏幕近似的限制。[现有耗尽语义](../../shaders/techniques/gi/InitialSample.glsl)
- 所有可搜索区间正常结束仍无表面命中时，按射线方向取环境探针。探针有有效几何时沿用距离渐变，无有效几何时取天空。这产生一个方向环境样本。只有可信表面才写正命中距离与几何法线。全程只提交一次候选，切段、仲裁或丢失历史都不会重抽方向。[环境查询](../../shaders/techniques/gi/Irradiance.glsl)

## 命中点如何取光

设 `V` 为从次级命中指回本次接收点的方向。候选保存工作色域下沿 `V` 的出射辐亮度。用命中材质计算当帧自发光和朝 `V` 的直接光，再取一份间接光。屏幕命中取当前 G-buffer 材质，体素命中取体素材质。体素命中投到屏幕时须确认像素记录的正是这个表面。下面的“有屏幕历史”都指**命中像素**的历史可用。[命中着色](../../shaders/techniques/gi/HitDirectLighting.glsl)

| 命中点的间接光来源 | 2.0 的选择 |
| --- | --- |
| 同一表面既有有效 RC 面又有屏幕历史 | 优先取 RC 沿 `V` 评估的结果。RC 给出有效的零也应保留。 |
| 只有有效 RC 面 | 取 RC。离屏方块仍可给接收点供光。 |
| 只有可靠的屏幕历史 | 取上一帧已累积、现已重投影到命中像素的漫反射，用当前命中材质恢复漫反射出射项。 |

两种间接来源均无效时，只用该表面的当帧自发光与直接光，保留几何命中。RC 关闭时也走这条判断。表面归属可由同位置的方块、面和材质核对。仅有 RC 查找权重、方向接近或位于网格内，无法证明实体与缓存面相同。

屏幕输入采用 `transient_gi2Reprojected.rgb` 的 diffuse fast history，沿用旧路径的 `min(hitAlbedo, 0.95)` 调制。依据命中处的重投影、历史有效性与重置状态剔除刚显露的背景。已经打包的 `transient_giRadianceInputs` 混有当帧漫反射直射与该历史，整包相加会重复计光。屏幕历史的 RGB 只有漫反射信息，不能恢复沿新方向的历史镜面光。当前 G-buffer 在追踪前已就绪，跨像素读取法线时直接用它，避免从正在同一次 dispatch 写入的 `transient_geomViewNormal` 取邻像素。[屏幕历史生产端](../../shaders/pass/composite/DirectLighting.glsl) · [历史有效性](../../shaders/techniques/gi/Reproject.glsl)

RC 面保存余弦加权的平均入射辐亮度，`rc_lookupDiffuseGI` 用面材质、法线和 `V` 将其转为出射量。读取 GI 前已更新完的 current side。RC 自身的递归反弹沿用 previous side。两种间接来源估计同一份反弹，首版按表面有效性择一。冻结入射光的参考模式关闭两种间接反馈，环境 miss 使用固定照明下的天空，避免探针把随帧变化的历史重新引入参考。[RC 取样](../../shaders/techniques/gi/RadianceCacheSample.glsl) · [RC 更新](../../shaders/techniques/gi/RadianceCacheUpdate.glsl) · [初始样本参考](../../shaders/pass/composite/GIReferenceAccumulate.comp.glsl)

## 接入 ReSTIR

1. 保留现有方向采样、可选路径引导和 PDF。在 `InitalSampleHiZ` 生成方向后及 `InitalSampleVoxelFallback` 求交的位置完成上述搜索，所有段使用同一方向。允许按实际访问成本拆分 dispatch。向参考累积和时间复用提交已解析的辐亮度、方向、PDF、相对原始 primary 的总距离和命中几何法线。体素 pass 中的 `restir_temporalSplat` 位于候选筛选之前，屏幕已经命中的像素也需要执行。[初始采样入口](../../shaders/pass/composite/GIReSTIRInitalSampleHiZ.comp.glsl) · [体素与 splat 入口](../../shaders/pass/composite/GIReSTIRInitalSampleVoxelFallback.comp.glsl)
2. 沿用 2.0 的时间 reservoir、重投影和 Jacobian。仅屏幕可见的次级表面跨帧复用时，将历史端点投到当前帧核对深度与几何法线。运动或显露导致端点无法确认时丢弃这份历史。若当前编码无法表达这类端点的归属，可让这些样本仅参与当帧着色和空间复用。离屏仍有体素表示的端点保留原有时间复用能力。重载、场景切换和历史重置同样适用于新增的屏幕贡献。[时间复用](../../shaders/pass/composite/GIReSTIRTemporalReuse.comp.glsl)
3. 空间复用沿用已有的有限邻居获选队列。从**当前接收 primary** 朝重连端点作相同的分段验证，只检查端点之前的遮挡，并给端点留自交容差。体素只查线段与盒的交集，盒外屏幕段走到预期端点且没有更近可信遮挡即可通过。更近的遮挡、预算耗尽或缺乏覆盖而无法验证的区间使该邻居贡献归零。盒内近段仍可由体素完成。保留原有 ratio 镜面分量、shadow hint 和路径引导的拒绝处理。方向环境样本沿用 2.0 的复用规则。[邻居入队](../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl) · [可见性消费端](../../shaders/pass/composite/GIReSTIRSpatialReuseTrace.comp.glsl)
4. 优先复用候选 tile 和可见性队列。实现将正距离的 FP32 位型写入现有 spatial-input tile，原 FP16 通道保存表面标记或终止状态。Voxy 远景验收核对了写入、读回和端点重建误差。降噪器的镜面距离限幅只用于滤波。Voxy 专设的 `colortex16/17/18` 保留原用途。旧 Iris 的 SSBO 绑定已随本版合入。[资源布局](../../shaders/shadesmith.json) · [旧 Iris 发布项](../roadmap.md)

重连时沿用 2.0 的近似，对新的主表面、方向和材质重算目标权重，次级端点携带初次取光时的 RGB。新连接的出射方向变化很大时，有光泽次级表面的亮度可能偏离重新着色的结果。用倾斜亮面比较关闭与开启复用的画面，将这项差异与选择了哪份间接光分开判读。[空间重连与目标权重](../../shaders/pass/composite/GIReSTIRPairedSpatialShade.comp.glsl)

## 2.0 验收

| 场景 | 可判定的结果 |
| --- | --- |
| 网格内的近处实体、薄模型，另一次在射线前面放体素墙 | 无墙时可取实体或模型。有更近的墙时墙遮住它。剪影附近掠过厚度带不会形成悬空命中。记录原点、屏幕和体素的命中距离即可核对仲裁。 |
| 从 Voxy 远景朝网格射入，再射向可见远景。另取一条完全绕过网格的射线 | 分别覆盖盒前、盒中、出盒尾段和纯屏幕命中。用同一个原点重建端点。空间复用开启时，能走通的盒外连接保有贡献，中段有墙时被拒绝。 |
| 发光方块转出画面、实体移开露出背景、RC 开关切换 | 逐项覆盖上表三种间接来源和均不可用的情况。RC 有效的离屏方块仍供光。无效历史不把背景光粘到实体。有两种来源时不出现双计。切换后旧历史不延续。 |
| 走出屏幕、正常出盒、步数耗尽 | 只有正常走完整条搜索路径才取环境。体素耗尽的射线不从出口继续，有限重连未走通时不放光。 |

在 Vibris 核对实际 pack、修订、场景、设置与 pass 成功加载。用固定场景记录初始命中来源及去向、降噪前 diffuse/specular 与最终画面，再做机位移动和历史重置对照。取同输入的 `5ba8718d4` 与候选做 720p 成对计时，记录初始采样、体素/尾段、空间可见性及一个稳定的对照 pass。近段距离和步数由这些对照定案。2.0 的完成条件是混合路径实机正确，2.0.x 性能回收单独跟踪。

### Greenfield 远景窄视野的本地修复

在 `3109eb0a` 上的本地候选 `0c4d7c76` 用几何平面在像素内的反向深度范围构建混合 GI 的 Hi-Z 包围区间。原中心深度极值留在 RG，平面范围存入 BA。投影变窄时，射线可以跨过与真实表面分离的图块。初始求交和有限空间重连仍使用 128 步，耗尽时分别写零和拒绝重连。

Greenfield 两个 1080p 屋顶复用封存镜位、内置 PBR、Iris 1.10.7 和 Voxy 0.2.13-alpha。在 1.875° 的两个稳定帧，屋顶一的耗尽数从 1,750/2,229 降至 13/2,229，屋顶二从 2,096/2,446 降至 23/2,446。按同一世界点匹配，从 15° 缩到 1.875°，候选滤波后漫反射均值分别从 5.57×10⁻⁵ 到 6.21×10⁻⁵、从 5.27×10⁻⁵ 到 6.11×10⁻⁵。视锥缩小仍会丢失离网且不在屏幕内的光源和遮挡信息。

实机在 26 个稳定帧上检查了 191,333 个远景样本。穿面、面前、离开视锥的有限段分别得到遮挡、可见、覆盖不全。对 4,096 步基线参考能判定的自然射线和斜向重连，未发现错误放行。两档 1.875° 的成对测量中，Hi-Z 生成、初始追踪和有限可见性三项合计分别从 7.257 到 6.041 ms、从 7.458 到 6.091 ms。计入后续 GI 计算分派，屋顶一从 10.749 到 9.722 ms。屋顶二的两个滤波分派增时，全链从 11.723 到 12.458 ms。1080p 下新增 Hi-Z 存储为 31.64 MiB。

运行 78373ac1-da9d-47fb-85d1-f86edfaaa2f3 保留成品、原图、统一亮度标尺图、配对时间及复测入口。对照成品为 b829f495，渲染代码与执行基线相同。

<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"report.md"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"reproduce.md"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"AP-329-0c4d7c76.zip"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"roof-1-before-after.png"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"roof-2-before-after.png"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"audit.json"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"occlusion.json"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"timings.json"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"radiance.json"} -->
<!-- vela:evidence {"run":"78373ac1-da9d-47fb-85d1-f86edfaaa2f3","path":"protocol.zip"} -->

#### Greenfield 连续变焦的本地验收

实际成品为 `0180c16053b8a17723fb5e9b5c823dac64d5b5af`（分支 `codex/greenfield-zoom-root-343`），对照为 `a55c4d6e`。在编号 12 检出打包。成品 `AP-343-native-emission.zip` 的 SHA-256 为 `d0c06e689b8377c8abfb31171f357e18f7d1748e0d4189f71cdc451c39500b56`。运行使用 Minecraft 1.21.11、Iris 1.10.7、Sodium 0.8.7、Voxy 0.2.13-alpha、内置材质，开启 RC 和路径引导。固定 Greenfield v0.5.4 的夜间屋顶。

基线在 0.11719° 时仍能看到接收面。第一盏萤石已完全离开视锥；对其实体有交点的 1660 条射线里，1647 条进入环境分支，13 条耗尽。宽视野缓存可以找回表面，但低分辨率灯像素可能跨入空气格，最后一个可见发光纹素还会随变焦位置改变整盏灯的缓存亮度。独立反例分别保存在运行 `b23807c9-0e45-474d-9475-92e4343602bc` 与 720p 运行 `eae54a8a-4f60-4c2e-a093-caaf82b70c02`。

修正沿原世界射线仲裁当前深度、体素、保留的宽投影与已观测的原生块模型。在局部 32³ 格区域，灯前遮挡按真实模型测试。原生像素的足迹或已见方块确定材质归属，按已观测的 8×8 面区域在线性空间估计内置自发光。候选保留原来的 PDF 与世界距离。普通视野运行独立程序，Voxy 的 colortex16/17/18 保留原用途。

| 同一屋面的稳定帧 | 基线漫反射 | 候选漫反射 | 确定无遮挡命灯 | 确定有遮挡仍命灯 | 未过灯实体仍命灯 |
|---|---:|---:|---:|---:|---:|
| 第一屋面 0.05859° | 0.0000492 | 0.0022598 | 1580/1702 | 0/199 | 0 |
| 独立屋面 0.05859° | 0.0000505 | 0.0008963 | 1393/2071 | 0/346 | 0 |
| Zoomify 第 20 档 0.06000° | 0.0000493 | 0.0022391 | 1096/1181 | 0/147 | 0 |

第一屋面在 0.05859° 单独熄灭该灯的次级发光后降至 0.0000631。以 0.11719° 冷启动且未观测到屏外的灯时，保留的离屏灯命中为 0。Zoomify 2.15.2 原生输入的最大除数是 500；另外用实测投影连续覆盖 30° → 0.05859° → 30°，保留了过渡帧、稳定帧和奇偶相位。整组未重启共享游戏。

| 1080p 成品 GI 整链，ms | 基线 | 候选 | 基线/候选对照 pass |
|---|---:|---:|---:|
| 70° 入场 | 11.148 | 12.012 | 0.039/0.038 |
| 30° | 13.251 | 25.366 | 0.039/0.039 |
| 1.875° | 16.190 | 36.708 | 0.033/0.034 |
| 0.46875° | 18.623 | 36.535 | 0.031/0.032 |
| 0.05859° | 19.376 | 33.898 | 0.029/0.029 |
| 70° 恢复 | 14.021 | 12.106 | 0.040/0.039 |

变化的固定分配在 1080p 为宽投影 31.64 MiB 和全局 SSBO 24.25 MiB。四臂逐进程显存及正反序阶段查询见 `timings-lights.json`。极窄视野中能看见约 64 像素周期的横纹。实际 500× 稳定帧的滤波后行亮度在该频率上的振幅约为均值的 21%。对首次进入视野前已在屏外的光源，以及未观测或在屏外变化的遮挡，仍缺少可用场景数据。

最终原件在运行 `fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6` 的 `report.md`、`verify.json`、`paired-3-7.png`、`paired-4-7.png`、`zoomify-500x.png`、分臂归档与 `protocol.zip`。两个屋面使用同一线性白点，原始终图保留游戏曝光。`source-energy.json` 给出真实命中灯面的源辐亮度，`edge-contacts.json` 给出存档实体与原点附近的亚格接触。

在 0.05859° 将同一世界点配准到 0.9375° 参考图，第一屋面的滤波后漫反射为参考的 3.40 倍，独立屋面为 1.03 倍。两个屋面都恢复受光，第一屋面的亮度仍明显偏高。独立屋面 0.11719° 第 159 帧有两条被存档几何分类为遮挡的命灯射线，与发射面交叠的最远出口距原点 0.000813 格。

<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"AP-343-certified-blocks.zip"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"AP-343-contact.zip"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"AP-343-model.zip"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"AP-343-observed-lights.zip"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"clean-outputs.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"contact-errors.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"diagnosis.md"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"emitter-footprint.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"fixture-accounting.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"identity343.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"lamp-context-3.png"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"lights-clean-outputs.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"local-world-geometry.zip"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"model-ready.txt"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"paired-3-7.png"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"paired-4-7.png"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"preliminary-archive.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"protocol.zip"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"report.md"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"reproduce.md"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"retained-occlusion-counterexample.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"static-contact.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"static-lights.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"static-model.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"static-owner.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"static-selected-materials.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"timing-raw.zip"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"timings-contact.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"verify.json"} -->
<!-- vela:evidence {"run":"b23807c9-0e45-474d-9475-92e4343602bc","path":"zoomify-500x.png"} -->
<!-- vela:evidence {"run":"64c59ce8-d8ee-494d-8214-ef702fa28eda","path":"preliminary-frames.zip"} -->
<!-- vela:evidence {"run":"64c59ce8-d8ee-494d-8214-ef702fa28eda","path":"preliminary-manifest.json"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"cold-3-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"coldowner-3-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"model-3-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"model-4-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"modelcut-3-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"scan-3-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"scan-4-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"split-3-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"split-4-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"splitcut-3-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"zbase-3-frames.zip"} -->
<!-- vela:evidence {"run":"20f904e8-2c39-4782-9b25-d95fb1b6b872","path":"zterrain-3-frames.zip"} -->
<!-- vela:evidence {"run":"eae54a8a-4f60-4c2e-a093-caaf82b70c02","path":"720-originals.zip"} -->
<!-- vela:evidence {"run":"eae54a8a-4f60-4c2e-a093-caaf82b70c02","path":"compare720.json"} -->
<!-- vela:evidence {"run":"eae54a8a-4f60-4c2e-a093-caaf82b70c02","path":"fixture-accounting.json"} -->
<!-- vela:evidence {"run":"eae54a8a-4f60-4c2e-a093-caaf82b70c02","path":"source-ownership.json"} -->
<!-- vela:evidence {"run":"eae54a8a-4f60-4c2e-a093-caaf82b70c02","path":"source-ownership.py"} -->
<!-- vela:evidence {"run":"0a247f85-beac-49e7-84a5-4822d034b013","path":"predecessor-images.zip"} -->
<!-- vela:evidence {"run":"0a247f85-beac-49e7-84a5-4822d034b013","path":"predecessor-timing.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"AP-343-native-emission.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"AP-343-source-radiance.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"coldowner-3-frames.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"contact-errors.json"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"edge-contacts.json"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"identity343.json"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"lights-clean-outputs.json"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"local-world-geometry.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"paired-3-7.png"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"paired-4-7.png"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"paired-timings.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"protocol.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"report.md"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"reproduce.md"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"row-variation.json"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"source-energy.json"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"split-3-frames.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"split-4-frames.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"splitcut-3-frames.zip"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"timings-lights.json"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"verify.json"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"zoomify-500x.png"} -->
<!-- vela:evidence {"run":"fe3f9c72-4e9f-48e1-bfee-9ad75a1cecd6","path":"zterrain-3-frames.zip"} -->

#### Greenfield 两个屋面的投影选择、条纹与成品复核

本地修订 55cab30c 的普通入口从 70° 放大后实测仍能追到离屏萤石。第一屋面以 70° 投影记录时，约 930 格处每像素覆盖 1.206 格，30° 时为 0.461 格。当稳定的新投影能分辨单格、完整包含原有 32³ 邻域而旧图每像素至少一格时，更新参考。先前保持粗投影的两次读回在 393 条可见命灯方向中只命中 50 条，一次有 338 条先碰到旧图的暗平面。修正后两个独立进程都取得 369/393、有墙 0/65、灯盒外误命中 0。缓存更新后，退出精细视锥的旧宽图区域仍有数据边界。直接从冷启动跳到极窄视野也缺少可恢复的几何。

原生 Zoomify 500× 的周期条纹源于接收面绝对深度量化。实测深度台阶宽 0.0625 格，预测周期约 65 像素，图上为 63.47 像素。使用相对匹配帧全精度深度的有符号残差后，原先约 21% 的谱线在稳定相邻帧为 0.121%、0.072%。该档同方向可见与遮挡命灯为 275/301、0/37，275 个本地命中均从实际 Voxy 模型图集取得发光表面纹素。两处屋面 0.05859° 相对各自 0.9375° 同点宽图为 0.855、0.939 倍，相对 a55 的宽图为 3.61、2.48 倍。基线第一屋面 70 个可见命中里有 55 个写入了背向射线的法线，影响复用权重。独立屋面的全速蒙特卡洛对照也显示，复用、滤波和显示曝光会改变绝对强度。

对照 a55，1080p 成品 GI 链在 70° 起步为 11.62 → 11.75 ms、30° 为 13.36 → 13.59 ms、0.05859° 为 18.82 → 40.83 ms、恢复 70° 为 11.46 → 13.36 ms，图像和 SSBO 分配增加 57.52 MiB。六臂对照否决了另拆绕网格与入网格 pass 的 5d8e2077，增量来自两个程序的合计。当前代价、标定与独立镜位、墙体和熄灯的原始计数，以及稳定奇偶帧在运行 1d41dade-eace-40fb-ac7b-46727f12d2e4 的 report.md 和 reproduce.md。同目录提供精确的 55cab30c ZIP 和配图。先前对深度、采样率、材质、保留投影和分支成本的消融运行保留在各自记录。

继续测试同一原生粗糙表面按材质纹素宽度扩大到 4×4 采样。实测最窄档 55cab30c 与 89173a6b 的成对整链为 38.21、30.61 ms；独立屋面相对自身宽图的稳定照明由 0.939 倍降到 0.499 倍，第一屋面绝对值也减半。因此主交付保留 55cab30c。该对照的六臂计时和同图可在 1d1a2563-c771-4935-8ee6-20d448f79267 复核。

<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"depth-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"depth-3-stable.json"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"depth-500x.png"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"fbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"fixture-accounting.json"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"frozen-attribution.json"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"fstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"job-logs.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"noLocal-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"noWide-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"raw-stable.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"replay-tools.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"revfbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"revfstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"sealed-arms.json"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"split-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"split-3-stable.json"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"split-4-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"split-4-stable.json"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"timings-lights.json"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"validatedcut-3-receipts.zip"} -->
<!-- vela:evidence {"run":"a708bc9a-c89b-4008-908a-e6f0a44f99b4","path":"validatedcut-3-stable.json"} -->
<!-- vela:evidence {"run":"df2de4a6-4249-40e9-b291-b96780a88325","path":"job-logs.zip"} -->
<!-- vela:evidence {"run":"df2de4a6-4249-40e9-b291-b96780a88325","path":"native-witness.json"} -->
<!-- vela:evidence {"run":"df2de4a6-4249-40e9-b291-b96780a88325","path":"raw-stable.zip"} -->
<!-- vela:evidence {"run":"df2de4a6-4249-40e9-b291-b96780a88325","path":"replay-tools.zip"} -->
<!-- vela:evidence {"run":"df2de4a6-4249-40e9-b291-b96780a88325","path":"zterrain-3-receipts.zip"} -->
<!-- vela:evidence {"run":"9dae2237-555c-4978-961b-c186e0392eca","path":"job-logs.zip"} -->
<!-- vela:evidence {"run":"9dae2237-555c-4978-961b-c186e0392eca","path":"native-witness.json"} -->
<!-- vela:evidence {"run":"9dae2237-555c-4978-961b-c186e0392eca","path":"raw-stable.zip"} -->
<!-- vela:evidence {"run":"9dae2237-555c-4978-961b-c186e0392eca","path":"replay-tools.zip"} -->
<!-- vela:evidence {"run":"9dae2237-555c-4978-961b-c186e0392eca","path":"zterrain-3-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"AP-343-da0cc5fb.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"edge-contacts.json"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"emission-reference.json"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"face-audit.json"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"fbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"fstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"fullrate-4-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"fullrate-result.json"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"job-logs.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"lights-clean-outputs.json"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"production-identity.json"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"raw-stable.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"replay-tools.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"report.md"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"reproduce.md"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"revfbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"revfstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"roof-1.png"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"roof-2.png"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"split-3-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"split-4-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"summary.json"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"timings-lights.json"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"validatedcut-3-receipts.zip"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"zoom-500x.png"} -->
<!-- vela:evidence {"run":"172b84b3-94c9-4617-be69-b15ee88e7bba","path":"zterrain-3-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"fbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"first-real-grid-split.json"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"fstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"job-logs.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"lights-clean-outputs.json"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"oldopt-3-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"paired-output.json"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"partition-results.json"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"raw-stable.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"replay-tools.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"revfbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"revfstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"revoldopt-3-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"split-4-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"warm-inputs.json"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"warm-paths.json"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"warm70-result.json"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"warmone-3-receipts.zip"} -->
<!-- vela:evidence {"run":"d903f499-714a-4d60-9bce-a988b6e75f47","path":"warmtwo-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"70-start-before-after.png"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"AP-343-55cab30c.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"edge-contacts.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"emission-reference.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"fbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"first-roof-promotion.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"fstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"identity.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"job-logs.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"lights-clean-outputs.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"raw-stable.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"registered-retest.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"replay-tools.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"report.md"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"reproduce.md"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"retained-projections.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"revfbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"revfstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"roof-1.png"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"roof-2.png"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"split-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"split-4-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"static.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"summary.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"timings-lights.json"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"validatedcut-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"warmone-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"warmtwo-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"zoom-500x.png"} -->
<!-- vela:evidence {"run":"1d41dade-eace-40fb-ac7b-46727f12d2e4","path":"zterrain-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"AP-343-89173a6b.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"fbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"first-check.json"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"fstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"identity.json"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"job-logs.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"lights-clean-outputs.json"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"oldopt-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"partition-results.json"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"rate-before-after.png"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"rate-quality.json"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"raw-stable.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"replay-tools.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"report.md"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"revfbase-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"revfstock-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"revoldopt-3-receipts.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"roof-2.png"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"split-4-receipts.zip"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"storage-at-handoff.json"} -->
<!-- vela:evidence {"run":"1d1a2563-c771-4935-8ee6-20d448f79267","path":"warmone-3-receipts.zip"} -->
