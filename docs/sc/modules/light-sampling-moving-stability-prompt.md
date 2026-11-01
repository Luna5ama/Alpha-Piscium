# Light Sampling 移动稳定性实验提示词

语言：简体中文 | [English](../../en/modules/light-sampling-moving-stability-prompt.md)

## 本轮结果（2026-10-31，待独立复查）

最小候选已提交 `0d1629ae`；旧源码固定为 `6d69b145`。[结果索引](../../../experiments/light-sampling/20261031-moving-stability.json)和[24 图对比页](../../../builds/light-sampling/20261031-moving-stability/comparison.html)保留四个 preset 的旧 A、新 A、Cosine BRDF，以及 1 spp／32 连续帧均值。
本轮质量和成本沿用已有受控基线 G=16；preset 未覆盖的 shader 默认 G=64 另作编译检查，不冒充已完成 G=64 质量验证。
首次旧 A 的三个输入漂移采集原样保留并排除；补采后每个 pose 的旧／新灯及完整世界体素输入均一致。pose 2→3 跨 y-brick 的实际灯／遮挡体也相同：旧语义拓扑改变，两个固定世界点的完整 PMF TV 距离约为 0.789／0.900；新拓扑一致，三个点 TV 均为零。
pose 1→2、3→4 则确实发生窗口灯及体素输入变化（灯数 1182→1136、1136→1083），不属于同输入不变性保证，不能宣称所有位置跳变已消除。
[世界坐标诊断](../../../scripts/light_sampling_moving_analysis.py)按实际 G 和重建相机还原体素，不使用原始槽号或内存哈希判断跨相机身份。三个接收点均按世界位置、法线、材质、反射率匹配，24 项均有完整 32 帧样本。

| Preset 后缀 | 新／旧全屏 RGB 样本方差比 | 质量判断 |
|-------------|-------------------------|----------|
| 1 | 3.376 | 明显回退 |
| 2 | 3.839 | 明显回退 |
| 3 | 0.0642 | 大幅改善，局部仍有回退 |
| 4 | 1.900 | 明显回退 |

45 组离线检查覆盖三种 G、二／四／八叉、负坐标、空／单灯及跨界完整 PMF。八份真实灯树的包围、mass、实取 PDF 和灯内 PDF 审计通过；Target Iris 同时完成默认四叉、32／64 网格的二／八叉 48B 及实验关闭的编译检查。
环绕反例确实把跨度 241 格的簇从第 1 层推到第 5 层合并，稳定性不能代替质量验收。代表场景 ABBA 测量完整 14 个 build 核、initial、共享 prepare 与 sky LUT；建树中位约 0.139–0.141 ms，没有可信加速结论。少量非有限 RGB 单列并从共同完整像素统计排除。
本轮结论为机制验证成立、质量有显著代价，不推荐无条件接受周期分组。未测试连续运动或跨界第一帧，不把 32 帧短窗口方差当作收敛误差或亮度偏差证明；保留候选和原件供讨论端复查，不扩大重构。

## 授权、起点与交付

用户在讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 的 turn `01a1947c-4f4f-7f21-bca6-465fe2a724af` 明确要求：

> 我准备了几个light-sampling-failure-moving preset，你自己简单设计一下修改方案然后让实验agent做实现，完事之后你让那个agent发消息回来提醒你做复查

本轮授权实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429` 实现下述最小候选、完成针对性验证，并主动发消息回讨论 chat。
用户报告的是“越过某个位置后，整片区域的噪声或亮度突然变化”。实施完成与复查接受分开；本提示词不是收益结论。

使用 `I:/code/mcshaders/Alpha-Piscium-4`（wt4）、`2.0/light-sampling`。编写时 HEAD 为 `6d69b145ec7a1e70ed719886549242e4fd93542f`，工作区干净。
执行时重新核对差异，不 reset；冻结实现前 shader 源码作为基线。遵循 [仓库规则](../../../AGENTS.md)、[shader 规则](../../../shaders/AGENTS.md)、
[scripts 规则](../../../scripts/AGENTS.md) 和[当前设计](light-sampling-design.md)。本轮取代旧权重批次的任务范围，不重启七方法矩阵。

## 已核对机制与最小候选

[LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl) 中，每个 16³ brick 内部先建树，再把根写入
`VOXEL_GRID_BRICKS + brickMorton`。这里的 `brickMorton` 来自相机相对的体素网格；外层按固定 Morton 层次归并。
相机跨过世界坐标 16 格边界后，窗口原点平移，外层分组可能改变，即使可达灯集合相同。
叶子 AABB 已经是世界坐标，仅更改 AABB 坐标表达不能修复这个分组依赖。

候选只改变 **灯树外层 brick 根的槽位**：由世界 brick 坐标低位决定，复用现有稠密外层存储。

```glsl
ivec3 worldBrick = (cameraPositionInt >> 4) - ivec3(VOXEL_GRID_SIZE / 2)
    + ivec3(morton3D_30bDecode(brickMorton));
uvec3 slotCoord = uvec3(worldBrick) & uvec3(VOXEL_GRID_SIZE - 1);
uint lightBrickMorton = morton3D_30bEncode(slotCoord);
// Publish the existing brick subtree at VOXEL_GRID_BRICKS + lightBrickMorton.
```

复用已有的 `brickOrigin` 或整数计算，避免重复工作。负坐标使用无符号低位掩码，不用 abs、截零或有符号余数改变映射。
`VOXEL_GRID_SIZE` 当前是 16／32／64；每轴 G 个连续 brick 对模 G 槽位是双射，因此当前 G³ 窗口不存在槽位冲突。
同一世界灯集合的键和父子成员关系不随相机 brick 改变；该保证不要求原子分配得到的内存地址相同。
讨论端已做 39 组一维整数映射检查，覆盖三种 G、负数、零和大坐标，并验证旧相对键改变邻接而新键不变；这不是 GPU 验收。

| 顺序 | 入口 | 本轮责任 |
|------|------|----------|
| 1 | [Voxelization](../../../shaders/techniques/voxel/Voxelization.glsl)、[VoxelAllocatorMP_Remap](../../../shaders/pass/begin/VoxelAllocatorMP_Remap.comp.glsl) | 保持现有相机相对体素窗口、分配、材料寻址和窗口裁剪 |
| 2 | [LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl) stage 0／1 | 保留逐帧清空所有外层叶槽；brick 内子树不变，仅改变根发布槽 |
| 3 | 同一构建 pass 的外层 reduce | 沿用连续孩子布局、固定归并层级和实际世界 AABB／mass |
| 4 | [LightSampling](../../../shaders/techniques/debug/LightSampling.glsl)、[LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl) | 沿用 Half-space A、四叉、32B、灯内立方体采样及实际 PDF；诊断只补必要缺口 |

不要修改 `voxel_allocatedBricks`、体素 Morton 键或共享体素树来配合灯树索引。保留二／八叉兼容；48B 候选布局也不能被破坏。
不引入永久用户开关、额外节点池、通用排序、H-PLOC、cut、visibility 引导、时间平滑或改变 RNG 来掩盖跳变。
直接改候选源码，用冻结 revision 做前后对照；不保留旧构建实现作为并行产品路径。

## 必须保留的反例和边界

这是一棵按世界坐标周期映射的有限树，并非无限范围的世界 Morton 前缀树。环绕可能降低空间紧凑性：
G=16、窗口的世界 brick x 范围为 `[1,16]` 时，同一 y／z 下 world x=1 和 x=16 映射到槽 x=1 和 x=0，
会在较低层合并，但相隔 240 格。槽位无冲突和时间稳定都不能证明这种分组的采样质量可接受。
必须检查环绕处簇的跨度、重要灯 PMF 和图像质量；如果出现显著退化，保留候选与证据回报，不以“稳定”作为通过理由，
不擅自升级成扩大稠密网格或全局排序重构。冻结窗口或推迟重建只推迟跳变，本轮不用作修复。

现有体素窗口进出、容量不足时的距离优先分配、shadow 提交内容变化仍可能改变灯／遮挡体集合。
本候选不保证这些输入变化时图像不变。固定世界接收点、相同积分输入且 PDF 正确时，单纯改 proposal 不应改变收敛期望；
32 帧观感变化不能直接判为偏差，也不能用这个理论排除真实输入丢失。

## 用户失败场景

2026-10-31 通过 wt4 的 Vibris 列表核对，四项均为原生 1920×1080。保持用户 preset，不改成旧 ls1～ls5 或偷偷降分辨率。

| Preset | SHA-256 |
|--------|---------|
| `light-sampling-failure-moving-1` | `3569c88f4694f5a0e294b6647099fcc3d2f097d97d5d9d3bcaf3c8f0b3059109` |
| `light-sampling-failure-moving-2` | `3096814726434fe24fb637bfe950f7eee02bf85d8a1392504521b201da97cdcf` |
| `light-sampling-failure-moving-3` | `25d9adaab2e85b2e8daf993b301c3438562a5556d09b2f79c20077638ab3034b` |
| `light-sampling-failure-moving-4` | `8133f617323d1587d3d79c472e2849911dfc9579962526e73b6cfc41ec671551` |

开工读取实际 pose／维度／配置，按坐标识别跨界前后，不假定编号就是相邻运动帧。保存相机整数位置、camera brick、窗口原点及实际帧号。
先以旧 A 复现，再比较候选 A；每次固定包含独立 **Cosine BRDF**，它不是 Cosine B。
现有 Vibris action 接口没有连续移动 action；四个静态 pose 的前后对照不冒充连续运动视频或跨界第一帧实测。
若已有其他受支持的连续采集能力，可记录实际能力与时序后使用；不为本轮扩建控制工具。

## 实施和针对性验收

1. **输入与归因。** 复用 [权重分析](../../../scripts/light_sampling_weight_analysis.py) 的可达叶子身份及拓扑递归，记录每个 pose 的世界灯集合、mass、
   遮挡体身份、分配量和溢出。该脚本现有 `voxel_identity` 使用相机相对 brick 编号且硬编码 grid16，不能直接用于跨相机／跨 grid 身份判断；
   本轮诊断需从实际 G、camera brick 和 Morton 解码还原整数世界位置，以世界坐标＋完整材料数据规范排序、比较。
   原始 buffer 地址／字节哈希只用于原件校验。输入不同的 pose 保留结果并分别归因，不混作同输入拓扑对照。
2. **数学与数据检查。** 覆盖 grid16／32／64、负坐标、16 格跨界、模 G 接缝、空／单灯以及二／四／八叉。
   同一固定灯集合在至少两个窗口原点下，要求可达叶集合、mass、语义拓扑和固定世界接收点的完整 PMF 不变（明确浮点容差）；
   AABB 包含后代、孩子范围、PMF 归一化、support 与实取 PDF 仍通过独立审计。仅比较少数随机抽样命中率不足以证明稳定。
   若真实四个 pose 的灯集合都不同，另用一份真实可达灯集合固定输入重建作离线归因，明确不把它称为 runtime 同输入证据。
3. **运行时和图像。** 确认真正加载的 source／config／preset、Target Iris 编译和默认 A。四场景比较旧 A、新 A、Cosine BRDF，
   保持 direct-only、同一采样预算，关闭既有历史／复用／降噪干扰；每项保存 1 spp、32 连续帧均值和原始方差／非有限统计。
   在共同可见的真实表面上匹配世界接收点、法线、材质；不要把两个相机的同一屏幕像素当成同一表面。
   关注原跳变区域及接缝处的保护区域；只在收敛亮度或质量归因仍不清楚时补相应 ROI 的匹配高样本参考，不重做全场景七方法矩阵。
   列明未复现、输入漂移与缺样；已有非有限事件单列，并使用共同完整像素，不能静默清零。
4. **成本和回归。** 核实节点容量／布局未增，完整建树及 initial 加一个稳定 workload sentinel；仅选代表 preset 做相邻正反计时，
   不以全帧代替受影响 pass，不以单次噪声声称加速。编译检查覆盖受影响配置，运行时重点验证默认四叉；额外 smoke 按实际影响补，避免无关矩阵。
5. **交付。** 提交小范围实现及必要诊断、双语状态记录和可复现结果；保存原始 captures／sidecar／source hashes、规范身份／PMF 结果，
   提供按 preset × 方法 × 1 spp／32 帧排列的离线比较页。结果按“复现／机制验证／可见改善／未解决”分别判断，不能只报编译通过。

使用现有 [权重参考](../../../scripts/light_sampling_weight_reference.py)、[近远分析](../../../scripts/light_sampling_nearfar_analysis.py) 和采集设施中仍适用的部分。
原件放入 [light-sampling 构建目录](../../../builds/light-sampling/) 下独立批次，跟踪摘要放入 [实验记录目录](../../../experiments/light-sampling/)。
不覆盖旧批次；仅在确认本地副本和哈希后按匹配 manifest 清理远端原件。长分析／编译用最低进程优先级，计时保持正常条件。
保持 Minecraft 当前实例，不重启、不替换运行时；Vibris durable job 按返回的 wait 契约等待，不为汇报另行 poll。
提交前做 `git diff --check` 和 `git diff --cached --check`；本轮不操作 `2.0/dev` 或 force push。

## 完成后主动通知复查

完成必要验证并保存交付后，调用 `send_message_to_thread`：目标 `01a18c0e-8678-7800-8ab8-04b69f072f12`，host `local`。
通知说明已完成到哪一层，并附实现提交、比较页／结果文件的绝对路径、四个 preset 的结果、同输入拓扑／PMF 判断、环绕质量与输入漂移／非有限限制，
明确写“请现在复查本轮移动稳定性修改”。发生无法继续的真实 blocker 时也回报，不能称为已完成。
回传授权来自本页首段的用户原话；需要验证时读取该讨论 chat 的对应 user turn，而不是把代理间消息本身当作授权。
讨论端收到通知后做独立复查；实验端的完成报告不替代接受结论。
