# Light Sampling 世界坐标分组方案与实验提示词

语言：简体中文 | [English](../../en/modules/light-sampling-world-grouping-prompt.md)

## 授权、状态与起点（2026-11-01）

用户在讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 的 turn `01a1961b-a945-7762-9500-333b877eeb35` 明确要求：

> 那就按世界坐标分组来改，写好方案让实验agent实现，再用那几个moving preset复测，完事让它通知你复查。

实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429` 负责本轮实现、针对性复测、提交与完成回传；讨论 chat 保留独立复查责任。
本页取代[上一轮取模提示词](light-sampling-moving-stability-prompt.md)的实施要求，包括旧的“不增加节点预算”限制；旧结果与失败原件继续保留。
工作区为 `I:/code/mcshaders/Alpha-Piscium-4`（wt4），分支 `2.0/light-sampling`。编写时工作区干净，HEAD 为 `fae8c413533d88e988db0b11b31552da1a1f9c48`。
执行前重新核对差异，不 reset。保留该提交新增的 bracketed input 检查和离线派生缓存。
当前 shader 仍包含已被质量验收否决的周期分组；它不是本轮接受基线。主对照是冻结的相机相对分组 `6d69b145ec7a1e70ed719886549242e4fd93542f`。

已完成源码核对和 CPU 布局推演，尚未实现新 shader，也没有新的 GPU 收益结论。遵循[仓库规则](../../../AGENTS.md)、[shader 规则](../../../shaders/AGENTS.md)、[scripts 规则](../../../scripts/AGENTS.md)与[当前设计](light-sampling-design.md)。

## 要实现的契约

- 每个 brick 仍是世界对齐的 16³ 区域；brick 内灯树、体素寻址／裁剪／分配保持原责任。
- **所有跨 brick 父簇按完整世界坐标的固定层级定义。** 窗口坐标只决定哪些节点需要存储及其数组地址，不决定父子关系。
- 默认四叉、Half-space A（model 1）、32B 节点、灯内采样、实际 PDF 累乘不变；保留二／八叉及原有 48B 模型兼容。
- 同一灯集合及相同灯属性，在不同可容纳它的窗口中得到相同的有序非空拓扑和固定接收点 PMF（含空样本概率）。地址可变。
- 输入灯集合／属性改变会合法改变质量和概率，不承诺窗口裁剪变化下的完全不变。
- 删除周期键和旧的按槽号直接归并路径；用冻结源码作对照，不添加构建模式开关、双产品路径或隐藏 fallback。

本轮不改权重、不启用 near/far、visibility、lookahead、H-PLOC、区域 cut 或 ReSTIR；不冻结窗口或用时间平滑掩盖跳变。

## 已选模型：完整世界前缀、有限分层存储

选择逐层世界对齐归并。它使用现有规则网格，首版不需要全局排序、哈希表、通用稀疏树框架或 GPU 64 位整数。
这些是本轮不需要增加的机制，并不意味着世界坐标可直接当作数组下标。

令 `a` 为叉数，`k=log2(a)`，`g=log2(G)`。level 0 是 brick 根；向上每层消费 Morton 顺序中的 k 位，最低位顺序为 x、y、z 循环。
level l 消费 `b=k*l` 位，各轴的移位数为：

```text
s(l) = (floor((b+2)/3), floor((b+1)/3), floor(b/3))
worldBrick = floor(worldBlock / 16)
u = worldBrick + 2^27
cell(l) = u >> s(l)          # 各轴分别移位
```

当前有符号 32 位 block 寻址产生 28 位有符号 brick 坐标；固定偏移将其双射到每轴 28 位无符号范围，保留高位和负坐标。
逻辑世界 Morton 键有 84 位，但实现只需三个 32 位分量及分层移位。不要把完整世界坐标交给仅保留每轴 10 位的 [morton3D_30bEncode](../../../shaders/util/Morton.glsl)。
偏移是全局常量，不跟相机移动；位移不超过 28。这个整数分组契约不扩大现有浮点世界位置的精度保证。

父簇身份是 `(level, cell(level))`。父子归属和孩子次序取完整世界位；只在给存储寻址时减去该层覆盖区间的最小坐标。
节点 AABB／mass 继续由实际后代合并，不能用空白世界单元的几何范围代替实际灯包围盒。

## 可直接实施的数组布局

设当前窗口最小世界 brick 对应偏移坐标为 `uMin`，`uMax=uMin+G-1`。每层：

```text
lo(l) = uMin >> s(l)
hi(l) = uMax >> s(l)
dim(l) = hi(l) - lo(l) + 1
index(l,c) = r.x + dim(l).x * (r.y + dim(l).y * r.z), r=c-lo(l)
```

取 `L=ceil(3*g/k)`，此时每轴覆盖至多两个世界单元，最上层存活记录至多 8 个。
level 1..L 各预留一段数组：**每个父单元预留 a 个连续孩子槽**。父单元使用上面的局部线性编号；孩子槽内次序使用这层消费的 k 个完整世界位。
因此同一父簇的孩子天然连续，但父簇的身份不会被数组原点改变。不要再次对世界身份取模。

令 `offset(l)` 是 level l 父单元的孩子槽段起点。level l-1 的节点存储地址为：

```text
offset(l) + a * index(l, cell(l)) + childOrdinal(l)
```

`childOrdinal(l)` 是全局 Morton 位 `[k*(l-1), k*l)`，按低位到高位收集，范围为 `[0,a)`。
每层消费的轴可能不同；四叉不能偷换成每层 x/y 都右移一次。最上面的 level L 结果另存到至多 8 个连续临时记录。

外层数组前 32 个节点预留：root 仍为 1，level L 临时记录可用 2..9，顶部合并的孩子记录可用 10..23。
一个额外的 workgroup 将这至多 8 个记录沿剩余的完整世界前缀归并到根；每组按 child ordinal 排序，**不能直接按临时数组顺序凑四个**。
可以由一个 lane 完成这个小规模归并；有运行时索引的数组放 shared memory，遵循 shader 规则。
继续到共同祖先或世界根（总计 84/k 层）；不在 L 层强行把所有记录接成一个四叉根。

所有层：0 个非空孩子写空节点；1 个非空孩子直接复制其完整节点以压缩单孩子链；至少 2 个才生成分支。
下部可保留固定 a 个槽及其中的空孩子，顶部按非空数连续写孩子。8 个输入在压缩后至多产生 14 个顶部孩子记录，32 个预留槽足够。
单孩子压缩保证大坐标／跨零点不会在每个 shading sample 增加大量概率为 1 的遍历层。

### 容量与生命周期

每轴的覆盖上界为 `D(l,j)=ceil((G-1)/2^s(l,j))+1`，`N(l)=product(D(l))`。
各段按上界预留，用实际 `dim` 寻址；所有这些上界仅用于容量，不能用于改变世界父簇身份。

```text
outerCapacity = 32 + a * sum(N(l), l=1..L)
offset(1) = 32
offset(l+1) = offset(l) + a*N(l)
localBase = outerCapacity
SSBO bytes = 32 + (outerCapacity + 262144) * nodeStride
```

| G | 二叉 outerCapacity | 四叉 outerCapacity | 八叉 outerCapacity | 原外层预留 |
|---|------------------:|------------------:|------------------:|-----------:|
| 16 | 10570 | 7620 | 7144 | 8192 |
| 32 | 74464 | 51732 | 46448 | 65536 |
| 64 | 558854 | 380432 | 333944 | 524288 |

这是布局推导值，不是性能测量。按所选 G／叉数分配，保留 262144 个 brick 内节点的现有预算及溢出报告；不要为隐藏丢灯而扩大灯预算。
32B／48B 两种 stride 的 [SSBO 配置](../../../scripts/shaders.properties)、[GLSL 布局](../../../shaders/techniques/debug/LightSampling.glsl)与构建必须一致。
每帧 clear 整个外层预留区，保证窗口边缘 padding 和先前帧写过的节点不会泄漏；brick 内节点依旧只读取当前帧可达部分。
阶段之间使用已有 dispatch 顺序与可见性规则；无需跨 workgroup 自旋等待。顶部收尾必须先于 sampling 执行。

## 实施入口与交付顺序

| 顺序 | 来源／消费者 | 本轮差额 |
|------|-------------|----------|
| 1 | [LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl)、[LightSampling](../../../shaders/techniques/debug/LightSampling.glsl) | 加入世界前缀／分层地址计算；改 brick 根发布位置、clear 范围和外层归并；保持局部构建和采样公式 |
| 2 | [programs.main.kts](../../../scripts/programs.main.kts)、[shaders.properties](../../../scripts/shaders.properties) | 注册 lower levels 与最后的小规模 root 收尾，修改 buffer 容量；移除旧 reduce 接线并生成 wrappers／properties |
| 3 | [moving analysis](../../../scripts/light_sampling_moving_analysis.py)、[weight reference](../../../scripts/light_sampling_weight_reference.py) | 更新真实树解析／重建；特别移除新布局下遍历 `[G^3,2*G^3)` 即得到 brick 根的假设；显式区分历史取模算法与新完整世界前缀算法 |
| 4 | [capture](../../../scripts/light_sampling_weight_capture.py)、[presets](../../../scripts/light_sampling_weight_presets.py) | 沿用实际源码、配置和前后输入验证及原件保留；不要调用仍假定二叉的旧诊断来验四叉 |
| 5 | [实验记录目录](../../../experiments/light-sampling/)、[构建产物目录](../../../builds/light-sampling/) | 独立批次 `20261101-world-grouping`；双语状态、原件、对比页、完整成本与完成通知 |

命名和 helper 位置可由实现者决定；保持模块内、直接的数据流，不新增通用构建框架或配置面。
从 scripts 运行 `kotlin options.main.kts` 生成受维护源对应输出；只有确需 bootstrap 或改 atlas 时才运行 Shadesmith。
使用当前可用的最新 GraalVM JDK，PowerShell 7 不加载 profile；长编译／离线分析低优先级，benchmark 正常优先级。

## 先验检查与需要补的证据

讨论端的 [CPU 布局推演](../../../builds/light-sampling/20261101-world-grouping-design/layout_reference.py)与[结果](../../../builds/light-sampling/20261101-world-grouping-design/layout-reference.json)比较了 4374 个布局：
162 个固定输入 cohort，各 27 个窗口位置，覆盖 G=16/32/64、二／四／八叉、空／单灯／多灯、负坐标、零、网格边界、超过 10 位 Morton 范围及大坐标。
对照是 Python 整数完整 84 位键的独立前缀树，非复用布局寻址。四叉下 world brick x=1 与 x=16 在未压缩层级的第 7 层才合并，而周期截断会在第 1 层合并。
这些只验证整数分组、边界地址和容量；实现者仍须独立验证真实 GLSL／GPU、质量和成本。

必要验收：

1. 独立 oracle 对新源码导出的世界分组、空／单孩子压缩和完整有序拓扑成立；覆盖窗口边界、旧模 G 接缝、负坐标／跨零、G32 四叉层数、空／单灯、连续孩子范围及容量。
2. 每盏灯恰好可达一次，AABB 包含后代，mass 守恒，无 overflow／丢灯；实取 PDF、灯内 PDF、完整 PMF 与空样本概率通过独立审计。相同输入在移动窗口间比较语义而非 buffer 哈希。
3. 清空后移入／移出区域不会留下旧节点；零灯帧、单灯帧和历史重置有效。float mass 比较记录容差，不能以原子地址差异当分组变化。
4. Target Iris 编译／最小运行覆盖 G16/32/64、二／四／八叉、32B／48B 及实验关闭，选择最少组合覆盖分支；全图质量只跑默认四叉 A。

## 四个 moving preset 的复测

2026-11-01 已用 Vibris 重新读取四项，SHA 与原轮一致，原生 1920×1080；使用用户指定的原 preset，不修改相机或 preset 文件。

| Preset | SHA-256 |
|--------|---------|
| `light-sampling-failure-moving-1` | `3569c88f4694f5a0e294b6647099fcc3d2f097d97d5d9d3bcaf3c8f0b3059109` |
| `light-sampling-failure-moving-2` | `3096814726434fe24fb637bfe950f7eee02bf85d8a1392504521b201da97cdcf` |
| `light-sampling-failure-moving-3` | `25d9adaab2e85b2e8daf993b301c3438562a5556d09b2f79c20077638ab3034b` |
| `light-sampling-failure-moving-4` | `8133f617323d1587d3d79c472e2849911dfc9579962526e73b6cfc41ec671551` |

- 分别比较 **G=16 历史控制规模**和 **G=64 实际默认规模**，每种 G 下四景均含“原相机相对 A／新世界前缀 A／独立 Cosine BRDF”。G32 仅布局与 smoke，不扩大全图矩阵。
- 取模候选留作已有失败证据，不新增第四方法全图矩阵。只有同输入归因需要时离线重建它；不同轮、G 或输入不可混为同一统计 cohort。
- 质量沿用这四个指定 preset 的原生分辨率，以衔接原图；保持 direct initial-only，关闭 RC／reuse／ReSTIR／denoising／TAA／visibility／lookahead，固定设置、采样预算、warmup 和曝光。
- 每项保存 1 spp、32 连续帧均值、原始方差、valid count／nonfinite／incomplete；使用共同完整像素掩码，不暗中清零或裁掉退化区域。
- 读取实际 effective settings、source revision、scene／preset hash、通过的 action receipts；保存采集前后相机／树／完整世界体素身份。世界身份须用实际 G 和相机重建，不能直接比较相对槽号。
- 更新采集脚本的布局理解，但保留 `fae8c4135` 的 bracketed input 和派生缓存校验。跨 revision、配置或输入的缓存必须失效；不要覆盖旧原件或把输入漂移样本混入配对。
- 逐景报告全图及既有保护 ROI；匹配同一世界接收点、法线、材质和反射率，报告跨 pose 的完整 PMF、空样本概率与同表面统计。
  旧 G16 pose 2→3 曾同输入，G64 必须重新证明。若新实测所有 pose 输入都变化，用真实灯集合固定输入重建补机制证据，并明确其为离线结果。
- 四个静态 pose 不等于连续运动视频或跨界首帧。只使用现有受支持能力；无连续采集能力时如实保留限制，不扩建运行时工具。
- 两种 G 各选一个代表 preset，原 A／新 A 做相邻正反（ABBA）计时；前后输入一致。报告 clear、全部 lower build、root 收尾的完整建树总成本，以及 initial 和稳定 workload sentinel；不以单个 kernel 或全帧替代。
- 新出现的明显退化可针对相同输入补一次窗口／ROI 核对；若确认是方案本身的质量代价，保留候选回报，不自动改权重、开启 H-PLOC 或扩展搜索矩阵。

保留独立 `grid × preset × method × 1 spp/32-frame` 比较维度和可缩放原图，提供本地 HTML 对比页及可复现结果索引。
不要把 PDF 审计通过、拓扑稳定或更紧的包围盒单独称为质量通过；结论分为实现／机制／图像质量／成本／未验部分。
Minecraft 不重启；运行时不可用时完成独立可做部分并报告 blocker。按 Vibris durable job 契约等待，不为状态汇报额外 poll。
仅在本地原件副本与哈希已核对后按匹配 manifest 清理远端原件。提交前做 `git diff --check` 和 `git diff --cached --check`；不操作 `2.0/dev` 或 force push。

## 完成出口与回传

完成实现和必要验证，不以计划／静态通过收尾。提交范围内源码、生成输出、必要诊断和中英记录，保留旧失败证据。
交付实现提交、实际运行状态、四景两种 G 的结论、完整成本、比较页、原件与输入／PDF／拓扑审计索引；没有通过的部分明确写出。

之后调用 `send_message_to_thread`，目标 `01a18c0e-8678-7800-8ab8-04b69f072f12`，host `local`，明确写：

> 请现在复查本轮完整世界坐标分组修改。

通知附提交、结果索引与比较页的绝对路径，以及质量回退／非有限／输入变化／未验限制；真实 blocker 也回传，但不能称已完成。
回传授权来自本页首段的用户原话与对应 turn，可通过 `read_thread` 核对，不是代理间消息自行授予权限。
