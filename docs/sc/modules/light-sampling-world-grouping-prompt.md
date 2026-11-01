# Light Sampling 世界坐标分组方案与实验提示词

语言：简体中文 | [English](../../en/modules/light-sampling-world-grouping-prompt.md)

## 当前任务：moving-4 逐灯 M2 与五点修法筛选（2026-11-01）

### 授权、目标与工作区

讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 的人类用户在 turn `01a196cb-8b2c-7843-8c28-6b7a9a9036e3` 明确要求：

> 那就按这个路线继续，写好提示词让实验agent补moving-4两个侧面和一个朝上面的逐灯M2，结合之前near far的两个失败点筛修法，完成后通知你复查。

实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429` 现在负责执行本任务；讨论 chat 负责结果独立复查。上述原始授权同时允许完成／实质阻塞回传，可用 `read_thread` 核对；不是代理消息自行授予回传权限。
使用 `I:/code/mcshaders/Alpha-Piscium-4`（wt4）、`2.0/light-sampling`；chat 元数据中的历史 cwd 不覆盖这个路由。准备时 HEAD 为 `ed4f5d192`，工作区干净；执行前检查后续差异，不 reset 到旧提交。

本轮交付三处指定接收点的真实条件 RGB 均值／M2、与旧 ls1 两点共同组成的候选筛选表、逐灯／分叉归因和后续修法建议。沿[主设计的已选路线](light-sampling-design.md#当前选择与-backlog2026-11-01)，以最新完整世界前缀 A 为研究起点，修跨 brick 粗簇估权。
这是一轮定点补测与离线筛选，允许完成它所必需的 fixture、mode 3 诊断、采集／分析脚本和生成绑定调整。不改生产采样默认，不实现共享 cut，不重开 H-PLOC／visibility，不启动全场景质量／性能矩阵；也不把筛出的候选直接提升为默认。
本节是当前可执行提示词；后面的完整世界分组实施任务已完成，不重新执行其 G16／G64 四景矩阵。

### 三个接收点与旧参考

使用 `light-sampling-failure-moving-4`、G=16、默认四叉、32B／Half-space A、现有立方体灯内采样，沿用原生 1920×1080 以复现指定像素和表面；其余有效设置从已有成功采集读取。
定位源是[本轮方向诊断](../../../experiments/light-sampling/20261101-orientation-review.json)、[完整接收点身份](../../../builds/light-sampling/20261101-orientation-review/orientation.json)及[标注图](../../../builds/light-sampling/20261101-orientation-review/moving4-receiver-points.png)。源采集为 [moving-4/world-a-recapture](../../../builds/light-sampling/20261101-world-grouping/captures/g16/moving-4/world-a-recapture/)。

| ID | 原始像素／显示像素 | 世界采样原点（已含法线偏移） | 几何法线 | material / G-buffer RGB 字节 |
|---|---|---|---|---|
| side-26 | (600,472) / (600,607) | (109.156212,76.138733,362.002991) | +Z | 4651 / (121,85,58) |
| side-27 | (840,472) / (840,607) | (112.391624,76.430634,364.002991) | +Z | 4651 / (89,61,41) |
| top-28 | (1080,472) / (1080,607) | (115.779617,76.003006,365.398621) | +Y | 4651 / (57,113,36) |

必须使用实际 `ls_loadReceiver` 的原点、几何法线、材质和 albedo；不要再次添加 0.003 偏移，不把显示坐标当原始纹理坐标，不换成 shading normal。按已有诊断核对位置误差 <0.005、法线点积 >0.999，以及材质和反射率身份；不悄悄挑换表面。
旧 ls1 参考直接复用 [3921 灯双点 fixture](../../../experiments/light-sampling/20261031-ls1-point-fixture.json)、[两份原生条件矩窗口](../../../builds/light-sampling/20261031-nearfar-iteration/candidate-runtime-evidence/ls1-point-windows/)及[复算脚本](../../../scripts/light_sampling_nearfar_reference.py)。两个原始像素为 (1400,550) 和 (120,700)，均为 -Z，分别是既有柱面／墙面失败点。
核对其原件／fixture 身份及灯内采样、真实评价和颜色空间语义相容后复算；不要覆盖旧 [light_sampling_reference.bin](../../../shaders/textures/light_sampling_reference.bin)，它仍是旧参考分析的真实依赖。旧点保留旧场景条件矩，先在旧树上复现既有 A／near-far 结果，再以同一灯集合离线构建完整前缀树评估目标路线；条件矩本身不依赖选灯树。新旧拓扑间按世界位置／材质和核对过的 mass 映射灯，不按节点数组下标对齐；不得把两场景的逐灯数组、PMF 或 M2 混接。

### 实现与采集边界

复用 [LightSamplingInitial 的 mode 3](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl)、[采集及输入包围检查](../../../scripts/light_sampling_weight_capture.py)、[世界身份重建](../../../scripts/light_sampling_moving_analysis.py)和[旧条件矩参考](../../../scripts/light_sampling_nearfar_reference.py)。读取适用的 shader／scripts AGENTS；若修改 GLSL，使用 opengl-glsl 的相关约束。
当前 mode 3 是两点、每点两个 header texel、随后灯记录的格式；[维护的纹理绑定](../../../scripts/shaders.properties)还写着旧 3925 texel。优先保持现有 shader 通路，用 (side-26, side-27) 和 (side-27, top-28) 两组覆盖三点，重复 side-27 作为跨组一致性检查。新 fixture 使用独立文件和冻结 source 绑定，保留每组的实际尺寸与源码身份，不覆盖旧 ls1 文件。若单次支持三点确实更简单，可做最小诊断适配，但不要新增产品选项、通用框架或兼容格式分支。
fixture 只从本次稳定树的根可达灯叶生成，保存完整位置／材质／mass 身份，不能扫整个保留池把旧副本当成灯；4243 是历史灯数，不应硬编码成所有新采集的必然值。解析尺寸、灯数和每点每灯每帧样本数从实际元数据／fixture／audit 计算，不复制旧 264／3921 常数；尾部未用像素不算条件样本。

先核对实际 shader pack、冻结 revision、preset、effective settings、成功 receipt、相机／接收点和完整世界灯／非发光体素身份。已有可靠双点通路可直接复用；首组结果必须确认 mode 3 receiver-valid、绑定尺寸、样本映射和条件矩身份真实有效，再接受整批。
每组采两个不重叠的连续 32 帧窗口，每窗 8×4 帧 batches；保存 mean、variance、valid count、nonfinite、incomplete／valid batches、帧号、原始 audit、树、G-buffer、相机及前后完整世界体素快照。沿用充分预热；一组两窗及两组之间都核对语义输入。原子分配／缓冲地址变化不等于世界输入变化。
优先复现历史 moving-4 输入。若灯或遮挡已不同但三处表面仍匹配，可自行建立新的稳定 cohort，并重算该 cohort 所有树／候选 PMF；清楚报告与历史的差异，不沿用旧 76% 或 13.44%→1.46% 数值作为新测量。输入在窗口内持续变化、目标表面不匹配或所述失配不能复现时，保留证据回传，不制造成功。
不同组若无法取得相同语义输入，应分别计算其真实输入上的 M2／PMF，明确缺少共同 cohort，不能无说明合并为一次三点比较。可以定向重试受影响组，不扩大为新场景搜索。

mode 3 中固定灯的 `lightPdf=1`，保留真实纹理、立方体条件采样、半球和 voxel visibility；RGB 输出沿用线性 sRGB，alpha 为 `dot(RGB,RGB)`。零贡献是有效样本，不能丢弃后重归一化。
不要套用普通图像的 alpha==1 接收点掩码；此处有效性由 audit receiver-valid 和独立完整性／非有限计数判断。
逐点逐灯报告完整／缺失／非有限样本数；各候选使用同一份有效条件矩。没有有效样本的灯不是零 M2；支持域缺口、trace exhaustion 和输入拒绝必须显式报告。
先在每个条件样本像素的 32 帧统计上核对 `E[||Z||²] = ||E[Z]||² + 31/32 * sum(sampleVarianceRGB)`，再按接收点／灯归并；两个窗口分别推断，再报告窗间差异，不把它当严格置信区间或收敛证明。保留逐灯结果供讨论端独立复算。

### 五点候选筛选

每个候选保持相同灯集合／条件采样，枚举完整实际叶子概率 `q_l` 和空样本概率；不要把 `q_l` 重新归一化为“成功选到灯条件下”的分布。
主比较在新 moving-4 的完整前缀树上进行；同输入原始相机分组 A 作为附加对照，可复用同输入原件或用已经核对过的重建方法获得，明确是 GPU 树还是离线重建。历史输入不匹配时不直接拿旧 PMF 作对照。
旧 ls1 两点也报告同灯完整前缀树上的候选结果，并保留旧树 A／near-far 的已知失败复算；复用已有 [世界布局重建](../../../scripts/light_sampling_world_layout.py)及其 GPU 对照依据，不需要重新采集旧场景。重建未核对时标为缺口，不能把仅在旧树上的改善当作新路线已通过。
固定以下小集合，公式沿用已审阅的 [point_diagnostics.py](../../../builds/light-sampling/20261101-orientation-review/point_diagnostics.py) 与 [nearfar_crosscheck.py](../../../builds/light-sampling/20261101-orientation-review/nearfar_crosscheck.py)；它们是研究参考，不要求原样变成产品框架：

| 候选 | 用途与边界 |
|---|---|
| Half-space A | 每个场景自身的基准 |
| Half-space A＋near/far（model 6） | 旧失败方法的对照，不能先验认为能修新问题 |
| bounded-near/far | 近臂分母 `max(dmin², halfDiagonal²,0.25)`；远臂、1/2 混合及半空间不变 |
| Cosine B（model 2） | 更紧余弦的方向反例对照 |
| 跨 brick 一层展开 | 仅对实际 bounds 跨世界 brick 的内部候选，用下一层 A 权重和替代当前候选权重；叶子半空间判据、mass、归属不变 |
| 跨 brick 两层展开 | 同上，最多继续两层；与已有离线定义一致 |

不扫混合系数、不按朝向或这五个点硬编码修正、不把半球内 mass 等离线 oracle 当廉价生产方案。已有其它消融可引用，但本轮不扩展候选矩阵。
对三个新点和两个旧点、各两个参考窗，计算 `sum(M2_l/q_l) - ||sum(E[Z_l])||²`，清楚标明树版本，并列出相对本场景完整前缀 A 及原始相机 A 的比例。旧树上的候选复算另列，不混作完整前缀结果；不要跨不同场景比较绝对方差大小。
任何测到正 M2 的灯若 `q_l=0`，标记支持失败，不能删灯或省略该项后算有限方差。只统计有贡献灯的命中率不够；按 `M2_l/q_l` 排列主要惩罚灯，给出 PMF、条件矩、关键父分叉与概率变化，检查 y≤0 顶层簇的实际贡献／二阶矩份额是否支持先前几何解释。
若使用 `sqrt(M2_l)` 聚合定位分叉失配，标明它是此有限参考下、固定条件采样的最优叶子概率诊断。区分不可约的灯内方差和树选灯失配，不能把几何代理 J 当真实 M2。
筛选出口是有依据的短名单或“本集合没有兼顾五点的修法”。报告双窗反例、上表面保护结果、旧 ls1 两点变化及新增查询工作量；接近窗间差异的收益记为不确定。无需为完成而选赢家。
受限展开的新增读取／计算可作静态数量说明，本轮不做 GPU 性能矩阵。定点方差改善也不等于全画面、连续移动或最终成本验收，更不能单凭 M2 决定共享 cut 的实现方式。

### 产物、执行权与完成回传

实验 chat 是本轮 wt4 修改和 live Minecraft 采集的单一协调者；继续沿用已有执行者／模型安排，不另开竞争操作者。讨论 chat 派发后不并行改 shader 或控制运行时。
遵守 [AGENTS](../../../AGENTS.md) 与[文档规则](../../AGENTS.md)：PowerShell 7 无 profile，非计时编译／离线长任务低优先级，最新可用 GraalVM；global Gradle 目录故障立即报告，不改用临时 HOME。生成绑定从维护源生成，按实际诊断修改做必要 Target Iris 编译和运行验证。
Minecraft／launcher 不重启；先用安全的非重启恢复，仍不可用则回传 blocker。Vibris 采集使用 durable job/wait，不为汇报额外 poll。收尾确认无存活采集任务，退出 mode 3 并报告实际保留的 pack／revision／preset／配置及未保存改动。
新原件与完整日志放独立 ignored 目录，例如 `builds/light-sampling/20261101-moving4-point-m2/`，不得覆盖历史批次。保存原生数据及必要身份、manifest、派生逐灯数组和可复算脚本；本地副本／哈希核对后才按对应 manifest 清理远端文件。
跟踪一个简洁实验索引、必要 fixture／工具修改和双语结果摘要，链接回本任务；不把大原件或逐文件哈希清单塞进文档。提交前核对双语／链接、`git diff --check` 和 `git diff --cached --check`。本轮不操作 `2.0/dev`、不 force push。

完成采集及五点筛选、自检、提交后，调用 `send_message_to_thread`，host=`local`、目标=`01a18c0e-8678-7800-8ab8-04b69f072f12`，明确写：

> 请现在复查 moving-4 三点逐灯 M2 与旧 near/far 两点的联合修法筛选。

附实现／诊断提交、任务与结果索引的绝对路径、原件和逐灯数组入口、实际输入一致性、双窗候选表、主要反例／支持失败／非有限及未验范围、运行时最终状态。实质 blocker 也回传，不伪称完成。最后由讨论 chat 形成独立复查结论；不以本提示词预设通过，也不只交计划／静态检查后结束。

以下保留已完成的世界前缀实施任务和历史诊断，供定位来源与限制。

## 授权、状态与起点（2026-11-01）

用户在讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 的 turn `01a1961b-a945-7762-9500-333b877eeb35` 明确要求：

> 那就按世界坐标分组来改，写好方案让实验agent实现，再用那几个moving preset复测，完事让它通知你复查。

实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429` 负责本轮实现、针对性复测、提交与完成回传；讨论 chat 保留独立复查责任。
本页取代[上一轮取模提示词](light-sampling-moving-stability-prompt.md)的实施要求，包括旧的“不增加节点预算”限制；旧结果与失败原件继续保留。
工作区为 `I:/code/mcshaders/Alpha-Piscium-4`（wt4），分支 `2.0/light-sampling`。编写时工作区干净，HEAD 为 `fae8c413533d88e988db0b11b31552da1a1f9c48`。
执行前重新核对差异，不 reset。保留该提交新增的 bracketed input 检查和离线派生缓存。
实施起点包含已被质量验收否决的周期分组；它不是本轮接受基线。主对照是冻结的相机相对分组 `6d69b145ec7a1e70ed719886549242e4fd93542f`。

完整世界前缀实现与本轮实验已落地，存在质量退步及 G64 输入限制；状态和证据见文末执行结果。遵循[仓库规则](../../../AGENTS.md)、[shader 规则](../../../shaders/AGENTS.md)、[scripts 规则](../../../scripts/AGENTS.md)与[当前设计](light-sampling-design.md)。

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

## 本轮执行结果（2026-11-01）

实现提交 `d29c5d178`，诊断提交 `0b9c3b08`。世界前缀、分层存储、顶层收尾与单孩子压缩已实现；默认四叉 A、灯内采样、32B／48B 兼容、局部灯预算和实际 PDF 保持原契约。旧取模路径已移除。
[结果索引](../../../experiments/light-sampling/20261101-world-grouping.json)、[比较页](../../../builds/light-sampling/20261101-world-grouping/comparison-upright.html)及[原件核验](../../../builds/light-sampling/20261101-world-grouping/parent-artifact-verification.json)保留来源和哈希。比较页须与同目录的 [comparison-upright.assets](../../../builds/light-sampling/20261101-world-grouping/comparison-upright.assets/) 一起保存。已按原始纹理行序做纵向翻转；旧版未翻转预览作为失败原件保留。

4374 个 CPU 布局对照通过；四组真实 GPU 功能夹具的 19 份树通过完整 84 位独立 oracle，覆盖 G16／32／64、二／四／八叉、32B／48B、跨零、旧接缝、空树清理和单灯根。另有三份无补丁 production smoke 通过：G32 四叉 A／32B、G64 二叉 model 4／48B、G64 八叉 model 5／48B，覆盖真实局部构建与外层归并；这些仅作编译和结构证据。实验关闭配置也通过 Iris 编译。10 份有效 A 树的有序分组、唯一灯叶、bounds／mass 与实取 PDF 审计通过，4 份 BRDF 验证了相同输入。另审计了未接受质量的 G64 第 2／3／4 景前后六棵候选树，结构均匹配 oracle；这不使其质量采集变为有效。

质量有效 14/24 项：G16 四景均有三方法，G64 第一景仅有两份 A；比较页是 28 张有效图和 20 个缺失格，浏览器验证覆盖全部 48 格。使用相同相机、完整世界体素、语义灯、匹配 G-buffer 和共同完整 receiver mask，未按亮度过滤。下表是原始线性 RGB 样本方差之和的全图均值比，低于 1 表示本次 32 帧窗口方差较低。

| G | Moving | 世界 A／原 A | 世界 A／Cosine BRDF |
|---|---:|---:|---:|
| 16 | 1 | 0.0090 | 4.6376 |
| 16 | 2 | 0.4440 | 5.3396 |
| 16 | 3 | 2.5223 | 2.1134 |
| 16 | 4 | 6.6227 | 2.1882 |
| 64 | 1 | 1.1490 | — |

第 3／4 景退步，经固定 4×3 保护 tiles 复核，显示画面的左上 tile 的世界／原 A 比为 4.3460／16.3931；不能将此候选认定为整体质量提升。G16 第 4 景世界 A 的 (777,424) 和 G64 第 1 景原 A 的 (109,73) 各有一个非有限 RGB 像素，保留原始计数并从共同完整 mask 排除；这些坐标以 OpenGL 左下为原点，对应显示坐标 (777,655)／(109,1006)；14 张 single 原件均有限。

G64 第 2／3／4 景的体素池达到 16384，前后真实世界输入变化；一次 1024 帧预热补采仍未稳定，因此九项均未接受。G64 第 1 景 BRDF 自身稳定，但与原 A 对照相差一个世界体素，与两份 A 定向补采相差两个，仍不能配对。失败与排除原件均保留；没有删体素、扩池、改变 preset 或重启游戏来制造成功。

两种 G 各四次 ABBA 的相机、灯、完整体素和分配数量在各自计时 cohort 内一致；quality 与 timing 是分开的 cohort。完整建树逐帧先求各 dispatch 总和再计算分位数：G16 原 A p50 为 0.342 ms，新 A 为 0.409／0.410 ms；G64 原 A 为 1.962／1.958 ms，新 A 为 2.329／2.085 ms。新顶部收尾分别约 0.082／0.146 ms。initial p50 在 G16 原 A 为 1.955／1.964 ms、新 A 为 2.025／2.034 ms；G64 原 A 为 4.086／4.184 ms、新 A 为 4.407／4.217 ms。sentinel 和逐 pass 完整数据见结果索引；没有整帧加速结论。

本轮没有取得同输入的实际跨 pose 配对。真实灯子集的离线移动窗口重建中，新树拓扑及三个完整 PMF 不变，旧树会变；这是机制证据。cycle 是 16 次隔两帧 readback，验证 clear／singleton 但不是连续 16 个 render frame；连续移动及跨界首帧未验。候选和完整记录保留在实验分支，讨论端结论如下。

## 讨论端独立复查（2026-11-01）

结论：完整世界分组机制通过；不接受为替换基线。G16 moving-3／4 的质量退化及本次建树成本增加得到独立确认，G64 质量矩阵仍不完整。
源码检查未发现偏离方案：取模及旧槽号归并已移除，权重和灯内采样公式没有改变。不能继续把本轮退化归因于取模绕回；完整世界分组与现有 A 权重组合仍有质量代价，具体灯贡献／概率失配尚未逐点归因。

[独立脚本](../../../builds/light-sampling/20261101-world-grouping/discussion_review.py)未调用实验分析函数；[复算结果](../../../builds/light-sampling/20261101-world-grouping/discussion-review.json)记录：

- 五份有效质量候选树及三份 production smoke 树，与独立完整 96 位世界 voxel Morton 前缀树一致；该 oracle 同时覆盖 12 位 brick 内层和 84 位上层。检查有序拓扑、唯一叶、AABB 与 mass。
- 14 份有效质量数据的 revision／settings／receipts／32 连续帧核对通过；直接读取 raw，加入完整 alpha、finite、valid count 和 incomplete 检查后，五项共同完整像素数及方差比仍与提交结果一致。已目视检查 moving-3／4 的正向均值图。
- 同 pose／G 的 before／quality／after 共 42 份快照，按格点位置消除原子 allocation 重排后，非空完整材料数据和相机矩阵在配对内一致。未用这个相对网格检查冒充跨 pose 世界身份验证。
- 八份 ABBA 原始逐帧 dispatch 样本重新求和／取中位数，复现建树总成本与顶部收尾耗时。顶部单 lane 实现是后续成本定位点，本轮没有进行优化或新增 GPU 测量。

本轮 G16 四景候选灯数为 8020／8019／4008／4243，已不同于上一轮取模批次；不能将两轮比值直接拼成同输入算法排名。
本次仅提交复查记录，没有回退或修改 shader；候选继续保留，完整矩阵、连续运动／跨界首帧、收敛和非有限原因未验收。

## 朝向退化归因与 near/far 的关系（2026-11-01）

用户指出中间草块的垂直侧面比朝上面退化明显。本轮从保存的 G16 原件继续离线分析，不修改 shader、不采集新 GPU 数据。
本节严格区分以下三个版本；brick 本身在三个版本中都按世界坐标对齐，改变的是跨 brick 的父簇归属。

| 名称 | 实现提交 | 跨 brick 分组 |
|---|---|---|
| 相机分组 A，原始对照 | `6d69b145` | 按相机相对窗口中的 Morton 位置归并，移动窗口可能改变父子关系。 |
| 世界取模 A，上一轮 | `0d1629ae` | 世界 brick 坐标取模 G 后归并；周期接缝可能把实际远隔的灯放进同一簇。 |
| 完整前缀 A，最新 World Prefix A | `d29c5d178` | 按完整世界坐标前缀归并，不丢高位；窗口只参与覆盖和存储寻址。 |

以下“本批世界 A／相机 A”是最新完整前缀与原始相机分组的配对，不是完整前缀与上一轮取模分组的直接比较。
[本轮索引](../../../experiments/light-sampling/20261101-orientation-review.json)、[法线统计](../../../builds/light-sampling/20261101-orientation-review/orientation.json)、
[定点概率诊断](../../../builds/light-sampling/20261101-orientation-review/point-diagnostics.json)及[接收点标注图](../../../builds/light-sampling/20261101-orientation-review/moving4-receiver-points.png)保留可复查入口。

### 实测方向差异

按采样器实际使用的世界几何法线分类，不用 normal map 的 shading normal，也不把画面朝左当作世界轴。
保留各批原有共同完整像素规则：本批要求全部 packed G-buffer 相同，取模批要求几何法线／材质相同；两者都检查 albedo、depth、32 帧完整性及有限值。
八个 G16 cohort 的全图像素数与方差比均复现。下表是本批世界 A／相机 A 的 RGB 样本方差比；它是各方向所有匹配像素的汇总，不表示该方向每块面都一致。

| Moving | +Y 朝上 | +X | -X | +Z | -Z |
|---|---:|---:|---:|---:|---:|
| 3 | 1.013 | 4.402 | 3.252 | 3.801 | 7.128 |
| 4 | 2.271 | 17.000 | 4.141 | 10.286 | 3.443 |

因此“垂直面更严重”有数据支持；moving-4 朝上面仍有退化。moving-1／2 的多种方向则改善，不能推广成固定朝向必坏。
取模批的 +Z 方差比分别为 1.115／1.114／1.087／1.080，-X 为 12.698／14.293／0.081／1.414；说明旧批方向差异确实不同，但灯集合不同，不据此给两种世界分组作同输入排名。

### 中间侧面的具体分支

moving-4 审计点 27 位于草块 +Z 侧面：原始像素 (840,472)，显示像素 (840,607)，世界原点约 (112.392,76.431,364.003)。
该点两棵树的 A 选灯空样本概率均为 0。新树根的一支实际 bounds 为 (-30,-48,240)–(224,0,496)，mass 约 2719.54，获得 **76.071%** 的概率。
这支所有灯都在 y≤0；同一原点仅把法线改为 +Y 时，半空间会完整剔除它；改为任意 ±X／±Z 时，它均获得相同的 76.071%。
这是[同原点法线反事实计算](../../../builds/light-sampling/20261101-orientation-review/normal-counterfactual.json)，不是额外 GPU 表面样本。
根的两个粗簇对四个水平法线的球形余弦上界都饱和到 1，当前权重因而不能体现其中少数近灯的重要性。下方灯是否被地形遮挡属于另一问题，本轮没有假定其真实贡献为零。

同一盏近灯 (110,76,364) 的完整 PMF 从原树的 0.134381 降到 0.014579，另两盏主要几何候选也约缩小 9.22 倍。
在条件采样／真实贡献相同的前提下，每盏这类灯的 `M2_l/q_l` 项相应放大约 9.22 倍；当前没有它们的真实逐灯 M2，不能据几何重要性宣布其精确实测贡献占比。
该点附近 351 个同材质／法线的共同完整像素方差比为约 10.24，仅作局部现象核对。完整 PMF 与既有参考遍历对照，实际 GPU 审计法线与精确轴法线的相对差异小于 4e-6。

### 与旧 near/far 合并讨论的边界

两者属于“粗簇重要性与实际灯贡献不一致，若干分叉累乘后压低重要灯概率”这个共同问题。
本轮使用 A，没有近臂的零距离规则；旧 ls1 的近／远问题包含含接收点 AABB 独占近臂的问题，见[之前的逐灯归因](light-sampling-reassessment.md#概率归因与一次离线修正)。
旧柱面和墙面两个点的法线其实同为 -Z；同方向仍可能需要不同概率修正，因此不能按垂直／水平面硬编码权重倍率。

为区分机制，离线计算 `g_l ∝ mass_l * H_l`，其中 H 是真实单位灯立方体在接收半球内的精确余弦积分，g 归一化；用 `J=sum(g_l²/q_l)` 检查几何分配，理想值为 1。
该代理不含纹理和 visibility，不是真实图像方差。向量化积分与已有逐面积分对照通过。旧 ls1 则继续使用两份不重叠的、保留真实纹理／遮挡的条件 RGB M2 参考窗。

| 离线权重／诊断 | 当前侧面 27 的 J | 旧 ls1 柱面真实 M2 预测方差／A | 旧 ls1 墙面真实 M2 预测方差／A |
|---|---:|---:|---:|
| A | 22.712 | 1 | 1 |
| A＋near/far | 2.431 | 1.806–1.810 | 1.297 |
| 已讨论的 bounded-near/far | 23.306 | 1.425–1.428 | 1.095 |
| 已有 Cosine B | 148.968 | 19.148–19.378 | 0.301–0.302 |
| 仅跨 brick 簇展开一层后求 A 权重和 | 8.356 | 0.836–0.837 | 1.0675 |
| 仅跨 brick 簇展开两层后求 A 权重和 | 6.775 | 0.665–0.666 | 0.9745 |

“跨 brick”由实际节点 bounds 是否覆盖多个世界 brick 判断；没有改变归属、mass、半空间或灯内采样。展开仅是离线查询对照，未开启运行时前瞻。
当前另外两个 +Z 侧面代理也改善，但保留相机树／其它接收点的反例：更多展开并非处处更好，一层展开在旧墙面仍退步。
将粗簇 mass 按半空间区间比例缩小，或精确改为该半球内叶子 mass 后再套 A，都使当前点的代理更差；“剔除更多”不是通用修复。
[旧逐灯交叉检查](../../../builds/light-sampling/20261101-orientation-review/nearfar-crosscheck.json)保留两窗结果。没有依据直接恢复 near/far 为默认，或把任一离线代理收益当作整场景验收。

### 当前修正方向与待补证

推进路线已收敛为最新完整世界前缀 A 上继续修复粗簇重要性退化，见[主设计的当前选择](light-sampling-design.md#当前选择与-backlog2026-11-01)。取模候选不再继续迭代；补逐灯条件矩用于在此路线内选择修法。
保留“世界坐标决定父簇、相机窗口只决定覆盖和存储”这个稳定性契约及 Half-space A；以最新完整前缀实现为当前研究起点，不等于接受其质量或锁定现有前缀树为最终结构。
修正对象是跨 brick 粗簇的接收点重要性估计，减少单个大 AABB 直接决定高层概率的影响。
已确认的是实测方向退化、完整 PMF 变化及半空间对各方向的不同作用；76.071% 是分支选择概率，不等于已证明这部分样本全部无效。
建议补采的范围是 moving-4 侧面 27／26 与朝上面 28。固定各接收点，逐灯强制执行现有灯内采样和真实纹理／遮挡评价，记录贡献均值及平方均值 M2；已有 ls1 两点参考可复用。
这些条件矩与候选实际选灯概率结合，能筛查改权重／细化簇后的真实定点方差，补的是修法依据，不是重新证明已观察到的退化，也不要求重跑完整场景／性能矩阵。
受限展开是已做的离线定位对照。M2 可检验概率分配是否改善，不能单独决定逐点展开与共享上层 cut 的成本取舍；后者还需要评估共享近似、额外读取和 GPU 时间。
共享不能让相机槽号重新决定分组／支持域。
展开每个内部候选会增加后代节点读取与权重计算；两层在四叉树上最多涉及 16 个后代位置，本轮没有测它的 GPU 成本，也未决定重新启用逐样本前瞻。
更紧的余弦、near/far 或额外空间摘要均不能仅凭本轮方向现象直接定案。上述诊断提交时只包含记录和离线结果，shader 未改；用户随后授权的逐灯补测按页首当前任务执行。
