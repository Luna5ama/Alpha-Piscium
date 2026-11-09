# Light Sampling 聚类修复实验提示词

语言：简体中文 | [English](../../en/modules/light-sampling-clustering-repair-prompt.md)

## 当前任务：红柱贡献失配分解（2026-11-08）

当前状态：实验提交 `f100765c7` 的离线部分已独立复查，见[因素诊断复查记录](light-sampling-design.md#红柱因素诊断复查2026-11-08)。均值／灯内波动分解、归档 RGB 变化及受限 CPU 遮挡证据成立；完整 actual/neutral × visible/unblocked 配对及执行态 PDF 仍未采集，受当前 Vibris 原件归档额度限制。本任务没有全部完成，也没有新修复基线；下文保留剩余采集契约，不自动重启旧聚类矩阵。

本轮只回答：**红柱的重要灯为什么与现有几何／通量估权失配，遮挡、真实发光和灯内采样分别提供了什么证据？** 不先指定主因，不要求交付修复候选。此节替代下文已经完成的准则筛选及未执行的 GPU 扩展；不重跑 14 个构造，不改聚类、Half-space A + Near/Far（model 6）、前瞻、K、32B 节点或默认，不建立探针／接收响应缓存。现有固定接收点只用于诊断，数据不进入建树。

用户本次原话：“那就先查清红柱的失配到底来自遮挡、实际发光还是灯内采样，写好提示词让实验 agent 做，完事通知你复查。前瞻和 K 先不动，也先别急着加探针。”沿用已指定的实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429` 与讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12`，host=`local`，模型与 effort 不变。该人类授权包括本轮诊断、必要的最小补采和完成后回传复查。讨论端维护双语设计／提示词；实验端实现诊断、采集和结果索引。

### 起点与原件

使用 wt4 / `I:/code/mcshaders/Alpha-Piscium-4` / `2.0/light-sampling`，准备起点 `024cc3366`。执行前检查最新提交、未提交改动及共享 runtime 所有者，不 reset 到旧快照。阅读[当前诊断依据](light-sampling-design.md#新准则筛选结果与下一步边界)、[实验索引](../../../experiments/light-sampling/20261108-clustering-repair.json)的 `clustering_cost_independent_review`、[全叶独立复核](../../../builds/light-sampling/20261108-clustering-repair/review/flat-reference-independent.json)。

- 复用[新接收点紧凑参考](../../../experiments/light-sampling/20261108-clustering-repair-reference.npz)，SHA-256 `3317ac116bc89c055528f9fb8038fb7bd791ffe27821aced8bd75a9d92167e0`；完整 ls1 [M2 原件](../../../builds/light-sampling/20261108-clustering-repair/cases/light-sampling-1/m2-1f1d84ea/)来自 fixture 提交 `6a8ec6234c93538baa648a2ddb5bb8edbdbe2afc`。
- 本 cohort 有 10,235 盏灯；灯集合 SHA `ce3347600e2d804aeaea14082eeb84a971dce2f2154797c7f1ba52f0795d0631`，世界 voxel SHA `4dfca2b6ba4714126b4962a206a4bda14e358bf9a33eaa67ac85bfb99e9d61df`。它与旧整图 grouping-control 的遮挡快照不同；不能只凭灯键相同混用 M2／画面。
- 主点 `left-red-pillar`：raw `(250,679)`，display `(250,400)`，1920×1080；采样 origin 约 `(158.996994,151.341949,83.529083)`，法线朝 −X，材质 `10938`。保留原中央顶面 raw `(1000,909)`、display `(1000,170)`、朝 −Y、材质 `8401` 为同场景控制。精确 origin、normal、albedo 与两窗身份以参考和 receiver audit 为准；不换回旧边缘点。
- 现有 [mode 3 条件评估](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl)固定 `lightPdf=1`，输出线性 sRGB 的 RGB 与 `dot(RGB,RGB)`；[提取器](../../../scripts/light_sampling_point_moments.py)保留均值、M2 和完整样本数。它没有保存逐样本 reason、材质发光、灯内 PDF 的联合记录，不能从零 RGB 反推遮挡原因。

### 先复用 M2，定位需要分解的灯

只比较原 Prefix、最接近的 top-down `upper/flux-diagonal`（无 Kr、保留原 brick 子树）及 flat 全叶参考，固定实际 A + Near/Far PMF。复用[分层输入](../../../scripts/light_sampling_clustering_hierarchy.py)、[准则构造](../../../scripts/light_sampling_clustering_rule_screen.py)和[已有矩分解实现](../../../scripts/light_sampling_point_screening.py)，不新建通用实验框架。flat 同时改变归一化域和混合层次，仍不是最优 PDF 或单独的 AABB 消融。

对每个窗口记 `mu_i=E[X_i]`、`M2_i=E[||X_i||²]`、`s_i²=M2_i-||mu_i||²`；`X_i` 是固定选中灯 i 后、除以灯内 PDF 的贡献，**不含选灯逆概率**。核对：

`V(q) = sum_i(||mu_i||²/q_i) - ||sum_i(mu_i)||² + sum_i(s_i²/q_i)`。

相对 Prefix 的差额为 `sum_i(M2_i*(1/q_i-1/qPrefix_i))`，分别列出均值项与灯内波动项的正、负、净差额，以及累计占比。灯内波动已经包含随机方向下的遮挡和纹理变化，不能据此直接宣布 cube sampler 有问题。保留两窗，说明有限样本下 `||mu_hat||²` 的偏差；用独立窗口交叉估计或原批次统计检查结论是否稳定，不靠截断负值掩盖误差。

逐灯表至少保留完整世界键、材质、mass、三套 q、near/far 份额（flat 两半及树上关键分支，不把逐层混合误作两棵整树的 50/50 混合）、mu、M2、s²、差额及参考最优份额 `sqrt(M2_i)/sum(sqrt(M2))`。主灯既包括 flat 的三盏净增量来源，也包括上层候选的反例：

| 世界键 `(x,y,z,material)` | 保留理由 |
|---|---|
| `(154,146,78,10254)` | flat 第一主项，原概率约 4.104% → 1.724% |
| `(150,146,78,10254)` | flat 第二主项，约 2.280% → 1.279% |
| `(157,145,76,682)` | flat 第三主项，约 1.297% → 0.541% |
| `(146,153,88,682)` | 原全局／上层反例；flat 反而提高其概率，不能与上三项混为同一失败 |

按两种候选各自的差额补少量必要对照：实际抢走概率的低贡献灯、同材质／相近几何的灯及负差额灯。报告选取依据和未覆盖差额；不只挑成功命中或某个材质。若现有统计足以排除某项，保留证据；其余再补缺失字段。

### 缺失因素的最小配对诊断

沿真实调用核对 [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl)、[cube 面选择与 PDF](../../../shaders/techniques/debug/LightSamplingCube.glsl)、[建树 mass](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl)、[体素表面读取](../../../shaders/techniques/voxel/SurfaceData.glsl)、[材质解码](../../../shaders/util/Material.glsl)和[体素追踪](../../../shaders/techniques/voxel/VoxelTrace.glsl)。当前 mass 是 hardcoded emissive 经曲线／倍率后的标量；实际 `surface.material.emissive` 还依赖纹理、PBR 分支、发光颜色／亮度曲线及色彩空间。区分灯源差异与红色接收面 albedo 对 RGB 方差的影响，不能把这两者都叫“发光强度”。

优先从已保存字段恢复；恢复不了时，可在隔离的诊断来源中扩展现有 mode 3，不新增面向用户的设置或生产路径。只补这两个固定接收点。保留同一个随机方向／面样本，同时记录实际贡献和以下 2×2 反事实；额外诊断不得消耗或改变原评估的 RNG 序列。

| 发光值 | 实际目标可见性 | 去掉中途遮挡 |
|---|---|---|
| 目标表面实际 RGB 发光 | 原始 X，必须与原评估核对 | 同一目标、面、UV、PDF 下的未遮挡贡献 |
| 建树 mass 对应的恒定中性发光 | 去掉面纹理／颜色和材质亮度失配，保留遮挡 | 仅几何及当前灯内 proposal 的控制 |

中性发光的 RGB 空间及统一比例必须明确，所有灯沿用原 mass，不按结果拟合逐灯尺度；接收点 albedo 保持不变。每格用自己的均值和 M2 计算相对 Prefix 的方差／差额，原 q 保持不变。反事实改变了积分目标，不能拿它与原始物理图像直接比较绝对噪声，也不能把两个因素的降幅当成可相加的因果百分比；报告交互和残差。

**未遮挡目标发光必须在被挡样本上也能计算。** 使用所选目标 cube 的正确入口、法线、材质、UV 和原材质解码；不能读取第一遮挡体的发光，也不能把被挡样本的目标发光填零。用实际可见样本验证目标重建与真实命中的面／UV／发光一致。cube 几何无有效采样域仍保持原语义，不能把几何失败改成可见。若目标重建无法核实，明确缺项而不是伪造未遮挡参考。

完整逐灯紧凑矩可复用现有 mode 3 的遍历；逐样本详情只需主灯和必要对照，避免全场景大转储。至少能区分无有效 cube 面／半球域、采样或 cube 入口失败、正常挡住、无命中、先命中另一发光体／目标身份不符、到达目标但实际发光为零、有效贡献，以及 trace exhausted／非有限值。记录必要的目标与实际命中、面、UV、`cosine`、面选择概率、完整灯内立体角 PDF、发光 RGB、贡献和原因。没有完整逐灯反事实时，只报告覆盖灯的差额与未解释部分，不能宣称全局方差比例。

### 灯内采样的判据与有限样本边界

对主灯和必要对照复核 PDF 是否包括面选择概率及面积／立体角转换，是否触发 tiny-sliver 面积 fallback、是否有非有限或失去支持。使用简单独立积分或独立参考采样检查恒定中性发光、无遮挡时的积分／PDF；这是诊断参考，不替换运行时 sampler。分别报告几何因子 `max(dot(n,w),0)/(pi*pOmega)` 的波动、实际发光变化、可见率及贡献尾部。

对理想的无采样噪声灯内积分，离线把每灯 M2 换为 `||mu_i||²`，只回答灯内波动可解释多少；它不等价于实现了更好的廉价 sampler。当前 lamp-optimal PMF 仍按 `sqrt(M2)`，不能换成通量或均值份额冒充同一最优目标。高灯内方差、材质代理失配、遮挡不均和 PDF／实现错误需分别下结论；允许多个因素共存。

沿用两个不重叠 32 帧窗口与原完整样本协议，保留每灯有效／缺失数和零样本。若主灯靠罕见贡献决定结论，可只对这些灯补独立样本并报告不确定性；不因零观测宣称永久不可见。trace exhausted／非有限值不可当零。先确认一份新诊断原件的字段、身份和原 X 复现，再扩大必要样本。

### 输入有效性、交付与回传

新采集必须核对实际 shader pack、完整源码、fixture、编译成功、场景／相机／接收点、G／渲染距离、叶灯／通量、世界 voxel、PBR／纹理与有关设置；旧文件未存的身份明确列为缺项。以原 1920×1080 固定像素协议为准。若无法恢复原遮挡／材质 cohort，在同一新快照重做本轮实际与反事实控制并重新求三套 q，单独记录；不能把新因素乘到旧 M2 上。不要悄悄换点或移动相机寻找有利结果。

实验端可改必要的诊断脚本及临时采集来源，保留可复算源码／补丁／refs；普通评估和候选公式不变。新增记录放入[实验索引](../../../experiments/light-sampling/20261108-clustering-repair.json)的独立 `red_pillar_factor_diagnosis` 字段，保留旧记录及紧凑格式。原件放在[当前 run](../../../builds/light-sampling/20261108-clustering-repair/)的独立子目录，交付输入／源码哈希、完整逐灯统计、主灯配对明细、复算命令及短结论。无需新增整图、moving／ls5、性能或 BRDF-only 矩阵；本轮不是画面质量或性能验收。

先回答哪项有直接证据、哪项被排除到什么范围、交互／未解释差额和下一步最小修复方向；发现真实 PDF／采样实现错误时给复现和定位，不把修复混进本轮对照。无需等待额外阶段批准即可完成上述诊断及必要补采，但不自动实施探针、响应聚类或其他新方案。

实验端独占本轮共享 runtime 操作；PowerShell 7 无 profile，耗时非 benchmark 工作 Idle；如用 Pyxis，沿用昂贵执行前核对有效 priority tier 的约定。不得重启 Minecraft／launcher；无重启路径仍不可用时带已有离线结果和缺口回报。已有清理暂停及自动删除拒绝继续有效，不重试删除或换工具绕过。保留新必要原件至复查，注明路径／用途／回收条件，保护无关改动。提交前完成适用的 Target Iris 诊断编译／原评估核对及两个 diff 检查；没有 GPU 工作则明确说明。

完成或遇到实质阻塞后，停止提交新 runtime 作业。恢复本轮接手时已核实的普通评估来源与配置；若不能安全恢复则明确停留状态，不将诊断当基线。核验并交还实际来源、场景、配置、冻结状态和活动任务。使用 `send_message_to_thread` 向 host=`local`、讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 发送结果入口、提交／refs、原件、结论和未验范围，明确“请复查红柱因素诊断”。回传依据为本节可核对的人类授权；讨论端负责独立复查和后续路线决定。

## 聚类准则再筛（2026-11-08）

本轮已完成并复查；六规则 × 两种范围及 bottom-up 两项共 14 个候选均未达共同保留门槛。全叶归一化补充诊断也已交叉复核，结果与下一步提议见[设计记录](light-sampling-design.md#新准则筛选结果与下一步边界)。以下保留执行规格；没有新的 GPU 阶段或已接受基线。

上一轮要求是继续研究能同时保住红柱、顶面和 moving 侧面的聚类方案，前瞻与 K 不动。本轮从 `906982dce` 的分层复查继续，范围为**离线准则筛选与路径诊断**；不恢复旧提示词的 GPU 扩展、动态构建、运行时或默认修改。Near/Far 继续固定为 model 6，原件与清理限制照旧。

研究假说：当前面积代价没有直接约束采样器余弦界所用的包围盒对角线；另一方面，树的空间／通量分配失衡可能使重要小簇在多层连续失去概率。这些是假说，不是已确认原因。[Lightcuts 的全方向聚类度量](https://www.cs.cornell.edu/~kb/projects/lightcuts/lightcuts.pdf)使用通量乘包围盒对角线平方，可作为尺度代价的依据；这里测试的是同类度量的 top-down 划分，不称为原论文 bottom-up 算法的复现。

以现有 [SAH 构建和评分脚本](../../../scripts/light_sampling_clustering_repair.py)、[分层输入与接口](../../../scripts/light_sampling_clustering_hierarchy.py)和三份已核对的 M2 紧凑参考为起点。保留原 Prefix、原全局面积 SAH 和原只改上层结果作为控制。只筛下面六项，不按结果扫描参数、平移网格或挑轴：

| 规则 | 固定定义 |
|---|---|
| flux-diagonal-Kr | 原 12 桶、三轴候选；代价改为 `Kr * (PhiL * dot(dL,dL) + PhiR * dot(dR,dR))` |
| flux-diagonal | 同上，去掉 Kr |
| count-diagonal | 同上，无 Kr，以当前构建原语数替代 Phi；上层控制原语为完整 brick 子树 |
| spatial-midpoint | 选中心跨度最大的世界轴，在中心范围中点切分 |
| count-median | 同一最长中心轴，按稳定顺序尽量平分原语数 |
| flux-median | 同一最长中心轴，选择使左右通量最接近的非空连续划分 |

前三项使用完整原语 AABB 的对角线与原通量。后三项使用加倍整数包围盒中心、世界轴固定 tie-break 和完整世界键稳定次序；无可分中心时按世界键确定性二分。中点规则不得产生空孩子。各规则分别跑**全局逐灯**与**仅上层、完整原 brick 子树保留**两种范围，共 12 个有定义的候选。不可拆 brick 仍只是控制，不缩小最终跨 brick 范围。固定原来的二叉转四叉规则，不混入新的宽化策略。

建树只能读取完整世界灯键、原语 AABB、通量及该规则明确使用的原语数；不能读取测试点、接收法线、M2、遮挡、相机或材质语义。材质 ID 仅属于完整灯身份／确定次序。不能改变叶、32B 最终记录、Half-space A + Near/Far、cube 采样、实际 PDF；禁止任何前瞻、K 摘要、朝向倍率或 visibility 权重。

在原五个输入组的全部 28 点窗评分，严格分 cohort。逐窗给条件方差相对 Prefix、正 M2 支持、null、均值参考身份，单列新红柱／中央顶面、moving 内部 side-a／side-b／top 和 ls5 垂直点；旧边缘点及旧 ls1 也完整保留。报告候选最坏点，不按平均或胜场隐藏回退。首选筛选条件是各指定内部点的两个窗口均不高于 Prefix；若没有候选满足，明确返回最接近者及仍失败的点，不能宣布修复。有限窗改善仍不是整图或移动验收。

同时记录节点数、最大深度、相同点 PMF 下期望遍历内部层数与非空孩子记录读取量；这些只是构造／查询工作的离线代理，不是 GPU ms。核对灯键／通量完整、确定性输入置换、子树接口、所有正 M2 支持和概率守恒。对最有希望的至多两项，沿红柱、顶面、moving 退化灯路径解释概率变化，区分 near/far 两半；以单独参考聚合检查概率差异，不只报总代价。

实验端可新增一个紧凑研究脚本，复用现有输入／评分，向现有实验索引增加本轮字段并保留完整旧记录。运行产物放在当前 clustering-repair run 的独立子目录；耗时 CPU 用 Idle。讨论端拥有双语设计／提示词，实验端不要覆盖它们。提交脚本及必要 Git 跟踪摘要，交付可复算命令、输入及脚本哈希、完整逐点结果、构造限制和最坏反例。不做 GPU、新采集或清理。

沿用此前人类指定的实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429`、讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 和双向回传授权，host=`local`、模型与 effort 不变。工作区为 wt4 / `2.0/light-sampling`，执行前检查最新状态并保护未提交文档。完成或发现改变设计的反例后通知讨论端独立复查；本轮完成后不自动启动 GPU 阶段。

### 首轮结果后的构造方向对照

六规则筛选提交 `05c59e0a8` 已完成，尚无候选通过；最接近的 upper/flux-diagonal 仍有约 5.6% 的红柱回退。讨论端独立复查同时，追加唯一的 bottom-up 规则：`d(A,B) = (PhiA+PhiB) * dot(diagonal(AABB_union), diagonal(AABB_union))`，分别以单灯和完整原 brick 子树构建，共两个候选。它使用同一尺度度量，区分 top-down 切分和 bottom-up 合并；不扫描幂次、惩罚项、混合系数或网格偏移，不重开 H-PLOC kernel／Morton 局部近邻／并行调度。

参照 [Fast Agglomerative Clustering](https://www.graphics.cornell.edu/~bjw/IRT08Agglomerative.pdf)的单调合并代价及互最近邻构造。离线可以批量精确比较所有活跃簇，无需新建复杂 kd-tree；最近邻必须按完整合并代价求，不是中心距离。等价用对称的世界键对消歧，孩子按世界键排序；用一个原始灯小子集对照朴素全局最小对合并，检查等价及确定性。宽化沿用原规则，不能额外展开输入 brick 的内部。仍仅以世界灯键、AABB、通量建树，不用接收数据；28 窗、支持／null、工作代理及双向回传契约不变。先保存原六规则结果，新结果追加独立字段；无论是否通过，本轮结束后均不自动接 GPU。

## 复查后的离线补充任务（2026-11-08）

该补充范围已完成并独立复查；四树隔离及复查端新增受控组合的结果见[设计记录](light-sampling-design.md#分层隔离与受控组合的复查)。以下保留执行规格，不表示继续启动旧 GPU 阶段；当前没有接受新的聚类基线。

这是基于结果提交 `e426ff265` 的后续诊断，替代本提示词中尚未执行的 GPU 候选扩展／动态构建安排。前述红柱反例停线继续有效。本轮仅用现存参考做离线因果隔离，不提交新 GPU 作业、不改 runtime／权重／默认，也不恢复前瞻或 K 系列。

原始 World Prefix/Camera 对照只改变跨 brick 上层；本次全局 SAH 同时改变了上层和 brick 内部。因此还不能把全局 SAH 的红柱回退全归到上层。请在固定 12 桶通量 light-SAH + Kr、同一确定性四叉宽化和 A + Near/Far 下比较四种逻辑树：

| 方法 | 跨 brick 上层 | brick 内部 |
|---|---|---|
| 现有控制 | 原 World Prefix | 原 Morton 子树 |
| 只改上层 | 以实际 brick 根的 AABB／通量构建 light-SAH | 原子树完整保留 |
| 只改内部 | 原 World Prefix 拓扑 | 各 brick 的灯单独构建 light-SAH |
| 已有全局候选 | 全局 light-SAH | 无固定 brick 子树边界 |

上层 SAH 的输入是可变大小的真实 brick 根 AABB，不能当成同尺寸点或单位立方体。按真实包围盒中心分桶，使用其完整面积、通量及稳定世界 brick 键。只改内部时，上层语义子树归属保持不变；重新汇总的浮点通量误差单列，别当作分组效应。可以从原生 brick 根或逻辑树的最大单-brick 子树提取，需核对覆盖完整、无重无漏及每个 brick 的世界灯键。**不可拆 brick 只是一项诊断控制，不是收缩最终全局跨 brick 范围的决定。**

复用[新接收点紧凑参考](../../../experiments/light-sampling/20261108-clustering-repair-reference.npz)、[原五点参考](../../../experiments/light-sampling/20261108-cluster-summaries-reference.npz)与[新增 moving 内部点](../../../experiments/light-sampling/20261108-nearfar-combination-reference.npz)。各完整输入群分别计算，保留两个窗口；不把旧 ls1 与新整图相乘。无需再测 Kr 或新增聚类算法。复用已有控制结果时先核对建树和公式相同。

给出逐点条件方差、正 M2 支持、null 概率与节点／深度，重点报告新红柱、中央顶面、ls5 垂直面及 moving 内部点。红柱另做一个仅诊断的“保持所有下层条件 proposal，根层分支概率换成该条件 M2 下最优”计算：根孩子 j 的条件二阶矩为 `A_j = sum_i(M2_i / p(i|j))`，最优根概率正比 `sqrt(A_j)`，最小方差为 `(sum_j sqrt(A_j))² - ||sum_i mean_i||²`。这不是运行时 oracle 或新权重候选；用来判断只修根层能解释多少退化，不能把全局逐灯最优份额替代它。

延用现有脚本／结果索引，在单独字段记录本补充范围。交付可复算结果，并说明是上层、brick 内部还是交互主导反例，以及只改上层是否保住原有改进点；不要按结果推广成实机修复。双语设计入口由讨论端负责本轮更新，实验端写脚本、实验索引及必要紧凑数据，保留旧结果。耗时计算用 Idle，现有原件保留约定不变。

沿用人类已授权的两 chat 分工与双向回传：实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429`，host=`local`；完成后向讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 回传提交、结果字段及限制，要求复查。无需另行等待阶段批准。

## 本轮任务与授权

优先修复世界稳定灯树的聚类质量，并核实 Near/Far 的当前成本与收益。用户最新决定是“先优先看聚类”，“一层前瞻、K1、K2、K4 先等聚类修好了再搞”，并授权“看着安排下午工作”。本轮暂停这些摘要／前瞻候选的新增组合和选择；保留已有代码、图片与证据，不把暂停解释为删除。聚类比较统一用 Half-space A + Near/Far（model 6），不切生产默认。

沿用实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429` 执行、讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 独立复查，host=`local`。人类此前已指定此实验 chat，并要求完成后通知讨论端复查；本轮延续范围内实现、必要 Vibris 实测、双向派发及完成／实质阻塞回传授权，模型与 effort 不变。wt4 为 `I:/code/mcshaders/Alpha-Piscium-4`、分支 `2.0/light-sampling`，准备起点 `29a5f1be3`；先检查新提交与工作区，不 reset。

## 来源与已知差额

先读[当前设计决定](light-sampling-design.md#当前优先级先修聚类2026-11-08)、[同输入分组记录](../../../experiments/light-sampling/20261108-grouping-control.json)与[对比页](../../../builds/light-sampling/20261108-grouping-control/viewer/comparison.html)。当前同输入结果为 ls1／ls5 的 World Prefix/Camera RGB 方差 1.333／1.502，而 ls1 局部又改善。此次调查的起点是已经证实的分组差异；历史视距变化不是这组同输入回退的解释。

冻结控制来源为 `experiment/light-sampling-grouping-prefix`（`e9132607e`）和 `experiment/light-sampling-grouping-camera`（`24e7ca13d`）。后者是当前评价路径上的原始相机分组，不是已放弃的世界取模方案。当前开发入口为 [LightSamplingBuild](../../../shaders/pass/prepare/LightSamplingBuild.comp.glsl)、[LightSamplingLayout](../../../shaders/techniques/debug/LightSamplingLayout.glsl)、[LightSampling](../../../shaders/techniques/debug/LightSampling.glsl) 和 [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl)。

可复用[五点条件参考](../../../experiments/light-sampling/20261108-cluster-summaries-reference.npz)、[Near/Far 组合记录](../../../experiments/light-sampling/20261108-nearfar-combination.json)及[新增内部点参考](../../../experiments/light-sampling/20261108-nearfar-combination-reference.npz)。旧 ls1 五点批次的体素身份与新分组对照不同，不能因灯键相同就共用 M2 或乘合比例；按各自完整输入分别筛选。

## 聚类候选及必须保持的语义

保持灯集合与通量、真实纹理／遮挡评价、cube 灯内采样、A + Near/Far 公式及实际 PDF。保持四叉最终格式和 32B 节点；新增构建临时数据需单列容量与成本。禁止朝向倍率、修改 near-zero 规则、visibility 引导、逐像素 cut 或前瞻混入这次分组因果对照。

相同世界灯键、material 和通量必须产生相同的语义拓扑；对固定世界接收点与法线，叶 PMF 也应相同，忽略物理地址。不能依赖相机窗口、世界取模或原子分配顺序。**允许替换固定 Morton 前缀父簇**。brick 只负责提取，全局候选以单盏灯为输入，允许相邻 brick 的部分灯成簇，不能悄悄缩成只重排不可拆的 brick 根。单纯全局 Morton 排序后重建原前缀不算新聚类。

首选实现一个确定性的全局分桶 light-SAH 候选。沿三个轴按灯中心分桶，选择最小代价的非空二分；用现有通量 `Phi` 和包含完整单位立方体的 AABB 面积 `S`：

```text
C(axis, split) = Kr(axis) * (Phi_left * S_left + Phi_right * S_right)
Kr(axis) = max(parent_diagonal) / parent_diagonal[axis]
```

这是 [PBRT light-BVH](https://pbr-book.org/4ed/Light_Sources/Light_Sampling) 代价在统一全方向界下的简化，使用现有 mass 作为通量代理，不新增方向摘要。首次采用固定小桶数（例如 12），不做桶数扫描；无可分中心轴时按完整世界键确定性二分。同价、同中心、归并次序及宽树孩子次序均须确定。离线可并列普通 `N_left * S_left + N_right * S_right` 几何 SAH，判断光强项／空间划分是否有用；这只是解释对照，不自动增加 GPU 矩阵。

临时二叉树在构建阶段按固定规则展开为最多四个孩子并压紧；首版可每个输出节点消去下一层二叉内部节点，遇叶停止。所有候选采用同一宽化规则，避免把受接收点驱动的展开混入聚类。记录最终深度、节点／空槽、跨 brick 簇与临时峰值；SAH 和面积只作解释指标。候选不使用接收点、法线、逐灯 M2 或真实 visibility 来建树。H-PLOC 并行核心本轮未指定，也不以论文追踪成绩代替采样结果。

## 执行顺序与实验出口

1. **离线先检查分组是否真的改善选灯。** 在可核对的同输入参考上重建两种控制与候选，逐灯对齐 PMF、零支持和 null 概率，用真实条件 RGB M2 计算 `sum(M2_i / p_i) - ||sum(mean_i)||²`。零概率且正 M2 是失败；零观测 M2 不能证明永久不可见。沿退化灯的祖先记录概率首次失配的位置、簇跨度／通量和 Near/Far 两半的分配，避免只报总 SAH。已有改善点也必须保留。先给出可复算的首份筛选结果，不为此另建通用测试平台。
2. **覆盖用户指出的实际区域。** ls1 包含左侧红柱已改善面及上方中间黄色受光顶面；ls5 包含整图和已复查的退化垂直面；moving-4 用最近选出的面内部点，旧 side-26/top-28 只作历史补充。在出候选结果前，按几何／材料选固定内部 ROI，记录世界坐标、法线和显示坐标，在页上标出，避开方块棱边。不要把旧右侧柱点当成左侧红柱。必要时只补这些区域中缺失代表点的真实逐灯 M2；有身份不匹配的旧参考不能顶替。M2 的窗口和不确定性沿用已有协议。
3. **把有根据的候选接入实机。** 离线有改善且守住支持／稳定契约时，自行继续一个候选的实现与实机比较，不再等逐阶段批准。优先接通实际动态构建到现有采样器；可为质量隔离先用冻结树原型，但须标明上传／冻结方式，并继续区分动态构建未完成的部分。离线候选均无益或出现严重反例时，回传数据与首个失败原因，不强推复杂 GPU 构建，也不擅自重开 K／前瞻。

GPU 构建可以分 dispatch，临时孩子索引、分区及压紧由实验端选择。不得跨工作组无界自旋；核对 dispatch 间发布、间接计数和容量，溢出不得以漏灯降级。若需要新资源，连同生成器、消费者、生成物及契约一起改。实验来源用可追溯 ref，不为一次比较新增生产选项。原型只能声明已实现的部分，不能把 CPU 建树／冻结上传的成本当成零。

## 实机质量、稳定性和 Near/Far 成本

共享 Minecraft 由实验端单独操作，先核对在途作业与当前来源；不要抢占未交还的状态，也不等待无关 workspace 的全局 idle。沿用可用 Vibris 队列和已验证采集路径，不重启 Minecraft／launcher。固定本轮实读的渲染距离（用户当前为 32，不做 16／24／32 扫描）、场景加载和 tick 条件。

质量先测 ls1、ls5、moving-4：World Prefix + A + Near/Far、Camera Grouping + A + Near/Far、一个新聚类 + A + Near/Far，另加独立 Cosine BRDF。保持之前原生 1920×1080、G16、四叉、cube、direct-only，关闭复用／RC／TAA／抖动／降噪，256 帧预热，1 spp 单帧及连续 32 帧均值。保留真实 PDF 与最终遮挡。复用原件必须匹配实际灯键／通量、世界体素材料、接收几何、相机与评价配置；只匹配灯数或 preset 不够。必要时补受影响的一组，不能强凑旧灯数。实际 Target Iris 载入与 effective settings、生成资源及选灯 PDF 均需核验。

所有方法按共同完整有限接收像素统计整图与固定 ROI 的 RGB 方差、均值、零样本覆盖和剔除数，不按亮度筛选；图中更暗不直接等于有偏。有限 32 帧与有限 M2 都需标注范围，必要的窄范围复窗由具体异常触发。不能用新树整图均值接近就证明 PDF 正确，也不能只挑最好区域。

先复用四个 moving preset 的已存跨界输入，检查同一世界输入下候选拓扑／PMF 不随相机地址变化；接入动态构建后再用这些 preset 跨界复测。区分灯／遮挡集合真的变化和纯相机移动。额外用已存旧／新增灯集合做一次离线扰动诊断，报告共同灯的拓扑及 PMF 变化，不把增灯导致的合法重建宣称为已稳定。冻结树截图本身不能证明动态移动修复。

Near/Far 成本单独做 **当前 World Prefix 的 model 1 对 model 6**，仅 ls1、ls5，ABBA 相邻正反序，每项 128 帧 metric 预热及 128 个计时样本，关闭审计。测 [LightSamplingInitial](../../../shaders/pass/composite/LightSamplingInitial.comp.glsl)、完整建树阶段总和、相关共享 prepare 及稳定 sentinel，报绝对 ms、增量／比例与漂移。initial 包含 proposal／cube／trace，不冒充纯查询成本。旧 A→A+Near/Far 增量约 +5.23%／−2.10%／+5.19%／+14.22%（ls1／2／4／5）仅供历史对照。

同时核对当前输入下 A 与 A + Near/Far 的收益；可用完全匹配的已有质量／M2，否则只补这两景的 A 控制。已有 Near/Far 局部反例继续列出，不恢复其他摘要组合矩阵。有动态新聚类时再测它相对 World Prefix 的完整构建 + initial 成本；冻结原型单列 CPU／上传／查询成本与未实现项。不能只测最快核心 kernel。

## 交付、责任与回传

实验端可改范围内的构建源码、必要资源／生成器、诊断脚本及双语结果摘要；讨论端保留路线变更、整体接受和独立复查责任。若需要改变选灯支持、权重公式、32B 最终表示或世界稳定语义，先带证据回传设计问题。其他局部实现选择自行处理。候选失败是有效实验结果，但不能标作聚类已经修复。

沿用本主题入口，跟踪小型结果索引和可复算必要代码，输出放在 `builds/light-sampling/20261108-clustering-repair/`，建议索引 `experiments/light-sampling/20261108-clustering-repair.json`。新增独立比较页，保留之前页面与来源；检查原始 OpenGL 行方向、无损图像和真实浏览器切换。交付输入／源码身份、全图及 ROI、M2／PMF、运动与成本结论，明确静态原型、动态实现、未验及拒绝项。

PowerShell 7 无 profile；耗时非 benchmark 工作 Idle；若用 Pyxis，昂贵执行前核对有效 priority tier。遵守仓库规则，提交前运行两个 diff 检查。已有原件清理暂停和自动策略拒绝继续有效，不重试删除或换工具绕过。新原件保留至独立复查，列明路径、用途和回收条件，不创建无必要的 worktree。

完成或遇到实质设计反例时，停止提交新 runtime 作业，核验并交还实际来源、场景、配置、冻结状态和活动任务。通过 `send_message_to_thread` 向 host=`local`、讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 回传本任务入口、提交／refs、结果页、必要原件及未完成义务，并明确请求独立复查。授权来自上述人类已有双向安排；首份有价值的离线结果可提前回传，继续不依赖新决定的工作。
