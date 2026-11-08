# Light Sampling Near/Far 组合与面内诊断实验提示词

语言：简体中文 | [English](../../en/modules/light-sampling-nearfar-combination-prompt.md)

## 授权、目标与交接（2026-11-08）

讨论 chat `01a18c0e-8678-7800-8ab8-04b69f072f12` 的用户要求：World Prefix + Half-space A 保持基线；World Prefix A、一层前瞻、Ward K1／K2／K4 各保留现有版本，再各加一组 Near/Far。独立 Cosine BRDF-only 继续保留。用户同时指出旧 side-26／top-28 标记靠近边角，要求理解一层前瞻表现较稳、K1 严重退化、K4 反而差于 K2 的原因。

由现有实验 chat `01a18648-f66e-78e0-8d01-cd906ca45429` 执行，在 host=`local` 完成后回传讨论 chat 请求独立复查。沿用人类已经授权的实验、按需 Vibris 启动、派发／完成／实质阻塞回传及范围内修订；模型和 effort 沿用当前安排。使用 [wt4](../../../) `I:/code/mcshaders/Alpha-Piscium-4`、`2.0/light-sampling`，准备时 HEAD 为 `4a164df4b`、工作区干净。执行前核对新差异，不 reset。你统一协调本轮源码和运行时，讨论端负责方法与结果复查。

完成一份可解释的 11 方法同输入实机对比，并说明 Near/Far 与粗簇修正组合后的改善／回退。必须保留反例，不能因为离线筛选有回退而省略用户要求的组合。正确性或输入身份失败则修复／重采受影响项，无法继续时回报具体阻塞，不把无效画面放进有效比较。

## 当前起点和已知问题

读取[设计与独立复查](light-sampling-design.md#簇摘要实验独立复查2026-11-08)、[五点结果及参考](../../../experiments/light-sampling/20261108-cluster-summaries.json)、[实机结果索引](../../../experiments/light-sampling/20261108-cluster-iris.json)和[当前 72 图页](../../../builds/light-sampling/20261108-cluster-iris/comparison.html)。实机原型源码在本地分支 `experiment/light-sampling-cluster-iris`（`a4188825`）；moving-4 专用 atlas 在 `experiment/light-sampling-moving4-iris`（`fdd330fc`）。工作分支并未默认启用这些原型，不能只看其 shader 就判断实际采集代码。

本轮讨论端又检查了实际 atlas 与已复查 float32 分量，逐项一致；按当前原生寻址重算 moving-4 的一层前瞻／K1／K2／K4 全部 252 个有效审计记录，GPU PDF 最大相对差约 8.70×10⁻⁷。旧条件采集到本次实机采集之间有 5,606 个节点物理地址变化，比较必须按本次实际树或世界身份对齐，不能复用旧 brick 内地址。

K2／K4 各自逐层压缩，344 个跨 brick 簇中有 57 个的 K4 不是 K2 的细分。节点 70 的 K4 对 K2 成员交叉数为 `[[40,0],[1,47],[0,76],[0,13]]`；其通量加权分量内部方差和由约 119,539 降至 53,735，但 top-28 关键近灯的估权变差。拟合目标与接收点贡献不一致；分量方差球不是成员包围界。上述检查支持已有模型反例，不代表全帧全部路径已无其他问题。

## 固定比较矩阵

沿用 `light-sampling-1` 至 `light-sampling-5` 和 `light-sampling-failure-moving-4` 六景、原生 1920×1080、1 spp 单帧及 32 帧线性均值（8×4 batches）。沿用 direct-only、G16、四叉、cube、RC／复用关闭及原显示变换；Cosine BRDF-only 保持独立。总计 11 方法×6 景×2 帧数＝132 图；若原 72 图仍匹配本轮输入，可以复用，新增 60 图。否则对发生输入变化的场景补齐同输入对照，保留历史来源，不跨批拼成受控结果。

| 结构／摘要 | Near/Far 关 | Near/Far 开 |
|---|---|---|
| World Prefix | World Prefix + Half-space A | World Prefix + Half-space A + Near/Far |
| 一层前瞻 | One-level Lookahead + Half-space A | One-level Lookahead + Half-space A + Near/Far |
| K1 | Ward K1 + Half-space A | Ward K1 + Half-space A + Near/Far |
| K2 | Ward K2 + Half-space A | Ward K2 + Half-space A + Near/Far |
| K4 | Ward K4 + Half-space A | Ward K4 + Half-space A + Near/Far |

显示名也可拆成方法和 Near/Far 两个维度，但 BRDF-only 只能有一个实际案例。Near/Far 开／关是配对比较，基线继续是不开 Near/Far 的 World Prefix + Half-space A。

## 本轮组合公式

下面是讨论端为首轮组合选定的定义；这是待验实验模型，不预设它能修好旧 Near/Far。沿用原 Near/Far 的 1/2 混合和零近距规则，不另加距离钳制、可见性、系数搜索或新的聚类目标。

现有 Near/Far 分支会重算并覆盖单个粗节点的权重。直接把该分支套在现有前瞻／Ward 标量权重后面，会抹掉这些修正。因此每个候选孩子 j 先提供三个未归一化分数 `N[j], F[j], Z[j]`；当前兄弟集合统一归一化和混合。不能在每个孩子的代表集合内部先归一化再求和。

每个代表 r 保留原方法的非负角度振幅 `a_r = mass_r * cosine_r`，以及真实成员包围 AABB `B_r`。使用接收点到 `B_r` 的最近／最远距离平方：

```text
N[j] = sum(a_r / near2_r for near2_r > 0)
Z[j] = sum(a_r for near2_r == 0)
F[j] = sum(a_r / far2_r)

if sum(F) == 0: return existing null event
p_near[j] = Z[j] / sum(Z) if sum(Z) > 0 else N[j] / sum(N)
p[j] = 0.5 * (p_near[j] + F[j] / sum(F))
```

零振幅代表不参与除法。以实际归一化后的 `p[j]` 选分支，实际灯 PDF 沿树累乘，保留既有 null 语义。严格半空间测试沿用原方法的真实 AABB 和容差。三个方法族提供代表的规则如下：

- World Prefix：一个原生候选节点作为一个代表；A 的余弦振幅和原生 AABB。Near/Far 开必须复现当前 model 6 的分支概率与 null。
- 一层前瞻：跨 brick 内部候选用直接孩子作为代表，各自采用 A 的余弦振幅、真实 AABB 和半空间测试；先累加三种分数，再在当前兄弟层归一化。其余节点沿用原生代表。
- Ward：跨 brick 内部候选使用其原有 K 分量及球形余弦振幅，只把距离部分替换为上述 Near/Far；保留整候选 AABB 半空间测试，不额外加入分量级硬剔除。每个分量必须新增真实成员 AABB，由相同 Ward 合并过程对子分量 bounds 作 min/max 得到。不能用 `sqrt(2*variance)` 当作支持界，也不能用同一个父 AABB 冒充不同分量的 bounds。K1 的分量 bounds 应等于节点 bounds。

Near/Far 开时，brick 内和叶候选统一使用原生 Half-space A + Near/Far；关时保持已验的 A。Ward 分组、mass、中心和方差不随 Near/Far 开关改变，新增 bounds 只补统计，不能影响 pair 代价与 tie-break。旧分量球形余弦仍可能低估近灯，这是本轮需要保留的反例，不悄悄一起修掉。

若发现这一定义无法与已有消费者或原始 Near/Far 语义一致，先带最小反例回传讨论端，不自行改成两个完整 proposal 的混合，也不把四种组合都退化为 model 6。数据布局、shader 组织、烘焙方式和具体配置编号可在此语义内调整；修改维护源时同步所有生成物。原型可继续离线烘焙，需明确其不含动态建树成本。

## 面内部诊断及实现核对

旧 side-26／side-27／top-28 留作历史锚点，不再用单个边角点代表整片面。先在 moving-4 选两个侧面和一个朝上面的内部接收点及小区域，利用实际 gbuffer／深度／世界位置核对同一连续表面，避开几何边缘、遮挡轮廓、草叶及材质／法线跳变。根据基线几何选点并固定后再比较所有方法，不按哪种方法好看挑区域。保存坐标、世界原点／法线、材质及示意图；新旧标记明确区分，区域标记可关闭，不烙入原图。

先用现有原件做面内部区域的有限 32 帧统计，报告零样本覆盖和完整像素数，避免把更暗或一次没抽中当成偏差证明。对上述三个新点补同输入双窗逐灯 M2，按已有条件采集协议审计，比较全部十种树方法的完整 PMF/null、真实 M2 方差及主要惩罚分支。旧五点参考仍可用于复算，但旧 ls1 的灯集合与实机批次不同，不能混用。

重点检查：

- K2／K4 的成员归属、mass 和矩守恒；新增支持 bounds 确实包住全部成员，与 GPU atlas 一致。
- Near/Far 关与现有实现一致；Prefix 开与 model 6 一致；前瞻／Ward 开确实使用各自代表，避免被最后的 Near/Far 代码覆盖。
- 每个新配置已在目标 Iris 执行；检查实际源码、revision、场景、设置、灯键／mass、体素、树及选灯 PDF。每个批次使用自身原生寻址。对实际有效审计记录核对完整路径概率，不把只设了开关当作机制生效。
- 关键近灯的低概率发生在哪个分支：压缩分组、余弦下限、近距为零独占，还是实现／映射问题；区分已证实机制与尚不能排除的部分。

## 交付、成本、收尾和回传

更新[同一实机对比页](../../../builds/light-sampling/20261108-cluster-iris/comparison.html)，保留已有可复用的 72 图，新增组合图及面内部标记。核对全部 132 个有效组合、图像方向和无损转换、浏览器方法／帧数／场景切换及标记缩放；明确重采、缺项或输入变化。不得静默用模拟图替代 Iris 画面。

报告五组 Near/Far 开／关配对结果、对固定基线的结果、面内部和旧点反例。先给出原型新增 bounds／atlas 字节与查询工作量；烘焙时间和静态访问数不能称为 GPU 构建或等时间收益。本轮要求质量组合对照，不附加完整性能矩阵。若顺手测实际 initial 路径，保持同条件并单列范围，不能漏掉准备成本后宣布生产加速。

遵守仓库及相关 shader／脚本／文档规则、PowerShell 7 no profile、耗时非 benchmark 任务 Idle。沿用 durable Vibris wait，宿主由你单独协调；可按既有授权冷启动所需服务，不能擅自重启已经运行的 Minecraft／launcher。收尾核验实际恢复状态与存活作业。当前不提升生产默认。

按用户最新 workspace-cleanup 约定收尾：优先复用 wt4；新临时检出、重复源码／压缩包和日志由创建者清理，保护独有实现、未提交改动、活动任务及 Git 对象依赖。最终对比页和图片是用户交付物，保留至被后续对比替代；完整原件只为当前采集核验及讨论端独立复查暂留，回传具体路径、用途和复查后回收责任。有价值结论进入现有跟踪索引及双语主题摘要，不默认长期归档完整输出。不要绕过已有自动策略拒绝。

提交原型源码及必要生成物、结果索引和双语摘要，先通过实际适用的检查和 `git diff --check`／`git diff --cached --check`。完成或遇到实质阻塞时，调用 `send_message_to_thread`，host=`local`、目标=`01a18c0e-8678-7800-8ab8-04b69f072f12`，附本提示词、提交、对比页／索引路径、实际组合公式、三个新点的证据、已验／未验范围和收尾状态，明确请求复查。普通进度不用逐项回传，不能以计划或离线筛选代替用户要求的完整实机页。
