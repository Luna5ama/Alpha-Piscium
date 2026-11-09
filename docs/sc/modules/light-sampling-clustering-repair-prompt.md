# Light Sampling 聚类修复实验提示词

语言：简体中文 | [English](../../en/modules/light-sampling-clustering-repair-prompt.md)

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
