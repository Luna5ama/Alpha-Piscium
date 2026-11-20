# AP 仓库与文档结构

## 已决定

AP 的内部文档是供 agent 检索的项目知识。人可以直接问「当时为什么淘汰这个方案，今天还适用吗？」作者自行选择适合材料的语言、篇幅和写法。

取消内部文档的中英文镜像、语言选择器、翻译一致性检查和固定排版要求。用户要求继续设计整套结构，迁移留待结构确定后讨论。设计会话见主检出的 `.vela/local/design-sessions.md`。

## 目录草案

建议沿用开发入口，将内部知识的另一条主线按可独立追问的问题组织。下面用已有材料试排，目录名和拆分均待定。

```text
docs/
  README.md                     从问题找到主题、维护源或工具入口
  development/                  生成、打包、用 Vibris 验证等跨主题工作
  topics/
    pipeline.md                 理解管线依赖的起点
    gi/
      light-sampling.md         选灯，含 Anchor 淘汰的理由
      denoising.md              降噪，有自己的候选和验收判断
    bloom/
      README.md                 生效实现、当前设计和待验证问题
      user8-small-lamps.md      独立复用的小悬灯实验摘要
```

可用现有 `modules/` 名称承载同一结构。选灯和降噪可以各自推进，值得分开；Bloom 的一次短实验可以直接回填主题，只在证据需要独立引用时拆页。若按 `design/`、`research/`、`acceptance/` 划顶层，同一个 Anchor 问题会跨三处，适合单独追踪的跨主题研究则可能更好找。初步建议把长期设计问题放在一起，到确有跨主题消费者时给共享材料一个独立位置。

主题入口可以链接实现和拟合工具，无需据此移动源码。已获授权的可执行规格可以引用主题文档的固定 Git 修订，由根 [VELA.md](../VELA.md) 发布。迁移时要核对两版独有的信息及旧链接。根 README 的安装指南、对外发行说明和游戏内选项各有玩家读者，建议将这些发布物的语言作为单独的范围决定。

## 进一步的建议

- **允许在发现发生时落档。** 根 [AGENTS.md](../AGENTS.md) 目前要求用户明确指示才更新文档。建议让 agent 在用户改变取舍、实验推翻旧判断或实现暴露新约束时直接更新所属主题。只保留会影响下一次判断的差异。一项新发现改变了用户此前接受的前提时，带着反例回到主题会话讨论。
- **把可推导的事实交回维护源。** [programs.main.kts](../scripts/programs.main.kts) 已经定义 compute pass 的顺序、启用条件和生成方式，shader 入口可以直接查它。管线文档值得留下的是 pass 为什么必须先后执行、涉及哪些跨帧依赖、改动可能破坏什么。根指令里的 include 次数、入口总数随代码变化，可以直接查询。
- **让实验检出可以回收，取舍仍可追溯。** Light Sampling 淘汰当前 Anchor 的逐项决定留在实验检出的 `builds/light-sampling/20261109-selection-review/walkthrough.md`。两个收益样本来自同一简单场景的两档网格，复杂场景出现退化，方案还混有 Near/Far 分配的影响。建议在追踪的主题记录保存这类理由、适用范围和能重开问题的新证据。全量运行产物沿用 Vela 管理的位置，无法重建且仍要复核的原件另行保留。
- **用真实问题验收目录调整。** 新会话能否找出 Anchor 的淘汰理由，以及这个决定是否覆盖所有世界空间分组？再问「增加一个受设置控制的 compute pass，从哪个源文件开始？」前一问验证历史判断能否找回，后一问验证能否回到当前实现。先用答案里的缺口决定怎样整理，再考虑移动 shader、脚本或数据。

设计取舍可参考 Fornax 的 [lessons 索引](https://github.com/Echonolix/fornax/blob/cb33759c373d20b11d5d1dea3e15ec9cf39e3aa0/docs/lessons/index.md)与[验收摘要约定](https://github.com/Echonolix/fornax/blob/cb33759c373d20b11d5d1dea3e15ec9cf39e3aa0/docs/acceptance/AGENTS.md)、Vela 的[主题文档与运行记录](https://github.com/Endoqa/Vela/blob/6b3051265d20a953964664690614e3723ba86e80/vela-main-task/references/documents.md)与 [Neo Software Engineering 讨论](https://github.com/Endoqa/Vela/blob/6b3051265d20a953964664690614e3723ba86e80/docs/research/neo-software-engineering.md)。
