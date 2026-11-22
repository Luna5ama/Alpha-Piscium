# AP 仓库与文档结构

AP 的内部文档供 agent 按问题检索。文档写入、归档与证据留存沿用 [Vela 的公共约定](https://github.com/Endoqa/Vela/blob/main/vela-main-task/references/documents.md)。agent 在已授权工作中可以直接更新主题，语言和写法由作者选择。用户的工程取舍保存在所属主题，工作发布见 [VELA.md](../VELA.md)。

## 版本与发布分支

跨版本路线图和派工清单各维护一份，位置是项目主检出 `codex/whole-bloom-dpf` 的[路线图](roadmap.md)和[发布索引](../VELA.md)。各版本从已有[项目知识入口](README.md)查阅。2.0 的实现与随版说明交给 `2.0/dev`，2.1 的交给 `2.1/dev`。`main` 按明确安排的发布或默认分支变更更新。`1.10/dev` 已收尾，仅在明确安排该版修复或勘误时更新。

2.1 专属设计细节在 [2.1 的路线图](https://github.com/Luna5ama/Alpha-Piscium/blob/2.1/dev/docs/roadmap.md)维护。进入派工时，共用索引引用冻结规格；跨版本路线图在版本归属或范围改变时才更新。同一段设计正文只维护在所属版本。跨版修复在各目标确认问题和验收范围，记录来源提交，然后按根指南在本地整合并直推。

## 查找与更新主题

从[内部文档入口](README.md)按需要回答的问题查找维护源、工作流程或渲染主题。在所属主题更新决定、反例与验证条件，过时的决定可由固定 Git 修订追溯。公开安装说明、发行说明和游戏内选项的语言另按其玩家用途处理。

[项目配置](../.vela/project.toml) 将 records 映射到 `docs/`，其中受 Git 跟踪的主题摘要可直接用于 Vela 证据回执。[shader 指令](../shaders/AGENTS.md) 将 compute pass 顺序和启用条件指向 [programs.main.kts](../scripts/programs.main.kts)。管线文档用于解释跨阶段依赖及原因。

可从同一入口找到 [Anchor 的淘汰依据和适用条件](rendering/light-sampling.md#anchor)，以及[增加受选项控制的 compute pass 的维护源和命令](development/generators.md#add-a-compute-pass-controlled-by-a-setting)。Anchor 的原评审位置和设计会话见主检出的 `.vela/local/design-sessions.md`。
