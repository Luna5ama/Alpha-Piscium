# AP 仓库与文档结构

AP 的内部文档供 agent 按问题检索。文档写入、归档与证据留存沿用 Vela 的《主题文档与运行记录》。安装 Vela 后可从 `$CODEX_HOME/skills/vela-main-task/references/documents.md` 查阅原文，在线阅读需要 Vela 源仓库的访问权限。agent 在已授权工作中可以直接更新主题，语言和写法由作者选择。用户的工程取舍保存在所属主题。

## 查找与更新主题

从[内部文档入口](README.md)按需要回答的问题查找维护源、工作流程或渲染主题。在所属主题更新决定、反例与验证条件，过时的决定可由固定 Git 修订追溯。公开安装说明、发行说明和游戏内选项的语言另按其玩家用途处理。

[完整主题](https://github.com/Luna5ama/Alpha-Piscium/blob/codex/whole-bloom-dpf/docs/README.md)和[工作发布索引](https://github.com/Luna5ama/Alpha-Piscium/blob/codex/whole-bloom-dpf/VELA.md)由同一个项目检出维护。[实际配置](https://github.com/Luna5ama/Alpha-Piscium/blob/codex/whole-bloom-dpf/.vela/project.toml)把 `records` 映射到 `docs/`，受 Git 跟踪的主题摘要可直接用作 Vela 证据回执。各版本的实现说明和验收边界从[本版入口](README.md)查询。

[shader 指令](../shaders/AGENTS.md) 将 compute pass 顺序和启用条件指向 [programs.main.kts](../scripts/programs.main.kts)。管线文档用于解释跨阶段依赖及原因。

可从同一入口找到 [Anchor 的淘汰依据和适用条件](rendering/light-sampling.md#anchor)，以及[增加受选项控制的 compute pass 的维护源和命令](development/generators.md#add-a-compute-pass-controlled-by-a-setting)。

[原迁移规格](https://github.com/Luna5ama/Alpha-Piscium/blob/7980ad56bb71bf2c218a4a907846e5fdf5a6c0e3/docs/repository-structure.md)和[首次验收时的知识入口](https://github.com/Luna5ama/Alpha-Piscium/blob/28b3001d9b109bee539aeff6b1197990564a05ba/docs/README.md)可用于追溯这次整理的依据。
