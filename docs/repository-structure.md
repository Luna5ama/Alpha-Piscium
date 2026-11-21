# AP 仓库与文档结构

AP 的内部文档供 agent 按问题检索。文档写入、归档与证据留存沿用 [Vela 的公共约定](https://github.com/Endoqa/Vela/blob/main/vela-main-task/references/documents.md)。agent 在已授权工作中可以直接更新主题，语言和写法由作者选择。用户的工程取舍保存在所属主题，工作发布见 [VELA.md](../VELA.md)。

## 重整现有内容

旧中英文树是迁移的输入。按后续 agent 需要回答的问题整理，可以合并、重写或移除已经失效的说明。两版独有且仍有用的决定、反例和验证条件应能从新入口找回。`development/` 和 `modules/` 是否值得保留由内容决定。公开安装说明、发行说明和游戏内选项的语言另按其玩家用途处理。

[项目配置](../.vela/project.toml) 将 records 映射到 `docs/`，其中受 Git 跟踪的主题摘要可直接用于 Vela 证据回执。[shader 指令](../shaders/AGENTS.md) 将 compute pass 顺序和启用条件指向 [programs.main.kts](../scripts/programs.main.kts)。管线文档用于解释跨阶段依赖及原因。

整理时可以抽查两个问题：新检出能否找回淘汰当前 Anchor 方案的依据，并解释该判断适用于哪些条件？要增加一个受设置控制的 compute pass，应从哪个维护源入手？Anchor 的原评审位置和设计会话见主检出的 `.vela/local/design-sessions.md`。
