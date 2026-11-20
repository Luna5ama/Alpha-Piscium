# AP 仓库与文档结构

## 已决定

AP 的内部文档是供 agent 检索的项目知识。人可以直接问「当时为什么淘汰这个方案，今天还适用吗？」作者自行选择适合材料的语言、篇幅和写法。

取消内部文档的中英文镜像、语言选择器、翻译一致性检查和固定排版要求。用户要求继续设计整套结构，迁移留待结构确定后讨论。设计会话见主检出的 `.vela/local/design-sessions.md`。

## 复核后的范围

Vela 的[主题文档与运行记录](https://github.com/Endoqa/Vela/blob/6b3051265d20a953964664690614e3723ba86e80/vela-main-task/references/documents.md)已经规定如何更新已有主题、按需拆页、维护入口、保留反例以及安置原始产物与 fixture。其默认做法是沿用项目现有结构。上一稿预设了 `topics/gi/`、`topics/bloom/` 及实验专页，这些层级缺少独立的 AP 需求，撤回这棵目录树。工作发布沿用 [VELA.md](../VELA.md) 的现有入口。

**建议的最小布局。** 合并内部双语树时，保留两版独有内容，把现有的 `development/`、`modules/` 和文档首页提升到 `docs/`。模块实际长到需要拆分时再按 Vela 约定处理。迁移时更新 [根指令](../AGENTS.md)、[文档指令](AGENTS.md)、[shader 指令](../shaders/AGENTS.md) 和两份公开 README 对内部文档的引用。公开安装说明和发行说明的语言属于玩家发布范围。

**需要用户决定的本地规则。** 根 AGENTS 要求只有用户点名时才能更新 `docs/`。Vela 说明如何落档，项目仍持有写入授权。建议允许 agent 在已授权的项目工作中，把已经作出的取舍和已取得的证据回填所属主题。实现中新发现的前提若会改变设计，记为待讨论，回到主题会话取舍。该建议尚未生效。

**实施时需要核对的两个落点。**

- [项目配置](../.vela/project.toml) 当前将 `records` 指向仓库根的 `records/`，本次复核时该目录尚不存在。若采用上述布局，建议让 records 指向已有主题所在的文档目录，使 Vela 的证据回执可以关联同一份受 Git 跟踪的摘要。迁移时核对该映射和实际记录路径。
- [shader 指令](../shaders/AGENTS.md) 让作者到管线文档查实时 pass 顺序，根指令则指向 [programs.main.kts](../scripts/programs.main.kts)。compute 顺序和启用条件应指向生成器，管线文档保存跨阶段依赖及原因。根指令中的静态 include 数量可在整理时去掉。

可用已有问题抽查迁移效果：新检出能否解释 Light Sampling 淘汰当前 Anchor 的理由及适用范围，能否找到受设置控制的 compute pass 的注册源。Anchor 的原始评审定位见主检出的 `.vela/local/design-sessions.md`。这两个案例供本次迁移复核。
