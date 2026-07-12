# 文档说明

语言：简体中文 | [English](../AGENTS.md)

这些说明适用于 [`docs/`](../) 下的文档。

## 范围

- 只记录 Alpha Piscium 特有的行为。通用 Iris shader pack 语法或行为应链接到 Iris 文档，而不是重复说明。
- 文档应聚焦维护者工作流、项目结构、生成的配置和主要渲染模块。
- 工作流文档应覆盖常用构建/生成工具：Shadesmith、program list generator、options generator 和 ZIP packaging；其余工具脚本放入单独的
  scripts 文档。
- 每个主要渲染模块都应简要说明代码位置、管线集成、内部 pass 流程、资源、设置和相关限制。

## 语言

- 英文文档位于 [`docs/en/`](../en/)，简体中文文档位于 [`docs/sc/`](./)。
- 成对文档应保持结构和语义等价；一种语言的内容或格式改动通常也需要同步到另一种语言。
- 每份文档标题下方都应放置语言选择器。当前语言以纯文本显示在最前面，所有其他可用语言作为链接列在后面，例如
  `Language: English | [简体中文](...)` 和 `语言：简体中文 | [English](...)`。
- 根目录英文简介是 [`README.md`](../../README.md)，简体中文简介是 [`README.sc.md`](../../README.sc.md)。

## 链接与引用

- 每个具体仓库路径、文件、脚本和 shader pass 都应链接，方便维护者直接跳转。
- 检查每个行内代码标记。如果它表示真实文件或 pass，应将该代码标记作为 Markdown 链接文本；设置、宏、资源名、offset
  和字面值仍保留为普通行内代码。
- 使用仓库相对 Markdown 链接，并验证每个本地目标存在。
- pass 应使用完整逻辑名称并链接到实现文件。为提高信息密度，显示链接文本时省略 shader 文件扩展名；不要使用生成 wrapper
  编号或截断的 pass 名称。

## 工作流格式

- 多阶段 pass 流程使用包含顺序、pass/阶段和用途列的 Markdown 表格。
- 只有在完整高层管线用纵向箭头流程比表格更清楚时，才使用纵向箭头流程。
- 多命令 shell 示例使用标明语言的 fenced code block。
- 从 [`scripts/`](../../scripts/) 运行 Kotlin 生成器：

  ```sh
  cd scripts
  kotlin programs.main.kts
  kotlin options.main.kts
  ```

- 也从 [`scripts/`](../../scripts/) 运行 Shadesmith：

  ```powershell
  cd scripts
  .\shadesmith.ps1
  ```

## 内容要求

- README 文件应包含简明项目结构大纲，以及指向开发快速入门和详细文档的链接。
- 快速入门文档应说明最短的正常开发循环，以及每种改动需要运行的生成器。
- 对于名称相近、但归属不同的配置或生成文件，应链接到各自来源并明确区分职责。
- 生命周期、尺寸规则或持久化语义不同的资源应使用单独的 bullet 或表格行。
- 实现细节应放在执行该细节的 pass 或阶段对应流程行中，而不要放在脱离流程的独立段落中。

## 验证与提交

- 提交前检查双语对应关系、未解析的 wrapper 编号引用、未链接的可解析行内代码引用、失效的本地 Markdown 链接和
  `git diff --check`。
- 每个完成的文档轮次都应提交，便于增量审核。
