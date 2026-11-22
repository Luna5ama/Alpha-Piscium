# Alpha Piscium 项目知识 · Agent 入口

此页对应 1.10 实现，使用屏幕空间 GI 和普通 Bloom。[版本路线图](roadmap.md)列出 1.10 至 2.9 的范围及两个 2.0 候选的实机验收。

新会话先按问题找本版的维护源、设计取舍和实测条件。已授权的工作产生新结论时，直接写回所属主题。[查找与更新主题](repository-structure.md)

跨版本的设计、实验和反例见[完整项目知识库](https://github.com/Luna5ama/Alpha-Piscium/blob/codex/whole-bloom-dpf/docs/README.md)。各版本共用一个[工作发布索引 VELA.md](https://github.com/Luna5ama/Alpha-Piscium/blob/codex/whole-bloom-dpf/VELA.md)。[项目配置](https://github.com/Luna5ama/Alpha-Piscium/blob/codex/whole-bloom-dpf/.vela/project.toml)中的 `records = "docs"` 指向主题记录，`index = "VELA.md"` 指向发布索引。

## 修改 Shader Pack

| 问题 | 入口 |
| --- | --- |
| 新增受选项控制的 compute pass，要改哪些源文件、运行什么？ | [注册 compute pass](development/generators.md#add-a-compute-pass-controlled-by-a-setting) |
| 生成选项、资源绑定或测试 ZIP 时，从哪里运行脚本？ | [生成与打包](development/generators.md) |
| 入口 shader、共享实现和资源声明放在哪里？ | [代码与资源布局](development/project-structure.md) |
| 怎样生成离线纹理、验证云散射或执行发布？ | [离线工具与发布](development/offline-tools.md) |
| 实机任务怎样按需启动 Vibris 专用实例？ | [启动与连接](development/vibris-startup.md) |

## 理解渲染依赖和取舍

| 问题 | 入口 |
| --- | --- |
| 一帧的输入如何流经光照、体积、半透明和后处理？ | [管线依赖](rendering/pipeline.md) |
| Anchor 为什么淘汰，结论适用于哪些条件？ | [选灯与聚类](rendering/light-sampling.md#anchor) |
| 几何与材质如何进入 G-buffer？ | [几何与材质](rendering/geometry-materials.md) |
| 阴影和 RTWSM 的跨帧依赖是什么？ | [阴影](rendering/shadows.md) |
| 屏幕空间 ReSTIR、环境探针、GI 历史和降噪如何配合？ | [全局光照](rendering/global-illumination.md) |
| 天空 LUT、云与 epipolar 空气散射怎样合成？ | [大气、天空与云](rendering/atmosphere-clouds.md) |
| 水面、焦散与半透明如何使用体积和深度？ | [水与半透明](rendering/water-translucency.md) |
| AA、Bloom、曝光与显示变换如何衔接？ | [后处理](rendering/post-processing.md) |

Iris 的接口、stage 语义和属性语法见 [Iris Shaders 文档](https://github.com/IrisShaders/docs)。
项目记录见[仓库与文档结构](repository-structure.md)。
