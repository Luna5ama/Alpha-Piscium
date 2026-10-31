# Alpha Piscium 开发文档

语言：简体中文 | [English](../en/README.md)

本目录只记录 Alpha Piscium 自己的架构、资源约定和开发流程。Iris
已经公开说明的着色器包接口、阶段语义和属性语法不在这里重复；需要这些背景时请直接查阅 [Iris Shaders 文档](https://github.com/IrisShaders/docs)。

## 从这里开始

1. [开发快速入门](development/quick-start.md)
2. [项目结构](development/project-structure.md)
3. [常用工作流与配置](development/workflows.md)
4. [其他脚本](development/scripts.md)

## 渲染管线与模块

- [管线总览](modules/pipeline.md)
- [几何、G-buffer 与材质](modules/geometry-materials.md)
- [阴影](modules/shadows.md)
- [全局光照](modules/global-illumination.md)
- [大气、天空与云](modules/atmosphere-clouds.md)
- [水、焦散与半透明](modules/water-translucency.md)
- [后处理与显示变换](modules/post-processing.md)

## 文档边界

这些文档重点回答四类项目内问题：代码在哪里、pass 如何接入、资源由谁生成、改动后运行哪个脚本。生成文件会明确标注；不要直接编辑生成文件来代替修改其源文件。
