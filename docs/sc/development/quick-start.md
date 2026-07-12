# 开发快速入门

语言：简体中文 | [English](../../en/development/quick-start.md)

## 前置条件

- 用于最终着色器编译和画面验证的 Minecraft/Iris 测试实例。
- `git`、能够运行 `.main.kts` 的 `kotlin` 命令，以及与 [`scripts/shadesmith.jar`](../../../scripts/shadesmith.jar) 兼容的
  JDK。
- 如果默认 Java 不合适，请在本地 `scripts/config.properties` 中设置 `JAVA_PATH`；参见[工作流与配置](workflows.md)。

在全新 checkout 中，首次生成 options 之前，先在 [`scripts/`](../../../scripts/) 中运行一次 Shadesmith：

```sh
cd scripts
./shadesmith.ps1
```

[`scripts/shadesmith.shaders.properties`](../../../scripts/shadesmith.shaders.properties) 是被忽略的生成片段；若没有它，options
会遗漏最终 [`shaders/shaders.properties`](../../../shaders/shaders.properties) 中的 image/atlas 声明。

## 最短开发循环

1. 使用[项目结构](project-structure.md)和[管线总览](../modules/pipeline.md)定位入口 pass 与共享实现。
2. 将新的入口着色器放在 [`shaders/pass/`](../../../shaders/pass/) 下。现有模块仍在 [
   `shaders/techniques/`](../../../shaders/techniques/) 中保留少数直接 compute 入口；共享 GLSL 位于 [
   `shaders/techniques/`](../../../shaders/techniques/) 和 [`shaders/util/`](../../../shaders/util/)。
3. 仅在其所属源文件变动时，按下表运行生成器。
4. 在 Iris 中重新加载包，同时检查编译输出和目标场景。
5. 提交前运行：

   ```sh
   git diff --check
   git diff --cached --check
   ```

   将受跟踪的生成输出与其源配置放入同一提交。

## 修改后要运行什么

### 普通 `.glsl` 实现

不运行生成器。Iris 在运行时通过 `#include` 使用源文件。

### [`scripts/programs.main.kts`](../../../scripts/programs.main.kts) 中的 pass 顺序、条件、define 或 indirect dispatch

```sh
cd scripts
kotlin programs.main.kts
kotlin options.main.kts
```

重建 wrappers、被忽略的 program 片段，并汇总最终 properties。

### [`scripts/options.main.kts`](../../../scripts/options.main.kts) 中的设置、profile、界面或本地化

```sh
cd scripts
kotlin options.main.kts
```

同时运行 program 生成器，并重建 options、语言文件和最终 [
`shaders/shaders.properties`](../../../shaders/shaders.properties)。

### [`scripts/shaders.properties`](../../../scripts/shaders.properties) 中手写的 texture、image、SSBO、blend 或 uniform

```sh
cd scripts
kotlin options.main.kts
```

将手写片段重新汇总到 [`shaders/shaders.properties`](../../../shaders/shaders.properties)。

### [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 中的 transient、history 或 persistent tile

在 [`scripts/`](../../../scripts/) 中运行 Shadesmith 和 options：

```sh
cd scripts
./shadesmith.ps1
kotlin options.main.kts
```

更新 Textile/image 片段并汇总最终 properties。

### 测试 ZIP

```sh
cd scripts
kotlin make-zip.main.kts [version] [--no-commit-hash]
```

在 `builds/` 下创建包；打包会先运行 Shadesmith。

普通 GLSL 改动不需要运行 Shadesmith；日常开发中仅在 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json)
变动时需要运行，另一个例外是全新 checkout 的首次初始化。

## 源文件与产物对照

| 维护的源文件                                                                                                                        | 生成或使用的位置                                                                                                                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`scripts/programs.main.kts`](../../../scripts/programs.main.kts)                                                             | `shaders/{setup,begin,shadowcomp,prepare,deferred,composite}*.csh`、被忽略的 [`scripts/programs.shaders.properties`](../../../scripts/programs.shaders.properties)                                                                         |
| [`scripts/options.main.kts`](../../../scripts/options.main.kts)、[`scripts/options.lib.kts`](../../../scripts/options.lib.kts) | [`shaders/base/Options.glsl`](../../../shaders/base/Options.glsl)、[`shaders/base/TextOptions.glsl`](../../../shaders/base/TextOptions.glsl)、`shaders/lang/*.lang`、[`shaders/shaders.properties`](../../../shaders/shaders.properties) |
| [`scripts/shaders.properties`](../../../scripts/shaders.properties)                                                           | 合并到最终 [`shaders/shaders.properties`](../../../shaders/shaders.properties) 的手写片段                                                                                                                                                       |
| [`scripts/shadesmith.shaders.properties`](../../../scripts/shadesmith.shaders.properties)                                     | 由 options 汇总的被忽略 image/atlas 片段                                                                                                                                                                                                       |
| [`shaders/shadesmith.json`](../../../shaders/shadesmith.json)                                                                 | 受跟踪的 [`shaders/base/Textile.glsl`](../../../shaders/base/Textile.glsl) 绑定，以及 [`scripts/shadesmitth/`](../../../scripts/shadesmitth/) 中被忽略的片段/输出                                                                                       |
| [`shaders/pass/**`](../../../shaders/pass/)                                                                                   | 新 pass 入口的标准位置                                                                                                                                                                                                                        |
| [`shaders/techniques/**`](../../../shaders/techniques/)                                                                       | 共享模块，以及少数现有的直接 compute 入口                                                                                                                                                                                                             |
| [`shaders/util/**`](../../../shaders/util/)                                                                                   | 跨模块工具                                                                                                                                                                                                                                 |

## 最低验证要求

运行：

```sh
git diff --check
git diff --cached --check
```

仓库没有可替代实时 Iris 编译和画面验证的单一自动化入口；还应检查受影响的 setting/profile 分支、history reset 和代表性场景。
