# 常用工作流与配置

语言：简体中文 | [English](../../en/development/workflows.md)

本页记录 Alpha Piscium 自己的生成链。Iris 属性语法本身请查阅 [Iris Shaders 文档](https://github.com/IrisShaders/docs)。所有
Kotlin 命令都应从 [`scripts/`](../../../scripts/) 运行；PowerShell 示例从仓库根目录运行。

## 文件所有权

| 文件                                                                                                                                                                             | 所有者                                                                                               | 是否直接修改     |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|------------|
| [`scripts/programs.main.kts`](../../../scripts/programs.main.kts)                                                                                                              | Pass 顺序、包装器、条件和间接调度                                                                               | 是          |
| [`scripts/programs.shaders.properties`](../../../scripts/programs.shaders.properties)                                                                                          | 由 program 生成器生成的 ignored 中间片段                                                                     | 否          |
| [`scripts/shadesmith.shaders.properties`](../../../scripts/shadesmith.shaders.properties)                                                                                      | 由 Shadesmith 生成的 ignored image/atlas 片段                                                           | 否          |
| [`scripts/options.main.kts`](../../../scripts/options.main.kts)                                                                                                                | 设置、profile、界面和翻译                                                                                  | 是          |
| [`scripts/options.lib.kts`](../../../scripts/options.lib.kts)                                                                                                                  | 选项 DSL 实现                                                                                         | 只在修改 DSL 时 |
| [`scripts/shaders.properties`](../../../scripts/shaders.properties)                                                                                                            | Alpha Piscium 手写的属性、自定义资源、SSBO、混合和 uniform                                                        | 是          |
| [`shaders/shaders.properties`](../../../shaders/shaders.properties)                                                                                                            | [`options.main.kts`](../../../scripts/options.main.kts) 汇总所有 `scripts/*.shaders.properties` 和选项输出 | 否          |
| [`shaders/base/Options.glsl`](../../../shaders/base/Options.glsl)、[`TextOptions.glsl`](../../../shaders/base/TextOptions.glsl)、[`shaders/lang/*.lang`](../../../shaders/lang/) | [`options.main.kts`](../../../scripts/options.main.kts)                                           | 否          |
| [`shaders/{setup,begin,shadowcomp,prepare,deferred,composite}*.csh`](../../../shaders/)                                                                                        | [`programs.main.kts`](../../../scripts/programs.main.kts)                                         | 否          |
| [`shaders/shadesmith.json`](../../../shaders/shadesmith.json)                                                                                                                  | Textile tile 布局                                                                                   | 是          |
| [`shaders/base/Textile.glsl`](../../../shaders/base/Textile.glsl)                                                                                                              | Shadesmith 根据 [`shadesmith.json`](../../../shaders/shadesmith.json) 更新的 tracked 绑定                | 否          |

## 本地 `config.properties`

仓库不跟踪 [`scripts/config.properties`](../../../scripts/)。它只配置本地 Java/Shadesmith 工具，不是 Iris 的 [
`shaders.properties`](../../../shaders/shaders.properties)。支持两个键：

```properties
JAVA_PATH=C:/path/to/jdk
SHADESMITH_OUTPUT=./shadesmitth
```

- `JAVA_PATH` 是 JDK 根目录。[`shadesmith.ps1`](../../../scripts/shadesmith.ps1) 未设置时从 `java` 的 `java.home` 推导；[
  `make-zip.kts`](../../../scripts/make-zip.kts) 未设置时使用运行 Kotlin 的 JVM。
- `SHADESMITH_OUTPUT` 是预处理 shader pack 的输出根。默认拼写就是 `./shadesmitth`；按标准命令运行时，相对路径以 [
  `scripts/`](../../../scripts/) 为基准。
- [`scripts/.gitignore`](../../../scripts/.gitignore) 已忽略该文件；保持它为本机配置，不要强制加入 Git。

## 修改普通 GLSL

共享实现放在 [`shaders/techniques/`](../../../shaders/techniques/) 或 [`shaders/util/`](../../../shaders/util/)，新建的
`main` 入口放在 [`shaders/pass/`](../../../shaders/pass/)。当前 program list 仍直接引用少量 `techniques/` 下的 compute
入口（Bloom、RTWSM、atmospherics）；修改这些现存文件不要求顺带搬迁。只修改 `.glsl` 不需要运行 Shadesmith 或任何生成器，因为
wrapper 会在运行时 include 源文件。

## 新增或修改设置

1. 在 [`scripts/options.main.kts`](../../../scripts/options.main.kts) 用现有 `toggle`、`slider`、`constToggle` 或
   `constSlider` 模式声明 `SETTING_*`；同时放入合适的 screen、profile 和中英文 `lang`。
2. 再在 GLSL 或 [`programs.main.kts`](../../../scripts/programs.main.kts) 的 `cond(...)` 中使用它。不要先在 GLSL 中发明未注册的
   `SETTING_*`。
3. 运行：

```powershell
cd scripts
kotlin options.main.kts
```

4. 检查 [`shaders/base/Options.glsl`](../../../shaders/base/Options.glsl)、[
   `shaders/lang/en_US.lang`](../../../shaders/lang/en_US.lang)、[
   `shaders/lang/zh_CN.lang`](../../../shaders/lang/zh_CN.lang) 和 [
   `shaders/shaders.properties`](../../../shaders/shaders.properties)。

[`options.main.kts`](../../../scripts/options.main.kts) 会先调用 program 生成器，因此一次命令会同步包装器、program
属性片段和最终属性文件。可传入版本字符串覆盖界面版本；省略时脚本读取 [`changelogs/`](../../../changelogs/) 中的最高版本。

## 新增、删除或重排 pass

1. 将带入口的 shader 放入 [`shaders/pass/<stage>/`](../../../shaders/pass/)；可复用实现留在 [
   `techniques/`](../../../shaders/techniques/) 或 [`util/`](../../../shaders/util/)。
2. 在 [`scripts/programs.main.kts`](../../../scripts/programs.main.kts) 的正确 `ProgramType` 区块按实际执行顺序添加
   `pass(...)`。
3. 同一个 `pass(pathA, pathB)` 生成同序号的 `name.csh`、`name_a.csh`。当前启用 `allowConcurrentCompute=true`
   ，同组入口不能互相读写依赖；只在确实可并行共享调度位置时使用。
4. 按需使用 `define`、`constDefine`、`cond` 和 `indirect`。Iris 的 `workGroups`/`workGroupRender` 指令仍必须是字面量或展开成字面量的宏。
5. 快速查看编号时运行：

   ```sh
   cd scripts
   kotlin programs.main.kts
   ```

   提交前运行：

   ```sh
   cd scripts
   kotlin options.main.kts
   ```

   将更新后的 program 属性片段并入最终 [`shaders/shaders.properties`](../../../shaders/shaders.properties)。

删除或替换 pass 时，连同旧入口、旧包装器来源和不再使用的设置一起删除。

## 新增 texture tile

1. 在 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 选择正确生命周期：
    - `transient_*`：`screen` 中的临时资源。
    - `history_*`：`screen` 中的历史资源。
    - `persistent_*`：`fixed` 中固定尺寸、跨帧保留的资源。
2. 指定现有支持的格式；固定 tile 还要给出 `width`、`height`。
3. 在 shader 中通过 [`shaders/base/Textile.glsl`](../../../shaders/base/Textile.glsl) 生成的绑定使用 tile，不要手写一套平行的
   offset/format 定义。
4. 从仓库根目录运行：

```powershell
cd scripts
./shadesmith.ps1
kotlin options.main.kts
```

5. Shadesmith 在 [`scripts/`](../../../scripts/) 生成 ignored 的 [
   `shadesmith.shaders.properties`](../../../scripts/shadesmith.shaders.properties)；options 将它汇总进最终 [
   `shaders/shaders.properties`](../../../shaders/shaders.properties)。检查生成输出、tile 是否重叠、读写阶段和 history
   重置行为。除全新 checkout 首次生成外，只有 [`shaders/shadesmith.json`](../../../shaders/shadesmith.json) 改变时才运行
   Shadesmith。

## 制作 ZIP

```powershell
cd scripts
kotlin make-zip.main.kts [version] [--no-commit-hash]
```

脚本先运行 Shadesmith，再把预处理后的 [`shaders/`](../../../shaders/) 与仓库中的 [`changelogs/`](../../../changelogs/)、[
`licenses/`](../../../licenses/)、[`shaders/lang/`](../../../shaders/lang/)、[
`shaders/textures/`](../../../shaders/textures/)、[`LICENSE`](../../../LICENSE) 和 [`README.md`](../../../README.md) 合并到
`builds/`。默认文件名包含短 commit hash；非 `main`/`dev` 分支还包含分支名，`/` 会转换成 `-`。`--no-commit-hash` 去掉
hash/branch 后缀；版本可省略。`builds/` 已忽略。

## 提交前

- 普通 GLSL 改动不应产生生成文件。
- program/options/tile 改动必须同时包含 tracked 生成结果；ignored 片段、`shadesmitth/` 和 AOT cache 不提交。
- 运行：

  ```sh
  git diff --check
  git diff --cached --check
  ```
- 在目标 Iris 版本重新加载，并检查受影响的设置分支、历史重置和代表场景。
