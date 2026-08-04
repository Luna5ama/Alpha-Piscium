# 项目结构

语言：简体中文 | [English](../../en/development/project-structure.md)

## 仓库根目录

| 路径                                                              | Alpha Piscium 中的职责       |
|-----------------------------------------------------------------|--------------------------|
| [`shaders/`](../../../shaders/)                                 | 可直接装入 shader pack 的运行时内容 |
| [`scripts/`](../../../scripts/)                                 | 生成器、打包、发布和离线资源工具         |
| [`docs/`](../../)                                               | 本项目特定的开发与模块文档            |
| [`changelogs/`](../../../changelogs/)                           | 版本说明；options 脚本也用它推导默认版本 |
| [`data/`](../../../data/)                                       | 离线脚本使用的源数据               |
| [`licenses/`](../../../licenses/)                               | 随包分发的第三方许可证              |
| `.github/`                                                      | GitHub 工作流与仓库配置          |
| [`README.md`](../../../README.md)、[`LICENSE`](../../../LICENSE) | 对外说明和主许可证                |

## [`shaders/`](../../../shaders/)

```text
shaders/
├─ Base.glsl      公共 include 根
├─ base/          共享接口与生成的绑定
├─ pass/          按管线阶段组织的入口 shader
├─ techniques/    可复用模块和现有的直接 compute 入口
├─ util/          通用 GLSL 辅助函数与数据类型
├─ textures/      运行时 LUT、噪声、天空、云和采样资源
├─ lang/          生成的选项本地化文件
├─ *.csh          生成的 compute wrapper
├─ *.vsh/*.fsh…   Iris 兼容入口 wrapper
├─ shaders.properties
└─ shadesmith.json
```

### [`Base.glsl`](../../../shaders/Base.glsl) 与 `base/`

- [`shaders/Base.glsl`](../../../shaders/Base.glsl) 是绝大多数 shader 的公共 include 根，汇入 [
  `base/`](../../../shaders/base/) 下的兼容、uniform、texture、SSBO、option 等接口；util 文件按需另行 include。
- [`Uniforms.glsl`](../../../shaders/base/Uniforms.glsl)、[`Textures.glsl`](../../../shaders/base/Textures.glsl)、[
  `SSBO.glsl`](../../../shaders/base/SSBO.glsl) 定义项目统一的数据绑定；[
  `Configs.glsl`](../../../shaders/base/Configs.glsl) 单独定义 Iris/shadow 兼容常量与 attachment clear 行为，并保留注释中的
  format 参考表。
- [`Options.glsl`](../../../shaders/base/Options.glsl)、[`TextOptions.glsl`](../../../shaders/base/TextOptions.glsl) 由
  options DSL 生成。
- [`Textile.glsl`](../../../shaders/base/Textile.glsl) 承载 Shadesmith tile 绑定；tile 布局的源文件是 [
  `shadesmith.json`](../../../shaders/shadesmith.json)。

### `pass/`

这里只放具有入口点的 shader；它是新入口的标准位置：

- `setup/`：shader pack 加载或尺寸变化时的初始化与清理。
- `begin/`：每帧 global data、LUT、工作队列和临时资源准备。
- `geometry/`：solid、translucent 和 shadow 几何入口实现。
- `shadowcomp/`：shadow 几何后的水面法线处理。
- `composite/`：屏幕空间光照、GI、体积、半透明和后处理入口。
- `general/`：通用 full-screen/no-op wrapper 实现。

没有 `main` 的共享代码应放在 `techniques/` 或 `util/`。

### `techniques/`

此目录按算法或模块组织可复用实现；当前也保留了少量直接注册到 program list 的 compute 入口，例如 [
`Bloom.comp.glsl`](../../../shaders/techniques/Bloom.comp.glsl)、`rtwsm/*.comp.glsl` 和 atmospherics 的 LUT/cloud passes：

- `gi/`：ReSTIR、重投影、reservoir 与 GI 降噪共享实现；SST 核心在 technique 根部 [
  `SST2.glsl`](../../../shaders/techniques/SST2.glsl)。
- `atmospherics/`：空气散射 LUT、epipolar scattering、水下体积和云。
- `rtwsm/`：阴影重要性图、warp 与坐标映射。
- `displaytransform/`：exposure、DRT 和最终显示变换。
- `ffx/`：FidelityFX FSR3 升采样、FSR1 RCAS、SPD kernel 与第三方兼容层。
- 根部文件包括 bloom、DOF、Hi-Z、lighting、environment probe 和 water wave 等共享实现。

这些现存的直接 compute 入口属于既有布局，不应作为新增入口的模板。

### `util/`

这里放置跨模块真正共用的数学、坐标、采样、随机数、材质/G-buffer、BSDF/Fresnel 和颜色空间工具。调试实现位于 [
`shaders/techniques/debug/`](../../../shaders/techniques/debug/)。若代码只服务一个大模块，应优先留在对应
`techniques/<module>/`，而不是扩大通用层。

### 根部 wrappers 与属性

- `setup*.csh`、`begin*.csh`、`shadowcomp*.csh`、`composite*.csh` 由 [
  `scripts/programs.main.kts`](../../../scripts/programs.main.kts) 生成。
- `gbuffers_*`、`shadow*`、`dh_*` 和 `final.*` 是 Iris 需要的兼容入口，include `pass/` 中的真实实现。
- [`shaders.properties`](../../../shaders/shaders.properties) 是生成产物；应维护 [
  `scripts/shaders.properties`](../../../scripts/shaders.properties)、[program/options 源文件](../../../scripts/)。
- [`block.properties`](../../../shaders/block.properties) 提供 block 材质/发光映射；[
  `item.properties`](../../../shaders/item.properties) 当前只保留 `item.0=air`，用于 held-item/空手 ID 行为。
- `voxy*.glsl`、[`voxy.json`](../../../shaders/voxy.json) 是 Voxy 集成入口；Voxy 直接实现 `voxy_emitFragment`，不 include
  通用 geometry pass。

## [`scripts/`](../../../scripts/)

核心生成链是 [`programs.main.kts`](../../../scripts/programs.main.kts)、[
`options.main.kts`](../../../scripts/options.main.kts)、[`shadesmith.ps1`](../../../scripts/shadesmith.ps1) 和 [
`make-zip.main.kts`](../../../scripts/make-zip.main.kts)。其余 `*.kts`/`*.main.kts` 文件主要生成离线纹理/LUT
或执行发布操作；参见[其他脚本](scripts.md)。[`scripts/shaders.properties`](../../../scripts/shaders.properties)
是手写属性源，不是生成结果。

## 修改位置规则

- 新建有 `main` 的入口：[`shaders/pass/`](../../../shaders/pass/)；无 `main`：[
  `shaders/techniques/`](../../../shaders/techniques/) 或 [`shaders/util/`](../../../shaders/util/)。现存 technique
  compute 入口按原位维护，除非单独做结构重构。
- 新模块先复用现有目录；只有出现独立管线职责时才新增顶层 technique 目录。
- 生成文件通过源脚本更新，不直接修补。
- 新设置进入 options DSL；新 tile 进入 [`shadesmith.json`](../../../shaders/shadesmith.json)；新 compute pass
  进入[program list](../../../scripts/programs.main.kts)。
