# Alpha Piscium

语言：简体中文 | [English](README.md)

高质量的写实 Minecraft 光影包，具有全局光照、体积云、大气散射和出色的视觉效果。

官方网站：https://alphapiscium.org/ \
QQ 群：147399927 \
Discord 服务器：https://discord.gg/E2Uq2MmHgq

<img width="1920" height="1080" alt="Alpha Piscium" src="https://github.com/user-attachments/assets/bab05ffa-9c46-4f9b-a69c-94bc6c82f3d5" />

## 功能

- **实时全局光照**
  - 基于 [*Reservoir-based SpatioTemporal Importance Resampling*](https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf)（ReSTIR）的屏幕空间全局光照
  - 基于 [*ReBLUR*](https://doi.org/10.1007/978-1-4842-7185-8_49) 的降噪
- 逼真、实时且高效的 **大气散射**
  - [*Unreal Engine Sky Atmosphere Rendering Technique*](https://sebh.github.io/publications/egsr2020.pdf)
  - [*Epipolar Sampling*](https://gdcvault.com/play/1018227/Practical-Implementation-of-Light-Scattering)
- **实时体积云**
  - [*Nubis*](https://www.guerrilla-games.com/read/nubis-evolved)
- 高质量自适应 **阴影**
  - [*Rectilinear Texture Warping*](https://www.cspaul.com/publications/Rosen.2012.I3D.pdf)（RTWSM）

## 安装

1. 从 [Modrinth](https://modrinth.com/shader/alpha-piscium) 或 [GitHub Release](https://github.com/Luna5ama/Alpha-Piscium/releases) 下载适用于 Minecraft 的 Alpha Piscium 光影包。
2. 安装 [Iris](https://www.irisshaders.dev/download) 光影加载器。
3. 打开 Minecraft 启动器，并在所玩版本中选择刚创建的 Iris 配置文件。
4. 启动 Minecraft。
5. 依次进入 **选项** → **视频设置** → **光影包**，再点击 **打开光影包文件夹**。
6. 将下载的 ZIP 文件移入该文件夹（`.minecraft/shaderpacks`）。
7. 在游戏内的光影列表中选择新安装的光影包。（通过 Iris 选择后，其文件名会变黄。）
8. 点击 **完成** 或 **应用**，光影包即会加载。

## 常见问题

- **夜晚没有光照**：请使用 PBR 资源包。原版风格建议使用 SPBR；也可尝试 Patrix。
- **无法加载光影包**：请使用 1.7 或更高版本的 Iris。
- **转身后画面全黑**：这是屏幕空间技术的局限，暂无完美的解决方法。
- **Intel GPU**：Alpha Piscium 使用前沿图形技术。遗憾的是，Intel 对 OpenGL 驱动的维护不足，因此不支持 Intel GPU。
- **AMD GPU**：确认 GPU 驱动版本高于 22.7.1；如版本过低，请更新驱动。
- **其他加载错误**：在 Iris 的光影包选择界面按 Ctrl+D，然后重新加载光影包。请根据错误提示，在 [issue](https://github.com/Luna5ama/Alpha-Piscium/issues) 中附上错误信息及 `.minecraft/patched_shaders` 内的相关文件。

## 许可证

`scripts` 目录下的文件采用 MIT License。其他所有文件采用 GNU General Public License v3.0。


## 开发

Alpha Piscium 将 pass 入口、可复用渲染模块、生成绑定和开发脚本分开维护。请先阅读中文版[开发文档](docs/sc/README.md)，再按需查看[快速入门](docs/sc/development/quick-start.md)、[项目结构](docs/sc/development/project-structure.md)、[工作流](docs/sc/development/workflows.md)和[管线总览](docs/sc/modules/pipeline.md)。

项目结构大纲：

- `shaders/pass/`：包含 `main` 的着色器入口。
- `shaders/techniques/`：可复用渲染模块。
- `shaders/util/`：共享数学、采样、材质和数据工具。
- `shaders/base/`：公共接口及生成的设置、纹理绑定。
- `scripts/`：Kotlin、PowerShell 生成器、打包和资源工具。
- `docs/`：项目特定工作流与模块说明。

普通 `.glsl` 改动不需要生成步骤。修改 program list、options、手写的 `scripts/shaders.properties` 或 `shaders/shadesmith.json` 后运行对应生成器，准确命令见[快速入门](docs/sc/development/quick-start.md)。
