# 其他脚本

语言：简体中文 | [English](../../en/development/scripts.md)

日常开发只需要[工作流文档](workflows.md)中的 program list、options、Shadesmith 和 ZIP 脚本。本页记录其余工具，避免把一次性离线生成器混进正常
shader 编辑循环。除 PowerShell/Python 文件外，命令都从 [`scripts/`](../../../scripts/) 运行。

## 发布

### [`release.main.kts`](../../../scripts/release.main.kts)

```powershell
kotlin release.main.kts <version> [-1] [-2] [-3] [-4] [-5] [-6] [-7]
```

这是维护者发布工具，不是本地构建命令。它要求对应的 `changelogs/<version>.md` 和本地 `scripts/tokens.properties`，随后按编号执行：制作
ZIP、改名、创建并推送 Git tag、创建 GitHub Release、发布 Modrinth、发布 CurseForge、打印 Discord 公告。`-N` 跳过步骤 N。

它会切换分支、创建/推送 tag 并调用外部发布 API。ZIP 在脚本切换到 `main`/`dev` 之前从当前 checkout
构建，因此运行前当前分支必须就是要发布的内容。跳过步骤有依赖：后续上传步骤可能需要步骤 1/2 产生并改名的 ZIP，跳过步骤 4
也会跳过其末尾的文件名恢复。不要依赖目录中遗留的旧 ZIP。`tokens.properties` 不应提交。

## 离线纹理与查找表

这些工具多数生成已提交到 [`shaders/textures/`](../../../shaders/textures/) 的静态资产；`extract-opac` 还写 ignored 中间
CSV，`nishina-e` 只打印数值。运行前先查看尺寸、样本数和输出路径；它们不是 shader reload 的前置步骤。

| 脚本                                                                          | 用途                                                             | 主要关联资产                                                                                                                                              |
|-----------------------------------------------------------------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| [`reusetex.main.kts`](../../../scripts/reusetex.main.kts)                   | 生成 ReSTIR spatial reuse 的预计算采样纹理                               | [`restir_reusetex0.bin`](../../../shaders/textures/restir_reusetex0.bin) … [`restir_reusetex3.bin`](../../../shaders/textures/restir_reusetex3.bin) |
| [`gen-spec-brdf-lut.main.kts`](../../../scripts/gen-spec-brdf-lut.main.kts) | 生成 split-sum specular BRDF LUT                                 | [`specular_brdf_lut.bin`](../../../shaders/textures/specular_brdf_lut.bin)                                                                          |
| [`gen-f82-table.main.kts`](../../../scripts/gen-f82-table.main.kts)         | 生成材质 Fresnel/F82 查找表                                           | [`f82.bin`](../../../shaders/textures/f82.bin)                                                                                                      |
| [`gen-noisetex.main.kts`](../../../scripts/gen-noisetex.main.kts)           | 生成 64³ RGBA16 white-noise volume                               | [`white_noise_64x64x64.bin`](../../../shaders/textures/white_noise_64x64x64.bin)                                                                    |
| [`extract-opac.main.kts`](../../../scripts/extract-opac.main.kts)           | 解析 [`data/`](../../../data/) 中的 OPAC/CIE 数据，写中间 CSV 和运行时云相函数纹理 | [`data/opac_raw/`](../../../data/opac_raw/)，[`opac_cloud_phases.bin`](../../../shaders/textures/opac_cloud_phases.bin)                              |
| [`nishina-e.main.kts`](../../../scripts/nishina-e.main.kts)                 | 计算并打印 Nishina 相关数值数组                                           | 标准输出                                                                                                                                                |

通常直接运行：

```powershell
kotlin <script>.main.kts
```

对会生成运行时纹理的脚本，检查二进制尺寸/布局，并同步 [`scripts/shaders.properties`](../../../scripts/shaders.properties)
的 `customTexture.*` 格式、维度、类型；shader 侧同步 sampler 维度、整数/浮点访问类别及硬编码通道假设。

## Bloom 核拟合

[`fit-bloom.py`](../../../scripts/fit-bloom.py) 用四尺度高斯代理校准共用的五 tap 下采样与八 tap 上采样，写入
[`BloomTaps.glsl`](../../../shaders/techniques/BloomTaps.glsl)，并检查代理全部 256 个相位的单位质量与质心。
运行时另行按相等权重组合八层。[`fit-bloom-shape.py`](../../../scripts/fit-bloom-shape.py) 使用实际原始第 1 层 Bloom
载体进行稀疏拟合，生成 [`BloomShapeTaps.glsl`](../../../shaders/techniques/BloomShapeTaps.glsl)。
[`bloom_shape_targets.py`](../../../scripts/bloom_shape_targets.py) 定义方向采样与图案目标。图案拟合后按整数个第 1 层
texel 平移，使视觉包围盒居中。[`Trollface 输入`](../../../scripts/assets/trollface-source.png) 为 Carlos Ramirez 的画作，
取自 [Wikipedia 文件页](https://en.wikipedia.org/wiki/File:Trollface.png)。

两者需要 Python、NumPy 和 SciPy；形状输入还需要 Pillow。从仓库根目录以低进程优先级运行。只在有意修改标准核时重新校准
共用 tap，之后也要重新拟合形状载体：

```powershell
python -B scripts/fit-bloom.py
python -B scripts/fit-bloom-shape.py --builtins --output shaders/techniques/BloomShapeTaps.glsl
```

自定义目标先创建输出目录，再任选一个入口：

```powershell
python -B scripts/fit-bloom-shape.py --image <weights.png> --carrier bloom --mip 1 --tap-budget 64 --output <fit.json> --glsl <shape.glsl>
python -B scripts/fit-bloom-shape.py --function <kernel.py> --size 129 --carrier bloom --mip 1 --tap-budget 64 --output <fit.json> --glsl <shape.glsl>
```

PNG 输入是不透明 8/16 位灰度图，像素为原始线性权重。函数输入是可信 Python 文件，定义 `psf(x, y)` 并返回与坐标网格
相同形状的数组。两种入口均要求奇数尺寸、有限非负权重和正总质量。中心 texel 为原点，x 向右、y 向下，单位是全分辨率像素。
输入归一化为单位质量。JSON gather 偏移以 mip texel 为单位，GLSL 为 atlas 坐标翻转 y，并将相邻拟合 tap 合并成等价双线性读取。
运行时 GLSL 导出要求 `--carrier bloom --mip 1`。每个 invocation 均需在输出范围检查前调用生成的 `bloom_shapePrepare()`。

自定义输出是供维护者评估的滤波片段，不会自动新增玩家选项。`--report` 为 `--builtins` 增加 JSON 报告。
生成源码与生成器修改一起提交，临时拟合报告放入忽略的输出目录。修改权重后复查输入相位、边缘行为及 Iris 渲染。
运行时契约与已接受成本见 [Bloom](../modules/post-processing.md#bloom)。

## 二进制纹理辅助

[`bintex.main.kts`](../../../scripts/bintex.main.kts) 是把一个或多个输入文件转换成指定维度/通道数二进制纹理的独立 CLI：

```powershell
kotlin bintex.main.kts <dimensions_joined_by_underscore> <channels> <output> <input...>
```

[`mipsizepadded.main.kts`](../../../scripts/mipsizepadded.main.kts) 计算带 padding 的 mip 布局比例并打印最大 X/Y
ratio，用于离线尺寸排查。两者都不属于常规构建循环。

## program 调试

[`programs-full.ps1`](../../../scripts/programs-full.ps1) 是依次运行以下命令的便利脚本：

```sh
# 在 scripts/ 中运行
kotlin ./programs.main.kts
kotlin ./options.main.kts
```

必须从 [`scripts/`](../../../scripts/) 运行；脚本不会切换到自身目录。由于 options 生成器会再次调用 program 生成器，program
输出会出现两次；最短的完整生成仍是直接运行 [`options.main.kts`](../../../scripts/options.main.kts)。

## 云验证

从仓库根目录运行这些确定性检查：

```powershell
uv run scripts/cloud_isotropic_ms_check.py --source-root .
uv run scripts/cloud_boundary_confidence_check.py --source-root .
```

[`cloud_isotropic_ms_check.py`](../../../scripts/cloud_isotropic_ms_check.py) 检查各向同性多重散射递推、FP32
稳定性、采样分布、相函数读出和源项集成契约。[`cloud_boundary_confidence_check.py`](../../../scripts/cloud_boundary_confidence_check.py)
检查边界置信度公式、FP64/FP32 边界与边缘行为、突变拒绝以及 GLSL 源码契约。这些只是数学与静态检查；不会编译 shader，
也不能替代目标 Iris 编译和画面验证。

## 色彩与显示变换实验

| 文件                                                        | 用途                     |
|-----------------------------------------------------------|------------------------|
| [`agxtest.main.kts`](../../../scripts/agxtest.main.kts)   | 对 AgX/显示变换公式做离线数值实验    |
| [`agxinv.py`](../../../scripts/agxinv.py)                 | 验证或推导 AgX 逆变换          |
| [`colorspaces.py`](../../../scripts/colorspaces.py)       | 生成/检查颜色空间矩阵和转换常量       |
| [`adobe-fresnel.tsv`](../../../scripts/adobe-fresnel.tsv) | Fresnel 离线输入数据，不是可执行脚本 |

这些工具不会自动更新 GLSL。采用实验结果时，把明确的常量/公式改动放到 [
`shaders/util/colors/`](../../../shaders/util/colors/) 或 [
`shaders/techniques/displaytransform/`](../../../shaders/techniques/displaytransform/)，并单独做数值和实机画面验证。

## 数据文件

[`sponsors.txt`](../../../scripts/sponsors.txt) 是 options 生成器读取的赞助者列表，会进入生成的选项文本/语言文件。修改后运行
`kotlin options.main.kts`。
