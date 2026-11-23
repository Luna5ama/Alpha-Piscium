# Alpha Piscium 工作发布索引

各主题在原设计会话维护规格、验收条件和用户决定。可以交给本项目执行时，主题负责人在下方发布已获授权的规格修订；协调者用 `vela-resource releases` 查看可领取的工作。

| 主题 | 设计归属 |
| --- | --- |
| [2.0 屏幕空间与体素混合 GI](docs/rendering/hybrid-gi.md) | Alpha Piscium 屏幕空间 GI 设计会话，负责分段求交、命中取光及重连可见性 |
| [新一代降噪](docs/rendering/global-illumination.md#moving-mirror-pilot) | Alpha Piscium 降噪设计会话 |
| [DPF Bloom](docs/development/offline-tools.md#geometry-driven-dpf-fitting) | Alpha Piscium Bloom 设计会话，负责目标核、拟合方法及画质与耗时验收 |
| [Light Sampling](docs/rendering/light-sampling.md) | Alpha Piscium 选灯设计会话，沿用已有实验执行者 |
| [统一体积渲染](docs/rendering/atmosphere-clouds.md) | Alpha Piscium 大气与云设计会话 |
| Skibidi DRT | `drt-bench` 的设计会话。移植到 shader pack 时，在此发布对应的集成规格 |
| [仓库与文档结构](docs/repository-structure.md) | Alpha Piscium 仓库结构设计会话，负责文档、设计记录与实验材料的归属，以及与 Vela 工作流的衔接 |

本机的会话链接和协调入口见 `.vela/local/design-sessions.md`。

## 已发布修订

### 2.0 屏幕空间与体素混合 GI

在 `2.0/dev` 为不透明表面接入分段追踪。验收近处实体与体素墙的遮挡次序、Voxy 远景穿越网格的前后段、RC 与上一帧漫反射历史的取光，以及空间重连。此项在 2.0 发布前完成。[实现与实机验收规格](docs/rendering/hybrid-gi.md)

<!-- vela:ready {"item":"screen-space-gi-2-0","rev":"48c9e1f0f04b128fba89561194af40c2318a6484","spec":"docs/rendering/hybrid-gi.md","after":[]} -->

### AP 内部文档迁移

按发布标记固定的规格重整旧中英文内部文档。交付入口为 [docs/README.md](docs/README.md)，验收查阅 Anchor 的淘汰理由及适用范围，并定位受设置控制的 compute pass 的注册源和生成命令。

<!-- vela:ready {"item":"ap-docs-migration","rev":"7980ad56bb71bf2c218a4a907846e5fdf5a6c0e3","spec":"docs/repository-structure.md","after":[]} -->

### Vibris 专用实例按需启动

由 Vibris MCP 实现冷启动，在 AP 专用实例部署并验证，交付正常工具入口和实测结果。[启动规格](docs/development/vibris-startup.md)

<!-- vela:ready {"item":"vibris-on-demand-start","rev":"9b1b276f654af8bd784a9e714338e55a78352608","spec":"docs/development/vibris-startup.md","after":[]} -->

### 镜面 GI 历史写回对照

在镜室和第二个房间验证按像素选择写回多少空间滤波前的镜面信号，记录真实高光恢复、普通区域闪烁和暗区亮点寿命。以实景闭环结果决定候选的去留。[实验规格和现有基线](docs/rendering/global-illumination.md#selective-specular-history-writeback-experiment)

<!-- vela:ready {"item":"gi-specular-selective-writeback","rev":"59fe78cf124b67ddf4c2f684615ffddad8a8f79f","spec":"docs/rendering/global-illumination.md","after":[]} -->

### 跨帧验证镜面写回

将复用前的观测随镜面历史重投影，检验静止机位上 8–32 帧观测的判断能力能否在移动和遮挡变化时保留。沿用两间房间的高光、闪烁和暗区亮点对照。[实验规格](docs/rendering/global-illumination.md#reprojected-fresh-sample-evidence-experiment)

<!-- vela:ready {"item":"gi-specular-fresh-history","rev":"29e4086ed3a8917947748b18d303db4b153d89af","spec":"docs/rendering/global-illumination.md","after":[]} -->

### 程序化积云建模

实验分支已接收 `8e09035ce` 的浅积云轮廓与嵌套 LOD，实机记录在 `33c3e618f`。[建模规格与结果](docs/rendering/atmosphere-clouds.md#积云轮廓与细节层级修订)

<!-- vela:ready {"item":"procedural-cloud-shapes","rev":"8eafcf48395cc45eaec34bb0e7cfe26ffa064150","spec":"docs/rendering/atmosphere-clouds.md","after":[]} -->

### 层积云与高积云建模

原云形建模会话在同一可飞入的世界云场实施低层云卷与中层成片云块，交付地面、层间和俯视对照，以及光照两线可取用的密度版本。[建模与交付规格](docs/rendering/atmosphere-clouds.md#层积云与高积云的建模交付)

<!-- vela:ready {"item":"stratocumulus-altocumulus-shapes","rev":"ca28802c7edebcd463fe86ef10b8981b9985f335","spec":"docs/rendering/atmosphere-clouds.md","after":["procedural-cloud-shapes@8eafcf48395cc45eaec34bb0e7cfe26ffa064150"]} -->

### 统一体积云与多层合成

原云形建模会话在层积云与高积云的可飞入云场上接续七属。先以两束不同高度的卷云打通远山、天空和水面的实际光路，再用跨高度积雨云检查塔心连通与砧下净空。空气、雾与云按各段真实深度共同衰减和散射。[七属云形与空间查询](docs/rendering/atmosphere-clouds.md#十种云共用的世界体积) · [天空、雾和单段水](docs/rendering/atmosphere-clouds.md#多层云与大气雾和水的合成)

<!-- vela:ready {"item":"unified-cloud-volumes","rev":"3acdf23721948b78ca21e50f09d5ec8c03c88fee","spec":"docs/rendering/atmosphere-clouds.md","after":["stratocumulus-altocumulus-shapes@ca28802c7edebcd463fe86ef10b8981b9985f335"]} -->

### 主光体积散射的自适应极线采样

为现有空气、薄霾和单段水接入完整自适应布点、重建与断层补算。在相同成本下比较未遮挡入散射和考虑实际遮挡的细分判据，观察窄光束、半影、轮廓漏光及移动稳定性。[实施范围与对照](docs/rendering/atmosphere-clouds.md#主光体积散射的自适应极线采样)

<!-- vela:ready {"item":"adaptive-epipolar-scattering","rev":"87cf8fc1b391564b9dc48cf2fc045d86529a2b23","spec":"docs/rendering/atmosphere-clouds.md","after":["procedural-cloud-shapes@8eafcf48395cc45eaec34bb0e7cfe26ffa064150"]} -->

### 云光照的物理输运对照

比较同一介质的实时散射与离线参考。固定双塔用于诊断，下一组在已接收的程序化云形上核对误差和实机成本。[云光照规格](docs/rendering/atmosphere-clouds.md#云光照的物理对照) · [在途实验的云形版本](docs/rendering/atmosphere-clouds.md#两项光照实验如何跟进云形)

<!-- vela:ready {"item":"cloud-light-transport-comparison","rev":"d2f1ef0fe2b1ac9a962a062fe7d2af7f922611ba","spec":"docs/rendering/atmosphere-clouds.md","after":[]} -->

### 学习式云散射独立实验

独立的 GPT-6.2 Astra、max 执行会话在原 48–72 小时内比较小网络与小 LUT 对液态云高阶散射的预测，并以固定权重做新云形的整云留出检验。[实验规格与预算](docs/rendering/atmosphere-clouds.md#学习式云散射独立实验) · [跨版本对照](docs/rendering/atmosphere-clouds.md#两项光照实验如何跟进云形)

<!-- vela:ready {"item":"cloud-learned-multiple-scattering","rev":"9356e3e28a49b7c5be4b93b179b38a66ce28e082","spec":"docs/rendering/atmosphere-clouds.md","after":[]} -->

### 从目标核几何自适应拟合 DPF Bloom

让离线生成器从完整 RGB 核分配各层的采样位置、权重及数量。用 User 8 原配方及相同像素的无标注核图，对照旧半分辨率画质，并在 720p、1080p 和 1440p 核对独立 Bloom 的 4× COD 耗时上限。[实现与验收规格](docs/development/offline-tools.md#geometry-driven-dpf-fitting)

<!-- vela:ready {"item":"dpf-bloom-adaptive-fit","rev":"0945ab2a62357c391271c7f7fc37f057c6320329","spec":"docs/development/offline-tools.md","after":[]} -->

### Bloom 编辑器局部亮度剖面

在核图上拖线，同尺度叠看目标与拟合亮度。用已保存的 User 8 结果在浏览器核对相邻细芒之间的暗谷，记录端点和相位，首个自适应候选沿同一条线复查。[网页交互与验收规格](docs/development/offline-tools.md#interactive-bloom-line-profile)

<!-- vela:ready {"item":"dpf-bloom-profile-preview","rev":"37449aa50ed3b6129c612ebf242c971bf2ed74df","spec":"docs/development/offline-tools.md","after":[]} -->

### 用局部重建误差分配 Bloom 细节采样

沿用上一轮的实际输出和耗时。对照只改变亮峰、暗谷评分，以及允许从任意位置取局部滤波形状的两种改动。先用原生像素检查 User 8、旋转版与数字 67，候选守住细节后再核对 1440p 的 4× COD 上限。[对照与停止条件](docs/development/offline-tools.md#residual-guided-fitting-of-thin-features)

<!-- vela:ready {"item":"dpf-bloom-residual-footprints","rev":"f2c0a0d2ead8b531ec8c9423a5510850eaa2eec1","spec":"docs/development/offline-tools.md","after":["dpf-bloom-adaptive-fit@0945ab2a62357c391271c7f7fc37f057c6320329"]} -->

### 定位 Bloom 细芒和暗谷的损失

交换画质参考与候选的上采样滤波，保留各自已有的逐层采样。用 User 8、旋转版与数字 67 的八相位响应，判断细节损失来自哪一部分，或是否只在特定组合中出现。[交换方法与判读](docs/development/offline-tools.md#separating-upsampling-from-per-level-bloom-samples)

<!-- vela:ready {"item":"dpf-bloom-upsampling-replay","rev":"0389c911c759335bc778581c3c9595afd8879e03","spec":"docs/development/offline-tools.md","after":["dpf-bloom-residual-footprints@f2c0a0d2ead8b531ec8c9423a5510850eaa2eec1"]} -->

### 从高画质参考核压缩 Bloom 采样

从能分开细芒的半分辨率滤波核出发，自动找出可在多处复用的局部形状，剩余部分直接采样。在相近成本下与直接合并采样点对照，检查三种核的细芒、笔画、暗谷和宽光晕。[压缩实验规格](docs/development/offline-tools.md#compressing-a-reference-bloom-filter)

<!-- vela:ready {"item":"dpf-bloom-reference-compression","rev":"f17d8f14afb3737b00a00de9c4e4ff69d3cdec9d","spec":"docs/development/offline-tools.md","after":["dpf-bloom-upsampling-replay@0389c911c759335bc778581c3c9595afd8879e03"]} -->

### 按可见断段分配 DPF Bloom 采样

从 RGB 核图跟踪细芒和弯曲笔画，按重建结果里连续缺失的部分放置采样。用原成本比较 User 8、旋转版和 67 的峰谷、末端和宽光晕。画质通过后接续普通导出、实机截图及三档 COD 计时。由现有 DPF 实现会话按 Sol High 执行。[实现与验收规格](docs/development/offline-tools.md#allocating-dpf-samples-along-visible-features)

<!-- vela:ready {"item":"dpf-bloom-feature-continuity","rev":"e1f10bebd1e1f13f681306ac67b638a5d0bb84f4","spec":"docs/development/offline-tools.md","after":["dpf-bloom-reference-compression@f17d8f14afb3737b00a00de9c4e4ff69d3cdec9d"]} -->

### 在相同读取预算下拟合 DPF Bloom 细芒和外圈光晕

由[原 DPF 实现会话](codex://threads/01a1bd0c-c1f4-78d0-bed2-b153fb306371)接续离线拟合。沿用 User 8、旋转 9.5° 的版本和位图 67 的既有预算，同时核对细芒上的亮点和断段、芒间暗谷与外圈光晕。用半分辨率参考判断是否进入导出和实机验证。[冻结的实验范围与验收条件](experiments/dpf-bloom-20261108.md#fitting-ridges-and-outer-halos-at-the-same-read-budget)

<!-- vela:ready {"item":"dpf-bloom-ridge-halo-fit","rev":"9d3eda19aed4a4e3fe298d3cab984fd83e5e6498","spec":"experiments/dpf-bloom-20261108.md","after":["dpf-bloom-feature-continuity@e1f10bebd1e1f13f681306ac67b638a5d0bb84f4"]} -->

### GI 算法与体素追踪四小时性能实验

从 2.0/dev 已落地的 GI 管线出发，按相当画质下每帧节省的 GI 毫秒验收。排期补充 `a6867d91`：两项 2.0 发布工作用完资源后，从优化者同时取得 GPU 和专用实例起连续四小时，到点一次交付。[实验规格](docs/rendering/global-illumination.md#four-hour-gi-performance-experiment-2026-11-22)

<!-- vela:ready {"item":"gi-algorithm-optimization-4h-20261122","rev":"3ea4b5db2402f507b626d187e5858fe7331cc9d0","spec":"docs/rendering/global-illumination.md","after":[]} -->

### Vibris 作业期间的 Windows GPU 优先级

专用游戏执行任务时调到 Realtime，空闲时回到 Idle。按完整测量组边界部署并核对实际档位与作业时序。[实施范围与验收](docs/development/vibris-startup.md#专用游戏的-windows-gpu-调度实验)

<!-- vela:ready {"item":"vibris-gpu-priority-20261122","rev":"1d4b04b4a7759f844db8ef6e4e0e687a480bcc58","spec":"docs/development/vibris-startup.md","after":[]} -->

### 2.0 在旧 Iris 上的 GI 缓冲绑定

在 2.0/dev 修复超出旧版范围的绑定，在 Iris 1.10.4 实机核实加载、体素化和不透明表面 GI。与 2.0 混合 GI 同属第一顺位。[范围与资源交接](docs/roadmap.md#20-离发布还差什么)

<!-- vela:ready {"item":"iris-2-0-compatibility","rev":"a6867d919e70ea2fd02d704f6f2b575257f7b189","spec":"docs/roadmap.md","after":[]} -->

### 2.1 曝光与 DRT

在 `2.1/dev` 实施直方图曝光、四种 DRT 和 SDR／HDR 共享 Look，默认使用 Skibidi 与中性 Look。按[曝光与 DRT 规格](docs/roadmap.md#21-曝光与-drt)核对固定输入的 SDR 输出与 EV，验证暗洞、雪地和小面积强光的曝光响应。用户已在实现会话看过对照并选定 Blender 5.0 AgX；本修订接续 `f50b019c` 的实现和已有画面对照。本修订接续 `4789a2c9` 的 [HDR Mod 实机输出](docs/roadmap.md#hdr-mod-输出) 和原曝光与 SDR 工作。接续 `80131f3d` 的[显示色域缩限](docs/roadmap.md#显示色域缩限)，Skibidi 使用自身压缩，在 SDR 和 HDR 下核对默认 Display P3、其他预设与 D65 自定义三原色 xy。边界测试以 Skibidi 输出为准，后续共享 Look 允许越界。其他 DRT 按自身压缩能力接入。沿用已取得的 SDR 对照，核对预设与同坐标自定义模式，并用模组提供的实际参考白和峰值验收 HDR 高光。本修订接续 `2d020326` 的色域缩限规格，补验共享 Look 在参考白以上高光的响应，以及只调 UI 亮度时场景外观的一致性。统一时域重建沿自研／深改路线独立研究，比较结果与实验提案见[2.1 策略对照](https://github.com/Luna5ama/Alpha-Piscium/blob/881f3b31dfaa41d5613d7b1060eb8af0878d84e6/docs/rendering/post-processing.md#21-时域重建策略对照)。

<!-- vela:ready {"item":"exposure-drt-2-1","rev":"ab9a01bf6d5454f0d34eda4e3f4fbd2872c5d487","spec":"docs/roadmap.md","after":[]} -->
