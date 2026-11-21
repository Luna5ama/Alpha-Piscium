# Alpha Piscium 工作发布索引

各主题在原设计会话维护规格、验收条件和用户决定。可以交给本项目执行时，主题负责人在下方发布已获授权的规格修订；协调者用 `vela-resource releases` 查看可领取的工作。

| 主题 | 设计归属 |
| --- | --- |
| [新一代降噪](docs/rendering/global-illumination.md#moving-mirror-pilot) | Alpha Piscium 降噪设计会话 |
| [DPF Bloom](docs/development/offline-tools.md#geometry-driven-dpf-fitting) | Alpha Piscium Bloom 设计会话，负责目标核、拟合方法及画质与耗时验收 |
| [Light Sampling](docs/rendering/light-sampling.md) | Alpha Piscium 选灯设计会话，沿用已有实验执行者 |
| [统一体积渲染](docs/rendering/atmosphere-clouds.md) | Alpha Piscium 大气与云设计会话 |
| Skibidi DRT | `drt-bench` 的设计会话。移植到 shader pack 时，在此发布对应的集成规格 |
| [仓库与文档结构](docs/repository-structure.md) | Alpha Piscium 仓库结构设计会话，负责文档、设计记录与实验材料的归属，以及与 Vela 工作流的衔接 |

本机的会话链接和协调入口见 `.vela/local/design-sessions.md`。

## 已发布修订

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

在现有 Iris 渲染路径接入可飞入的淡、中、浓积云。以全景、穿云与云上视角核对布局和保守边界，用同场景整帧耗时核对首版预算。形态参考和交付范围见[首版建模规格](docs/rendering/atmosphere-clouds.md#首版建模交付)。

<!-- vela:ready {"item":"procedural-cloud-shapes","rev":"8532145674051de7192bfe3ee22c38a4ea38c7d8","spec":"docs/rendering/atmosphere-clouds.md","after":[]} -->

### 云光照的物理输运对照

在形状、光源与相机固定时增减邻塔，将实时单次和多次散射与同介质的离线参考对照。可从固定样本云体开工，以侧光和逆光的参考误差、运动时的噪声和实机开销决定候选去留。[云光照规格与双球基准](docs/rendering/atmosphere-clouds.md#云光照的物理对照)

<!-- vela:ready {"item":"cloud-light-transport-comparison","rev":"d2f1ef0fe2b1ac9a962a062fe7d2af7f922611ba","spec":"docs/rendering/atmosphere-clouds.md","after":[]} -->

### 从目标核几何自适应拟合 DPF Bloom

让离线生成器从完整 RGB 核分配各层的采样位置、权重及数量。用 User 8 原配方及相同像素的无标注核图，对照旧半分辨率画质，并在 720p、1080p 和 1440p 核对独立 Bloom 的 4× COD 耗时上限。[实现与验收规格](docs/development/offline-tools.md#geometry-driven-dpf-fitting)

<!-- vela:ready {"item":"dpf-bloom-adaptive-fit","rev":"0945ab2a62357c391271c7f7fc37f057c6320329","spec":"docs/development/offline-tools.md","after":[]} -->

### Bloom 编辑器局部亮度剖面

在核图上拖线，同尺度叠看目标与拟合亮度。用已保存的 User 8 结果在浏览器核对相邻细芒之间的暗谷，记录端点和相位，首个自适应候选沿同一条线复查。[网页交互与验收规格](docs/development/offline-tools.md#interactive-bloom-line-profile)

<!-- vela:ready {"item":"dpf-bloom-profile-preview","rev":"37449aa50ed3b6129c612ebf242c971bf2ed74df","spec":"docs/development/offline-tools.md","after":[]} -->

### 用局部重建误差分配 Bloom 细节采样

沿用上一轮的实际输出和耗时。对照只改变亮峰、暗谷评分，以及允许从任意位置取局部滤波形状的两种改动。先用原生像素检查 User 8、旋转版与数字 67，候选守住细节后再核对 1440p 的 4× COD 上限。[对照与停止条件](docs/development/offline-tools.md#residual-guided-fitting-of-thin-features)

<!-- vela:ready {"item":"dpf-bloom-residual-footprints","rev":"f2c0a0d2ead8b531ec8c9423a5510850eaa2eec1","spec":"docs/development/offline-tools.md","after":["dpf-bloom-adaptive-fit@0945ab2a62357c391271c7f7fc37f057c6320329"]} -->
