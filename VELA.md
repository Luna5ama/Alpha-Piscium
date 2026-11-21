# Alpha Piscium 工作发布索引

各主题在原设计会话维护规格、验收条件和用户决定。可以交给本项目执行时，主题负责人在下方发布已获授权的规格修订；协调者用 `vela-resource releases` 查看可领取的工作。

| 主题 | 设计归属 |
| --- | --- |
| [新一代降噪](docs/rendering/global-illumination.md#moving-mirror-pilot) | Alpha Piscium 降噪设计会话 |
| DPF Bloom | Alpha Piscium Bloom 设计会话，复用已交付的实验记录，下一轮由此确定实现范围 |
| [Light Sampling](docs/rendering/light-sampling.md) | Alpha Piscium 选灯设计会话，沿用已有实验执行者 |
| 统一体积渲染 | Alpha Piscium 大气与云设计会话 |
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
