# Vibris 专用 Minecraft 实例按需启动

需要实机的任务可以按需启动已停止的 `1.21.11-Vibris`。正在运行的实例按现有队列复用。重启运行中的游戏或启动器须由用户明确要求。

## 从 AP 使用 MCP

1. 调用 `vibris_get_status`，传入任务实际使用的绝对 `worktree_root`。此操作只读。Core 已连接时按返回的队列和准入状态继续。队列繁忙时等现有作业。
2. 仍需实机且 Core 离线时，在提供 `vibris_ensure_runtime` 的 bridge 上调用该工具。传入同一工作树，例如 `{"worktree_root":"I:/code/mcshaders/Alpha-Piscium","timeout_ms":45000}`。bridge 会核对目标进程、端口和启动记录，判断能否启动。调用超时本身无法确定游戏是否停止。
3. 收到 `STARTUP_PENDING` 时可用相同参数继续等。即使把 `timeout_ms` 设为 0，调用也可能发起启动。仅查看状态时使用第一步。确认目标在线后，再用同一工作树调用 `vibris_get_status` 查看作业准入状态。

## 开发机部署

MCP 连接 `127.0.0.1:50051`。专用实例为 PCL 的 `1.21.11-Vibris`，相邻的 `1.21.11-Manual` 属于另一个实例。需要冷启动的新客户端使用部署记录 `I:/code/vibris/build/vibris-on-demand-start/restart-entry/mcp-server.toml` 中的命令及 `--startup-config` 参数。全局默认 bridge 仍可向在线实例提交任务。调用者只提供工作树与等待时间。

专用实例的 `config/vibris/server.json` 为 Core 的计划重启配置了 `restart_executable`，本机核对为 `I:\PCL\1.21.11-Vibris.bat`。用户已指定同一入口用于冷启动，开发机部署会核对两处路径指向同一文件。现有 `vibris_restart` 需要在线 Core，并会排空已接纳的作业。

若 `vibris_ensure_runtime` 返回 `STARTUP_LAUNCH_NOT_CONFIGURED`，当前部署缺少可用的冷启动入口，应向任务发起方报告。若返回 `STARTUP_IDENTITY_UNKNOWN` 或 `STARTUP_UNCONFIRMED`，保留游戏、launcher 和启动记录，并报告工具返回的具体原因。

## 2026-11-21 开发机验收

Vibris `de750971e5bc` 已并排安装。AP 工作树的只读查询先返回 `SERVER_OFFLINE`，随后 `vibris_ensure_runtime` 在 17.516 秒内启动专用实例并返回 `online`。普通 `vibris_get_status` 返回 `operational=true`、`can_accept_job=true`。新客户端复用了同一个 Java 进程，既有 bridge 也可查询在线状态。

7 项相关 Release CTest 通过，覆盖跨客户端并发只启动一次、超时与断连重试、计划重启等待及目标身份核对。原始调用与部署配置见本机 `I:/code/vibris/build/vibris-on-demand-start/restart-entry/REPORT.md`。

## 专用游戏的 Windows GPU 调度实验

用户于 2026-11-22 在项目讨论确认将调度接入 Vibris。Windows 专用游戏进程启动后空闲时使用 WDDM Idle（0），执行实际游戏任务时使用 Realtime（5）。以游戏实际执行作业的生命周期为准，覆盖相邻作业、排队、完成、取消和失败。队列真正空闲时回到 Idle。仅调整已核对身份的专用游戏进程的 GPU 调度类，保留 CPU 调度与其他进程设置。[Windows 调度接口](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/nf-d3dkmthk-d3dkmtsetprocessschedulingpriorityclass)

交付包括实现修订、开发机部署工件、可复跑的生命周期测试和实机记录。按进程身份、作业 ID 和时刻将设置返回值及读回值与实际边界对齐。设置被拒、读回不符和计时受干扰分别记录。同一完整 GI 测量组使用同一策略，在可比后台负载下分别报告实际档位和配对计时的波动。共享 GPU 和游戏按项目租约交接。

本次 GPU 调度部署的一次性重启授权记录于项目讨论输入 `board-input-105`。实现和测试通过后，GI 负责人确认当前完整测量组已结束，暂缓提交下一组。Vibris 部署负责人核实队列排空后，独自执行一次专用 `1.21.11-Vibris` 的计划重启，保留启动器。部署负责人记录组边界、旧新 PID 与新版就绪状态，核对真实作业期间的 Realtime 和结束后的 Idle，再将窗口交回 GI 负责人接续测量。

### 已部署的调度与配对测量

2026-11-22，专用游戏 PID 54872 启动空闲时为 Idle（0）。洛杉矶时间 02:09:39，720p 作业 `9227f95c-3421-473c-85a2-7cbf42c93256` 开始，读回 Realtime（5）。02:10:34 作业完成并排空队列，读回 Idle（0）。进程外的 WDDM 读数与 Core 事件一致，CPU 调度类保持 Normal。安装包和构建包的 SHA-256 一致，启动器已恢复。本机实测日志位于 `I:/code/vibris/build/gpu-priority-20261122/gpu-boundary-events.log`。[已部署的 Vibris 修订](https://github.com/Luna5ama/vibris/commit/25eec03c07195cb13ebcf7518975c20e9437db3a)

普通外部 GPU 负载存在时可以直接提交配对测量。同一游戏实例串行使用，显式独占的测量保留预约窗口。每臂记录后台负载和哨兵耗时，按相近负载比较同场景、同配置的正反序配对，并标出负载突变或哨兵漂移的组。此次实机作业验证了调度类切换；有竞争时的计时稳定性仍需匹配负载的重复配对。
