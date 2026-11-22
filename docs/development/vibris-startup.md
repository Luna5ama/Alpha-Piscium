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

交付包括实现修订、开发机部署工件、可复跑的生命周期测试和实机记录。按进程身份、作业 ID 和时刻将设置返回值及读回值与实际边界对齐。设置被拒、读回不符和计时受干扰分别记录。在完整 GI 测量组结束后启用策略，保留运行中的游戏及启动器。同一组使用同一策略；在可比后台负载下分别报告实际档位和配对计时的波动。若当前实例需重载才能接入，记录已验证的路径、剩余实机条件及可用时间。共享 GPU 和游戏继续按项目租约交接。
