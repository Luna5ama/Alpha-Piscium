# Vibris 专用 Minecraft 实例按需启动

需要实机的任务可以按需启动已停止的 `1.21.11-Vibris`。正在运行的实例按现有队列复用。重启运行中的游戏或启动器须由用户明确要求。

## 从 AP 使用 MCP

1. 调用 `vibris_get_status`，传入任务实际使用的绝对 `worktree_root`。此操作只读。Core 已连接时按返回的队列和准入状态继续。队列繁忙时等现有作业。
2. 仍需实机且 Core 离线时，在提供 `vibris_ensure_runtime` 的 bridge 上调用该工具。传入同一工作树，例如 `{"worktree_root":"I:/code/mcshaders/Alpha-Piscium","timeout_ms":45000}`。bridge 会核对目标进程、端口和启动记录，判断能否启动。调用超时本身无法确定游戏是否停止。
3. 收到 `STARTUP_PENDING` 时可用相同参数继续等。即使把 `timeout_ms` 设为 0，调用也可能发起启动。仅查看状态时使用第一步。确认目标在线后，再用同一工作树调用 `vibris_get_status` 查看作业准入状态。

## 开发机部署

MCP 连接 `127.0.0.1:50051`。专用实例为 PCL 的 `1.21.11-Vibris`，相邻的 `1.21.11-Manual` 属于另一个实例。目标目录和可执行入口由 MCP 宿主的 `--startup-config` 指定。调用者只提供工作树与等待时间。

专用实例的 `config/vibris/server.json` 为 Core 的计划重启配置了 `restart_executable`，本机核对为 `I:\PCL\1.21.11-Vibris.bat`。脚本正文可能包含登录信息。现有 `vibris_restart` 需要在线 Core，并会排空已接纳的作业。

若 `vibris_ensure_runtime` 返回 `STARTUP_LAUNCH_NOT_CONFIGURED`，当前部署缺少可用的冷启动入口，应向任务发起方报告。若返回 `STARTUP_IDENTITY_UNKNOWN` 或 `STARTUP_UNCONFIRMED`，保留游戏、launcher 和启动记录，并报告工具返回的具体原因。

## 2026-11-20 验收记录

Vibris 提交 `9a02af58f1fa` 的候选 MCP 已并排安装。7 项相关 Release CTest 通过。隔离测试用两个 stdio 客户端和替身 Core 验证了并发只启动一次、超时和断连后重试、在线复用、计划重启等待及目标身份核对。

首次开发机实测时，专用游戏进程、PCL 进程和 50051 监听均不存在。候选从 AP 主工作树调用状态查询得到 `SERVER_OFFLINE`，调用按需启动得到 `STARTUP_LAUNCH_NOT_CONFIGURED`。当时的默认 bridge 仍为旧版，真实 PCL 冷启动和随后的 AP 在线握手待验。

现装 PCL 2.13.1.1 没有按版本名直接启动游戏的命令行参数。它的启动按钮依赖鼠标事件，版本列表会保存新的默认选择。候选部署当前仅核对实例并等待连接。恢复实机验收需要确认同一实例平时由哪个入口启动并完成登录，再将该入口接入部署。
