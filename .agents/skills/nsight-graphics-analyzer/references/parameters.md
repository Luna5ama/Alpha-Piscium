# Parameter Decision Tables — nsight-graphics-analyzer

Loaded on demand from SKILL.md when the Quick Start path is not enough.
Five sub-tables map common decisions to ngfx flags.

- A. User goal → command → required flags → what to read
- B. GPU Trace start trigger choice
- C. Trace quality vs cost trade-off
- D. Drill-down workflow after capture
- E. Advanced tuning (rare)

## Vibris/glc2vk replayer override

For Vibris OpenGL or Vulkan replayer traces, do not use whole-capture
duration, relative-to-capture values, `analysis.frame_budget`,
`fraction_of_gpu`, replayer CPU submission, `replay-perf`, Copy work, or
sleep/yield time as shader evidence. Measure only an individual pass inside
the complete outer `Replay` marker, or `pass_duration / Replay_duration`.
Prefer `gputrace-actions --in-marker "^Replay$"`; exclude the outer `Copy`
marker and the unmarked tail sentinel dispatch. This override takes
precedence over the generic tables below.

For any subcommand's full flag list run
`python "<SKILL_DIR>/scripts/nsight.py" <cmd> --help`.

## Sub-table A — User goal → command → required flags → what to read

| User goal | Command | Required | Read result |
|---|---|---|---|
| "Why is this frame slow on GPU?" (per-stage timing + counters) | `gputrace-capture` | `--exe`, a start trigger, `--out`, `--architecture` | `*.summary.json` then drill |
| "What API calls did this frame make?" (no GPU timing) | `capture` | `--exe`, a trigger, `--out` | `export-functions --summary` |
| "What does this frame look like?" (visual context only) | `capture` | `--exe`, a trigger, `--out` | the auto-exported `.png` |
| "Confirm the capture is from the right run" | `export-metadata` | path | `*.metadata.json` |
| "Is this `.ngfx-capture` healthy / how fast does it replay?" | `replay-perf` | path, `--loops` | `headline.avg_total_ms`, `derived_fps` |
| "I have a `.ngfx-gputrace` already, give me the JSON" | `gputrace` | path | three JSON files written next to it |
| "Is this opaque pass overdrawing? Is GBuffer wasted?" | `gputrace-overdraw` | path, optional `--in-marker "GBuffer.*"` | `verdict[]`, `*_ratios.overdraw_ratio` |
| "Is this frame memory-bound or compute-bound?" | `gputrace-bandwidth` | path, optional `--in-marker` | `dominant_tier`, `memory_vs_compute` |
| "Is SM the bottleneck? Which shader stage dominates?" | `gputrace-shader-bound` | path, optional `--in-marker` | `dominant_shader_stage`, `sm_stall_ratio`, `async_efficiency` |
| "Are there micro-triangles / over-tessellation?" | `gputrace-geometry` | path, optional `--in-marker` | `pixels_per_prim`, frontend ranking |
| "Is the GPU idle? Marker coverage healthy?" | `gputrace-stalls` | path | `gr_idle_pct`, `marker_coverage_pct` |
| "Is the texture cache thrashing? Mips OK?" | `gputrace-texture-cache` | path, optional `--in-marker` | `l1_hit_rate`, `miss_to_dram` |
| "Too many small draws? State-change overhead?" | `gputrace-draws` | path | `small_leaf_pct`, `top_leaf_names`, `state_change_ms_per_frame` |
| "Generate C++ source code that replays this frame" | `cpp-capture` | `--exe`, a trigger, `--out` | session dir contains the C++ project |
| "Just launch the game with ngfx, I'll capture later" | `launch` | `--activity`, `--exe` | game starts; user uses F11 hotkey |
| "Attach to a running process" | `attach` | `--activity`, `--pid` | ngfx attaches; capture via hotkey |
| "What features does this install support?" | `capabilities` | none | `wrapper_features` + per-binary `flags` |

## Sub-table B — GPU Trace start trigger choice

`gputrace-capture` requires exactly one start trigger. Pick by what is
stable to predict in advance.

| What you know about the workload | Trigger |
|---|---|
| Elapsed time after launch (e.g. 30 s for game load) | `--start-after-ms 30000` |
| Specific frame number (reliably reproduces issue) | `--start-after-frames 5000` |
| GPU submit count (more stable than frames in some engines) | `--start-after-submits 1000` |
| Manual: a human presses F11 in the running app | `--start-after-hotkey` |
| Agent-driven workflow, runtime unpredictable (other skills / bot loop / scripted tests in between launch and capture) | `--start-after-hotkey` + call `trigger-hotkey` later (see SKILL.md → Agent-triggered capture) |
| App calls `NGFX_GPUTrace_StartTrace` itself (SDK integration) | `--start-with-ngfx-sdk` |
| Tracing under ngfx-replay (start when replay begins) | `--start-on-replay-begin` |

For headless/CI automation **prefer numeric triggers**. For agent-driven
interactive workflows where total run time isn't predictable, use
`--start-after-hotkey` + the `trigger-hotkey` wrapper subcommand — the
agent synthesizes the F11 press via `SendInput` once the surrounding
workflow signals it's time. Long game loads (3+ min) → use
`--start-after-ms` instead of `--start-after-frames` (frame count
varies with menu screens).

Stop limits (at most one): `--limit-to-frames N` / `--limit-to-submits N`
/ `--stop-with-ngfx-sdk` / `--stop-on-replay-end`. The `--max-duration-ms`
hard cap (default 1000) always applies.

## Sub-table C — Trace quality vs cost trade-off

`gputrace-capture` defaults are tuned for low overhead; switch on the
expensive flags only when you need the data.

| Goal | Flag |
|---|---|
| **Recommended baseline** (most information without crashing) | `--time-every-action` |
| Pass-level totals only (no per-action timing) | omit `--time-every-action` |
| Need source-level shader hotspots (gives up SM/L1TEX detail) | `--real-time-shader-profiler` |
| `[BETA]` per-source-line active threads per warp | `--per-line-active-threads-per-warp` |
| Repeatable timings across runs (default) | `--set-gpu-clocks base` |
| Measure peak throughput | `--set-gpu-clocks boost` |
| Don't perturb workload (battery/thermal-sensitive) | `--set-gpu-clocks unaltered` |
| Busy frame is dropping events | `--allocated-event-buffer-memory-kb 40000` |
| Limit PM/Warp State Sampling background traffic | `--pm-bandwidth-limit <bytes>` |
| **GB20x+ exact compute timestamps** (Hardware Event System) | `--hes-enabled 1` |
| ~~Add hardware counters via multi-pass replay~~ | ⚠️ `--multi-pass-metrics` is **CURRENTLY BROKEN** (see below) |

⚠️ **`--multi-pass-metrics` is unusable through this wrapper** (Nsight
2025.3 + 2026.1, verified May 2026): combined with the mandatory
`--auto-export` it deterministically writes an unloadable
`.ngfx-gputrace`. The wrapper still accepts the flag for
forensic / bug-report purposes but emits a runtime WARNING and returns
`bundle_complete=False`. Full investigation in DESIGN.md → Investigations.

## Sub-table D — Drill-down workflow

After `gputrace-capture` writes the three JSON files, read them in this
order and drill in based on what each tells you:

| Question | Where to look |
|---|---|
| Frame budget verdict, dominant subsystem | `summary.json` → `analysis.frame_budget`, `analysis.throughput` |
| Slowest stage in the frame | `summary.json` → `analysis.hotspots.slowest_stage` |
| Top depth-1 stages with headline metrics | `stages.json` → `top_stages` |
| Top-20 slowest leaf actions | `actions.json` → `top_20_slowest_actions` |
| Children of a specific stage | `gputrace-stages --parent "<regex>" --top 10` |
| Slowest leaves under a subtree | `gputrace-actions --in-marker "<regex>" --top 10 --with-metrics` |
| Filter actions by leaf name | `gputrace-actions --filter "<regex>" --top 10` |
| Specific metric globally | `gputrace-metric --name "<regex>"` |
| Specific metric inside a marker | `gputrace-metric --name "<regex>" --in-marker "<regex>"` |
| What metrics exist in this trace? | `summary.json` → `metrics[*].name` |

`gputrace-actions --sort-by` choices: `duration` (default),
`avg_duration`, `instance_count`. `--top` defaults to 50; `--top 0` for
unlimited (use sparingly — agent context is precious).

## Sub-table E — Advanced tuning (rare)

Map 1:1 to ngfx flags. Skip unless the user explicitly asks.

| Flag | Meaning |
|---|---|
| `--allocated-event-buffer-memory-kb N` | Per-device event buffer in KB (default 20000) |
| `--allocated-hes-buffer-memory-kb N` | HES buffer in KB (GB20x+; default 2000) |
| `--allocated-timestamps N` | Per-device timestamp count (wrapper default 1000000; ngfx default 100000). Wrapper bumps to 10× to prevent overflow with --time-every-action. |
| `--pc-samples-per-pm-interval-per-sm N` | SM hardware sampling interval (power of 2; min 32) |
| `--per-arch-config-path PATH` | Multi-architecture metric-set JSON config |
| `--disable-collect-shader-pipelines` | Smaller trace, no PSO/Shader Source view |
| `--disable-collect-external-shader-debug-info` | Smaller trace, no external shader debug |
| `--disable-trace-shader-bindings` | Don't collect shader hashes |
| `--disable-nvtx-ranges` | Lower overhead, no engine NVTX markers |
| `--allow-tracing-replay-reset 1` | Under ngfx-replay: include reset time as marker |
| `--trace-timeout SEC` | ngfx-internal trace timeout (default 240 s) |
| `--keep-going` | Manual mode: keep collecting until Ctrl+C |
