---
name: nsight-graphics-analyzer
description: >
  Drive NVIDIA Nsight Graphics 2026.1+ from the command line for GPU
  performance analysis, frame capture, frame trace inspection, draw-call
  inspection, NVTX/D3DPERF stage timing, replay metadata extraction, and
  GPU Trace drill-downs. Use when the user mentions nsight, ngfx, GPU trace,
  frame capture, frame trace, gpu profiling, GPU performance, or asks why a
  frame is slow on the GPU. Captures frames from a target game (Graphics
  Capture or GPU Trace), auto-exports the GPU Trace TSV bundle, and exposes
  small JSON queries the agent can read cheaply. Wraps `ngfx.exe`,
  `ngfx-capture.exe`, `ngfx-replay.exe`.
metadata:
  version: 0.1.0
  short-description: Capture and analyze Nsight GPU traces.
  when_to_use: >
    GPU performance analysis, frame capture, draw-call inspection,
    NVTX-marked stage timing, or replay-time metadata extraction. Trigger
    phrases: nsight, ngfx, GPU trace, frame capture, frame trace,
    GPU performance analysis, GPU performance.
  requires:
    - Windows 10/11
    - NVIDIA Nsight Graphics 2026.1 or newer
    - Python 3.10+
  categories:
    - graphics
    - gpu
    - profiling
    - debugging
---

# Nsight Graphics 2026 Skill

Wrapper around NVIDIA Nsight Graphics 2026.1+. The agent captures GPU
frames, gets 3 small JSON artifacts, and uses drill-down subcommands
to answer follow-up questions without ever loading the raw 300+ MB
TSV bundle.

## When to use

User intents that should trigger this skill:

- "Why is this frame slow on the GPU?" / "Profile GPU performance" / "Grab a frame to inspect"
- "Capture a frame from this game" / "frame trace" / "GPU Trace"
- "What draw calls does this frame make?" / "API stream"
- "How fast does this `.ngfx-capture` replay?" / "replay perf"
- "Inspect markers / stages / metrics in this `.ngfx-gputrace`"
- "Generate a C++ capture so I can edit and replay it"

Trigger keywords: `nsight`, `ngfx`, `nsight graphics`, `GPU trace`, `frame
capture`, `frame trace`, `gpu profiling`, `frame profiling`.

## Quick start (90% case)

```
# 1. capture a 200 ms GPU Trace 30 s after launch (game must already be loaded)
python "<SKILL_DIR>/scripts/nsight.py" gputrace-capture \
  --exe "C:\Game\game.exe" --wd "C:\Game" \
  --start-after-ms 30000 --max-duration-ms 200 \
  --architecture Ada --metric-set-name "Throughput Metrics" \
  --time-every-action \
  --out "D:\captures\game.ngfx-gputrace"

# 2. the wrapper writes 3 small JSON next to the .ngfx-gputrace:
#    <session>/game.gputrace.summary.json  (~50 KB)
#    <session>/game.gputrace.stages.json   (~3  KB)
#    <session>/game.gputrace.actions.json  (~15 KB)

# 3. drill into the dominant stage from summary.json's analysis.hotspots
python "<SKILL_DIR>/scripts/nsight.py" gputrace-stages \
  "<session>/game.ngfx-gputrace" --parent "Render Camera" --top 10
```

Replace `<SKILL_DIR>` with the directory containing this `SKILL.md`. For
any subcommand's full flag list run `python scripts/nsight.py <cmd> --help`.

**Three rules to never break:**

- `GPUTRACE_REGIMES.xls` is **300+ MB** — never Read it directly. Drill
  through `gputrace-stages` / `gputrace-actions` / `gputrace-metric`.
- Anti-cheat games (EAC/BattlEye/Vanguard) refuse capture and exit with
  code 4. Try `cpp-capture` or disable the anti-cheat per game policy.
- If a capture fails with a permission error, re-run from an elevated
  PowerShell. `doctor` reports `is_admin`.

## Tool layout

A single Python entry point dispatches to 25 subcommands. Full flag list
for any subcommand: `python "<SKILL_DIR>/scripts/nsight.py" <cmd> --help`.

| Group | Subcommands | Purpose |
|---|---|---|
| Env | `locate` / `doctor` / `capabilities` / `kill` | Detect install, self-check, dump per-binary flags, kill residuals |
| Run no-capture | `launch` / `attach` | Run game under ngfx with no capture taken |
| Capture | `capture` / `cpp-capture` / `gputrace-capture` | Graphics Capture / C++ Capture / GPU Trace + 3 JSON |
| Trigger | `trigger-hotkey` | Synthesize F11 (or another F-key) into a target window so the agent — not a human — fires a `--start-after-hotkey` capture |
| Post-process | `gputrace` | Rebuild 3 JSON from existing trace |
| Drill | `gputrace-stages` / `gputrace-actions` / `gputrace-metric` | Stage tree / leaf-marker top-N / metric aggregate |
| Diagnose (7) | `gputrace-{stalls,bandwidth,shader-bound,texture-cache,overdraw,geometry,draws}` | Focused verdict[]-producing diagnostic commands |
| Replay | `export-{metadata,functions,screenshot}` / `replay-perf` / `replay-analyze` | Metadata / API stream / PNG extraction; replay timing; combined |

Exit codes: `0` success, `2` user error, `3` Nsight not found, `4`
underlying tool failed, `5` wrapper-side timeout. `stdout` is JSON for
query commands; pass `--out FILE` to redirect.

## Parameter Decision Guide

For decisions beyond the Quick Start path — picking a start trigger,
tuning trace quality vs cost, navigating drill-down after capture,
or reaching for an advanced ngfx flag — **Read
`references/parameters.md`**. That file contains five decision tables:

- **A.** User goal → command → required flags → what to read
- **B.** GPU Trace start trigger choice (`--start-after-ms` vs
  `--start-after-frames` vs `--start-after-hotkey` vs SDK)
- **C.** Trace quality vs cost trade-off (`--time-every-action`,
  `--set-gpu-clocks`, `--real-time-shader-profiler`, etc.)
- **D.** Drill-down workflow once the three JSON artifacts are written
- **E.** Advanced 1:1 ngfx flags (rare; skip unless asked)

⚠️ One red-line warning that belongs in the body, not the reference:
**`--multi-pass-metrics` is unusable through this wrapper** (Nsight
2025.3 + 2026.1, verified May 2026). Combined with the mandatory
`--auto-export` it deterministically writes an unloadable
`.ngfx-gputrace`. The wrapper still accepts the flag for
forensic / bug-report purposes but emits a runtime WARNING and returns
`bundle_complete=False`. Full investigation in DESIGN.md → Investigations.

## Capabilities-driven feature gating

The wrapper checks every conditional flag against the live ngfx install
and exits with code 4 + a clear message if your build lacks that flag —
preventing cryptic ngfx parse errors. Run `python scripts/nsight.py
capabilities` to see what's available locally; `wrapper_features.<key>`
maps each wrapper flag to its underlying ngfx flag.

## Subcommand reference

Full flag list for any subcommand: `python scripts/nsight.py <cmd> --help`.
This section is the agent-facing purpose only; flag detail lives in `--help`.

### `gputrace-capture`

Wraps `ngfx.exe --activity "GPU Trace Profiler" --auto-export ...`.
Always writes `<session>/<file>.ngfx-gputrace` + `BASE/` + 3 JSON
artifacts. Use `--dry-run` to preview the ngfx command line.

### `trigger-hotkey`

Synthesize a function-key press (default F11) into a target window via
Win32 `SendInput`. Pair with `gputrace-capture --start-after-hotkey`
run in the background to let the agent — not a human — fire the
trigger once an external workflow has reached the desired scene.
See the **Agent-triggered capture** workflow pattern below.

### `gputrace`

Re-runs the parser pipeline against an existing trace + BASE/. Use
after a skill upgrade to regenerate JSON without re-launching the game.

### `gputrace-stages`

Stage-tree drill. `--parent REGEX` returns children of all matching
parents grouped by name. `--depth N` restricts to a specific depth.

### `gputrace-actions`

Top-N leaf markers (deepest D3DPERF_EVENTS nodes). `--filter` matches
the leaf's own name; `--in-marker` matches any ancestor in the path.
`--with-metrics` adds the headline block via one streaming REGIMES pass
(~1-2 s). Important: read the `definition` field — actions are leaf
markers, not raw draw calls.

### `gputrace-metric`

Aggregate one metric. `--name PATTERN` is a regex over the metric
catalog; `--all-matches` allows multi-match output. `--in-marker REGEX`
returns per-marker windows in addition to the global aggregate.

### Diagnostic command family (7 commands)

`gputrace-stalls`, `gputrace-bandwidth`, `gputrace-shader-bound`,
`gputrace-overdraw`, `gputrace-geometry`, `gputrace-texture-cache`,
`gputrace-draws` — focused diagnostic commands. Each returns a
`verdict[]` array of `{tag, severity, message}`, `severity ∈ {info,
medium, high}`. **Thresholds are heuristics; treat verdicts as
investigation starters, not absolute judgments.** Most accept
`--in-marker REGEX` to scope to a subtree (`gputrace-stalls` and
`gputrace-draws` are whole-frame only).

**Recommended investigation order on an unknown frame:**

1. `gputrace-stalls` — rule out CPU-bound / pipeline-bubble first; if
   the GPU isn't busy, the rest of the analysis doesn't matter.
2. `gputrace-bandwidth` — memory-bound vs compute-bound axis.
3. Branch:
   - memory-bound → `gputrace-texture-cache` to localize
   - compute-bound → `gputrace-shader-bound` for SM utilization detail
4. `gputrace-overdraw` — opaque-pass quality.
5. `gputrace-geometry` — vertex/primitive frontend.
6. `gputrace-draws` — CPU-side state-churn / small-batch signals.

### `replay-perf`

Replays a `.ngfx-capture` N times via `ngfx-replay -n N --perf-report-dir`
and parses `iteration_times.csv`. Useful for replay-cost regression;
**not** for original-app GPU performance (that's `gputrace-capture`).

## How to read the artifacts

**Read `summary.json` first.** Four fields that matter:

- `analysis.frame_budget.verdict` → 60fps / 30fps / below_30fps
- `analysis.throughput.dominant` → most loaded subsystem
- `analysis.hotspots.slowest_stage` → where to drill
- `analysis.warnings` → natural-language flags

Then pick a drill direction: stage > 50% GPU → drill it; throughput
dominated by `dram`/`pcie` → memory-heavy; `sm` dominant → shaders.
**Always drill via subcommands**, never by reading `BASE/*.xls` directly
(REGIMES is 300+ MB).

**Top-level JSON shapes:**

```
summary.json:  source, session, summary{frame/marker/metric counts},
               analysis{frame_budget, throughput, hotspots, warnings},
               headline_metrics, metrics[], hardware_context
stages.json:   source, headline_metrics, roots[], top_stages[]
               each: name, depth, total_duration_ns, fraction_of_gpu, headline
actions.json:  source, definition, top_20_slowest_actions[]
               each: name, path, instance_count, duration fields, headline
```

**Diagnostic verdict thresholds** — use these to interpret `verdict[]`
beyond just `severity`:

| Command | Key signal | Healthy | Red |
|---|---|---|---|
| `overdraw` | `overdraw_ratio` | ~1.0 (1.5-3 typical) | > 3 |
| `overdraw` | `zcull_rejection_rate` | > 0.3 | < 0.3 (ZCull defeated) |
| `overdraw` | `late_z_attrition_rate` | < 0.3 | high (PS work thrown away) |
| `bandwidth` | per-tier % of peak | < 60% | ≥ 80% saturated (60-80% pressure) |
| `bandwidth` | dominant tier vs SM | balanced | tier ≥ SM by 15pp → memory-bound |
| `bandwidth` | PCIe sustained | < 30% | ≥ 30% (host↔device thrash) |
| `shader-bound` | `sm_stall_ratio` | low | high (memory latency / dep chains) |
| `shader-bound` | async-compute use | balanced | compute ≥ 30% but < 10% async (underused) |
| `geometry` | `pixels_per_prim` | ≥ 16 | < 4 (micro-triangles) |
| `texture-cache` | `l1tex__t_sector_hit_rate.pct` | ≥ 70% | < 50% (thrashing) |
| `texture-cache` | `miss_to_dram` | low | ≥ 10% (DRAM burned on misses) |
| `stalls` | `gr_idle_pct` | low | high (GPU not fed) |
| `stalls` | `marker_coverage_pct` | high | low (idle BETWEEN markers → CPU bound / waits) |
| `draws` | `small_leaf_pct` (< 5 μs) | low | high (batching candidates) |

## Workflow patterns

### First-pass perf analysis

`gputrace-capture` → read `summary.json` → 1-2 drill rounds → write
Markdown report. Stop drilling once the bottleneck is at the
engine-marker level (e.g. "GPUDriven.RenderMesh(Bush)") or attributable
to a metric pattern (e.g. dram throughput > 80%).

### Deep bottleneck dive (single stage > 80% of frame time)

```
gputrace-stages --parent "<stage>" --top 20
gputrace-actions --in-marker "<stage>" --top 20 --with-metrics
gputrace-metric --name "dramc__throughput" --in-marker "<stage>"
gputrace-metric --name "sm__throughput" --in-marker "<stage>"
```

### Capture-once-then-iterate

```
gputrace-capture --exe ... --out trace.ngfx-gputrace
# now drill repeatedly; no re-launch needed
gputrace-stages "<session>/trace.ngfx-gputrace" ...
gputrace-actions "<session>/trace.ngfx-gputrace" ...
```

### Reanalyze an old trace

```
gputrace "<session>/old-trace.ngfx-gputrace"   # rebuilds 3 JSON
```

### cpp-capture for repro (deterministic C++ replay code)

```
cpp-capture --exe ... --wait-seconds 10 --out D:\repro
```

### Attach to a running game (can't relaunch, e.g. multiplayer)

```
attach --activity "GPU Trace Profiler" --pid 12345
# user presses F11 in the running app
```

### Agent-triggered capture in a larger workflow

**Use this when:** the agent is orchestrating a wider workflow — launching
the game, running automated tests / bot actions / other skills, then
deciding when to capture — and only the agent (not a human at the
keyboard) knows when the target scene is on screen. Timer-based triggers
(`--start-after-ms` / `--start-after-frames`) don't fit because the
runtime of the surrounding workflow is unpredictable.

**Why F11 simulation is the only option:** ngfx's GPU Trace activity
exposes exactly five start triggers (`--start-after-ms` /
`--start-after-frames` / `--start-after-submits` / `--start-after-hotkey`
/ `--start-with-ngfx-sdk` / `--start-on-replay-begin`). `ngfx-rpc.exe`
is the UI↔replayer IPC, not a trace-trigger RPC, and `--start-with-ngfx-sdk`
requires the game to link the NGFX SDK and call `NGFX_GPUTrace_StartTrace`.
For an unmodified target, **hotkey is the only externally addressable
trigger**, and `trigger-hotkey` synthesizes it without a human.

**Pattern:**

```
# 1. Launch ngfx + game in the background, armed for F11 trigger.
#    DO NOT wait on this command — it blocks until the trace is written.
python scripts/nsight.py gputrace-capture \
  --exe "C:\Game\game.exe" --wd "C:\Game" \
  --start-after-hotkey \
  --max-duration-ms 200 \
  --architecture <arch> --metric-set-name "Throughput Metrics" \
  --time-every-action \
  --out "D:\captures\game.ngfx-gputrace"
# (Run via your harness's background-task primitive; capture the task ID.)

# 2. Attach a streaming watcher to the background task's output so the
#    agent is notified when the bundle is written. The wrapper prints
#    "GPU Trace report saved to ..." then "bundle_complete=True". Watch
#    for those lines (and also for "error", "fatal", "denied" to fail fast).

# 3. The agent now runs the rest of the workflow (other skills, automated
#    tests, scripted gameplay) — no polling. The capture is dormant in
#    the game process; F11 has not been pressed yet.

# 4. When the workflow signals "ready to capture":
python scripts/nsight.py trigger-hotkey --process game
# Returns JSON with foreground_ok / sent fields. exit 0 = key delivered.

# 5. ngfx hook receives F11, runs the 200 ms trace, writes the bundle,
#    wrapper post-processes into 3 JSON. The streaming watcher from step 2
#    fires; the agent reads summary.json and drills as usual.
```

**Notes:**

- **Background launch is mandatory.** `gputrace-capture` blocks until the
  bundle is written, which is "forever" from the agent's point of view
  (it depends on when step 4 fires). The agent must run step 1
  asynchronously.
- **Watch for completion, do not poll.** The wrapper writes `bundle_complete=True`
  + the saved path to its stdout/stderr; pipe that into your harness's
  streaming notification primitive so the agent reacts to the event
  rather than guessing.
- **Windowed/borderless mode preferred.** `SendInput` is reliable against
  windowed and borderless-window games. Exclusive-fullscreen + DRM
  combinations can swallow synthetic keystrokes; if `trigger-hotkey`
  reports `sent=true` but no trace appears, switch the game to
  borderless-window and retry.
- **Foreground stealing is restricted on Windows.** `trigger-hotkey`
  calls `AllowSetForegroundWindow(ASFW_ANY) + SetForegroundWindow`,
  which works from an elevated shell (`doctor` reports `is_admin=true`)
  and usually from a non-elevated shell that recently saw user input.
  If it can't bring the window forward it still sends the key (ngfx's
  hook is global) and emits a stderr WARNING. Pass `--no-foreground` to
  skip the foreground step entirely if you'd rather not steal focus.
- **Use `--use-scancode` if the default fails.** Some engines / DRM
  shims only honour hardware scancodes (KEYEVENTF_SCANCODE). The flag
  toggles the wParam encoding without other changes.
- **ngfx 2026.1.x cleanup-phase crash is benign.** After the trace is
  written, ngfx exits with rc `0xC0000409` *after* `trace_written=True`
  and `bundle_complete=True`. The wrapper detects this and still
  produces the 3 JSON. Don't treat the non-zero rc as failure; check
  `bundle_complete` instead.
- **Disambiguating multiple instances.** If `tasklist` finds more than
  one process matching `--process`, `trigger-hotkey` refuses and lists
  the PIDs; re-run with `--pid <N>`. Always use `--pid` when ngfx itself
  launched the game (you have the PID from the capture-task output).

## Out of scope

- **Per-API-call timing** (no `vkCmdDraw` durations — Nsight 2026.1
  removed `GPUTrace.pyd`; actions in this skill are leaf markers, not
  raw draw calls). Use RenderDoc/PIX for per-API timing.
- **HTML/PDF report generation** — agent writes Markdown directly.
- **Capture diff** (compare two runs) — future work.
- **Cross-platform** — Windows only.
- **Auto-installing Nsight** — `doctor` only detects.
- **Remote capture (ngfx-rpc)** — use `ngfx-rpc.exe` manually if needed.
- **NVTX marker injection** — game-side, not skill-side.
- **GUI launching** — open `ngfx-ui.exe` manually for visual inspection.
