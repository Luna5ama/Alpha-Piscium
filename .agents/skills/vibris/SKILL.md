---
name: vibris
description: Use Iris capture control, glc2vk OpenGL/Vulkan replayers, and Nsight GPU Trace for Alpha-Piscium compute shader debugging and profiling.
---

# Vibris

Use this skill when working on Alpha-Piscium compute shader captures, glc2vk
replays, direct Iris capture control, or Nsight GPU Trace captures of replays.

## Files

- `bin/replay-gl.jar`: OpenGL replayer.
- `bin/replay-vk.jar`: Vulkan replayer.
- `config.json`: local machine defaults. This file is ignored by git.
- `config.example.json`: portable template.
- `scripts/iris-control.ps1`: direct Iris capture control over the local HTTP backend.
- `scripts/run-replayer.ps1`: runs the OpenGL or Vulkan replayer.
- `scripts/capture-gputrace.ps1`: runs a replayer under Nsight GPU Trace.
- `<project-root>/.tmp/vibris/`: generated replay argfiles and replay AOT caches.

Always prefer `config.json` for `jdk` and `capture_path`. If a command does
not pass an explicit capture path, the scripts resolve `capture_path` as:

1. an exact capture directory when it contains `resource_metadata.json`;
2. otherwise, a root directory containing capture subdirectories, selecting the
   newest `resource_metadata.json`.

`replay_frames` controls the default frame count for both replayer scripts.
Prefer running several frames rather than a single frame; this gives the
driver and replay setup room to settle before the measured frame.

Both replay scripts write generated argfiles and replay AOT caches under the
project root at `.tmp/vibris/`, not under the skill directory. The current
files are `replay-<backend>.args` and `replay-<backend>.aot`.

## Iris Control

Use `scripts/iris-control.ps1` for the same operations exposed by
`iris_capture_mcp.py`. The script talks directly to Iris' local HTTP control
server and reads `iris_control_path` from `config.json`. The running Minecraft
client normally creates the control file at
`.minecraft\iris-capture-control.json` and must be running before the script
can reach Iris.

Available actions:

- `status`: checks the capture backend state.
- `reload`: reloads the active Iris shader pack.
- `capture-pass`: captures one compute pass.
- `capture-multi`: captures all compute dispatches in one composite-like
  program type: `prepare`, `begin`, `deferred`, or `composite`.

When profiling a known hotspot, prefer `capture-pass` for that single pass. It
keeps capture, replay, and GPU Trace turnaround much faster than
`capture-multi`. Use `capture-multi` when the target pass is not known yet,
when you need program-wide ordering context, or when the shader experiment may
reference resources that only appear in other passes.

If `-Path` is omitted, capture actions create a timestamped path under
`config.json`'s `capture_path`.

Capture requests are queued for the next rendered frame and saved
asynchronously. After queueing one, run `status` until `pending`, `active`, and
`saving` are all false. Stop on `lastError`; otherwise use `lastOutputPath` as
the capture directory to feed into the replayers. Resources are captured on
first reference during capture. If a replacement shader later references a
uniform or resource that was not captured, the replayer should fail instead of
continuing with invalid bindings.

## Replayer Usage

Use `scripts/run-replayer.ps1` with `-Backend gl` or `-Backend vk`. Omit
`-Capture` to use `config.json` capture selection, or pass a capture directory
explicitly. Use `-ShaderRoot <path>` for replacement shaderpack source and
`-ShaderPass <passName>` to limit replacement to selected passes.

The underlying replay CLI accepts
`<captureDir> [frameLimit] [--shader-path|--shader-root <path>] [--shader-pass <passName>]...`.

OpenGL replay uses the captured original OpenGL shader source unless
`--shader-root` is supplied. Vulkan replay uses runtime patching for shader
replacement. Do not manually copy edited shaderpack sources into the capture
directory; pass `-ShaderRoot` instead. When using a full shaderpack root to
test a known hotspot, also pass `-ShaderPass <passName>` so only that pass is
replaced and the other captured passes keep their original shader sources.

## Nsight GPU Trace

Use the bundled Nsight Analyzer at
`..\nsight-graphics-analyzer\scripts\nsight.py`. Do not read
`BASE/GPUTRACE_REGIMES.xls` directly; use the generated JSON files or the
`gputrace-stages`, `gputrace-actions`, and `gputrace-metric` analyzer
subcommands.

In most cases, capture GPU Trace with the OpenGL replayer. It uses the same
OpenGL shader compiler path as the target game and is the best default for
Alpha-Piscium performance work.

Use the Vulkan replayer only when you need finer Nsight information that is
harder to get from OpenGL, such as shader source profiling or per-line stall
reason detail inside one pass. In that case, run Vulkan trace with Nsight
multi-pass metrics and shader profiling options. Treat the result as a
diagnostic lens, not as exact OpenGL performance truth: Vulkan uses a different
driver compiler path, so shader codegen and stall behavior may differ from the
OpenGL shaderpack runtime.

Use `scripts/capture-gputrace.ps1` with `-Backend gl` by default. Select a
capture with `-Capture`, replacement source with `-ShaderRoot` and
`-ShaderPass`, and trace timing or limits with the parameters below. For
fine-grained Vulkan diagnostics, use `-Backend vk` with
`-MultiPassMetrics` and `-ShaderProfile`.

GPU Trace parameter coverage in `capture-gputrace.ps1`:

- Start trigger: `-StartAfterFrames`, `-StartAfterSubmits`, `-StartAfterMs`,
  `-StartAfterHotkey`, `-StartWithNgfxSdk`, `-StartOnReplayBegin`.
- Stop/limit: `-MaxDurationMs`, `-LimitToFrames`, `-LimitToSubmits`,
  `-StopWithNgfxSdk`, `-StopOnReplayEnd`.
- Buffers: `-AllocatedEventBufferMemoryKb`, `-AllocatedHesBufferMemoryKb`,
  `-AllocatedTimestamps`.
- Metrics: `-Architecture`, `-MetricSet`, `-MetricSetId`,
  `-PerArchConfigPath`, `-MultiPassMetrics`.
- Profiling: `-NoTimeEveryAction`, `-ShaderProfile`,
  `-PerLineActiveThreads`, `-PcSamplesPerPmIntervalPerSm`,
  `-PmBandwidthLimit`, `-HesEnabled`.
- Collection and runtime: `-SetGpuClocks`, `-CollectScreenshot`,
  `-DisableCollectShaderPipelines`,
  `-DisableCollectExternalShaderDebugInfo`,
  `-DisableTraceShaderBindings`, `-DisableNvtxRanges`,
  `-AllowTracingReplayReset`, `-KeepGoing`, `-TraceTimeout`,
  `-UseNgfxTimeout`, `-VerboseNgfx`, `-Timeout`, `-DryRun`.

All GPU Trace defaults live in `config.json` under `gpu_trace_args`. Command
line parameters override JSON only when explicitly provided; otherwise
`capture-gputrace.ps1` reads the matching JSON key. This makes the trace setup
editable by both users and agents without changing the script.

For GPU Trace, prefer `start_after_frames` instead of a millisecond delay.
The default is `start_after_frames = 3`, which skips the first few replay
frames and filters out initialization noise. Keep `replay_frames` larger than
the start frame plus the capture limit.

When `multi_pass_metrics` is enabled and `-Frames` is not explicitly passed,
`capture-gputrace.ps1` uses `gpu_trace_args.multi_pass_replay_frames`, default
`1000`. This keeps the replay process alive long enough for Nsight multi-pass
collection; the default process behavior is to exit after the GPU Trace capture
finishes.

Some raw `ngfx.exe --help-all` options are not exposed by the current Nsight
analyzer wrapper. Prefer the wrapper for normal work because it auto-exports
the trace and writes the summary/stages/actions JSON files.

Successful Nsight GPU Trace output includes:

- `<name>.ngfx-gputrace`
- `<name>.gputrace.summary.json`
- `<name>.gputrace.stages.json`
- `<name>.gputrace.actions.json`
- `BASE/*.xls`

### Replayer trace measurement contract

For `replay-gl.jar` and `replay-vk.jar` traces, use `summary.json` only to
confirm `bundle_complete=True`. Never use whole-capture duration,
relative-to-capture values, `analysis.frame_budget`, `fraction_of_gpu`,
replayer CPU submission, `replay-perf`, Copy work, or sleep/yield time as
shader performance evidence. Those values include replayer behavior and
unrelated scheduling noise that do not represent Iris runtime performance.

Report only:

- the duration of an individual pass inside the outer `Replay` marker; or
- that pass duration divided by the complete outer `Replay` marker duration.

Prefer `gputrace-actions <trace> --in-marker "^Replay$"` to select passes,
and use the stage tree to read the exact outer `Replay` duration. Exclude the
outer `Copy` marker and the unmarked tail sentinel dispatch from shader
comparisons. The sentinel exists only to absorb end-of-replay timing noise.

## Recommended Workflow

1. Ensure Minecraft/Iris is running in the target scene.
2. Reload the active shader pack through Iris control after shader source changes.
3. Capture only the target pass for a known hotspot; otherwise capture the
   relevant composite-like program type.
4. Wait for Iris status to report that capture and saving are complete.
5. Verify replay correctness with the OpenGL replayer.
6. Collect GPU Trace with the OpenGL replayer unless Vulkan-only diagnostics
   are required.
7. Confirm `bundle_complete=True`, then inspect only passes inside the outer
   `Replay` marker under the measurement contract above.

For shader experiments, edit shaderpack sources and pass `-ShaderRoot`; do not
recapture unless resource or uniform usage changed. If new shader code uses a
resource that was not referenced during capture, recapture that pass or
program type.
