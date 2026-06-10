---
name: vibris
description: Use Iris capture control, glc2vk OpenGL/Vulkan replayers, and Nsight GPU Trace for Alpha-Piscium compute shader debugging and profiling.
metadata:
  version: 0.1.0
  categories:
    - minecraft
    - iris
    - glc2vk
    - gpu
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

Always prefer `config.json` for `jdk` and `capture_path`. If a command does
not pass an explicit capture path, the scripts resolve `capture_path` as:

1. an exact capture directory when it contains `resource_metadata.json`;
2. otherwise, a root directory containing capture subdirectories, selecting the
   newest `resource_metadata.json`.

`replay_frames` controls the default frame count for both replayer scripts.
Prefer running several frames rather than a single frame; this gives the
driver and replay setup room to settle before the measured frame.

## Iris Control

Use `scripts/iris-control.ps1` for the same operations exposed by
`iris_capture_mcp.py`. The script talks directly to Iris' local HTTP control
server and reads the control file from `config.json`:

```json
{
  "iris_control_path": "I:\\code\\Iris\\fabric\\run\\iris-capture-control.json"
}
```

The Minecraft client must be running before the script can reach Iris. The
control file is normally created by the running client at:

```text
I:\code\Iris\fabric\run\iris-capture-control.json
```

Available actions:

- `status`: checks the capture backend state.
- `reload`: reloads the active Iris shader pack.
- `capture-pass`: captures one compute pass.
- `capture-multi`: captures all compute dispatches in one composite-like
  program type: `prepare`, `begin`, `deferred`, or `composite`.

When profiling a known hotspot such as `composite20`, prefer `capture-pass`
for that single pass. It keeps capture, replay, and GPU Trace turnaround much
faster than `capture-multi`. Use `capture-multi` when the target pass is not
known yet, when you need program-wide ordering context, or when the shader
experiment may reference resources that only appear in other passes.

Examples:

```powershell
I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\iris-control.ps1 -Action status

I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\iris-control.ps1 -Action reload

I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\iris-control.ps1 `
  -Action capture-pass `
  -Pass composite20

I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\iris-control.ps1 `
  -Action capture-multi `
  -Type composite
```

If `-Path` is omitted, capture actions create a timestamped path under
`config.json`'s `capture_path`, for example:

```text
R:\vibris\captures\composite-20260610-050000
```

The returned directory is the capture directory to feed into the replayers.
Resources are captured on first reference during capture. If a replacement
shader later references a uniform or resource that was not captured, the
replayer should fail instead of continuing with invalid bindings.

The in-game command is also available:

```text
/capturemulti <prepare|begin|deferred|composite>
```

## Replayer Usage

Run OpenGL replay with defaults from `config.json`:

```powershell
I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\run-replayer.ps1 -Backend gl
```

Run Vulkan replay:

```powershell
I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\run-replayer.ps1 -Backend vk
```

Run a specific capture:

```powershell
I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\run-replayer.ps1 `
  -Backend gl `
  -Capture R:\vibris\composite-20260610-050000
```

Run with replacement shader source:

```powershell
I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\run-replayer.ps1 `
  -Backend gl `
  -Capture R:\vibris\composite-20260610-050000 `
  -ShaderRoot I:\code\mcshaders\Alpha-Piscium\shaders
```

The replay CLI accepts:

```text
<captureDir> [frameLimit] [--shader-path|--shader-root <path>]
```

OpenGL replay uses the captured original OpenGL shader source unless
`--shader-root` is supplied. Vulkan replay uses runtime patching for shader
replacement. Do not manually copy edited shaderpack sources into the capture
directory; pass `-ShaderRoot` instead.

## Nsight GPU Trace

Use the bundled Nsight Analyzer skill through:

```text
..\nsight-graphics-analyzer\scripts\nsight.py
```

Do not read `BASE/GPUTRACE_REGIMES.xls` directly. It is large. Use the
generated JSON files or the analyzer subcommands:

```powershell
python ..\nsight-graphics-analyzer\scripts\nsight.py `
  gputrace-stages <trace.ngfx-gputrace> --top 20
```

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

To capture a replay under Nsight using `config.json` defaults:

```powershell
I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\capture-gputrace.ps1 -Backend gl
```

Specific capture:

```powershell
I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\capture-gputrace.ps1 `
  -Backend gl `
  -Capture R:\vibris\composite-20260610-050000 `
  -StartAfterFrames 3 `
  -MaxDurationMs 3000
```

Fine-grained Vulkan trace:

```powershell
I:\code\mcshaders\Alpha-Piscium\.agents\skills\vibris\scripts\capture-gputrace.ps1 `
  -Backend vk `
  -Capture R:\vibris\composite-20260610-050000 `
  -MultiPassMetrics `
  -ShaderProfile
```

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

Some raw `ngfx.exe --help-all` options, such as `--num-frames`,
D3D12-specific flags, or
`--disable-force-recompile-cached-vk-shader-stage-modules`, are not exposed by
the current Nsight analyzer wrapper. Prefer the wrapper for normal work because
it auto-exports the trace and writes the summary/stages/actions JSON files.

The script creates a Java argfile and passes it to Nsight as:

```text
--args @<generated-argfile>
```

This is intentional. Passing `--args "-jar ..."` through the Nsight wrapper can
quote the entire Java command as one argument, causing Java to treat it as a
class name and exit before trace collection starts.

Successful Nsight GPU Trace output includes:

- `<name>.ngfx-gputrace`
- `<name>.gputrace.summary.json`
- `<name>.gputrace.stages.json`
- `<name>.gputrace.actions.json`
- `BASE/*.xls`

Read `summary.json` first, then drill with `gputrace-stages`,
`gputrace-actions`, or `gputrace-metric`.

## Recommended Workflow

1. Ensure Minecraft/Iris is running in the target scene.
2. Use `scripts/iris-control.ps1 -Action reload` when shaderpack source changed.
3. If profiling a known hotspot pass, use
   `scripts/iris-control.ps1 -Action capture-pass -Pass <passName>` to capture
   only that pass into a unique path under `config.json`'s `capture_path`.
   Otherwise, use `scripts/iris-control.ps1 -Action capture-multi -Type composite`.
4. Run `scripts/run-replayer.ps1 -Backend gl -Capture <captureDir>` to verify
   replay correctness.
5. Run `scripts/capture-gputrace.ps1 -Backend gl -Capture <captureDir>` to
   collect GPU Trace for the replayer.
6. Inspect the generated summary JSON first. Use analyzer drill-down commands
   only after confirming `bundle_complete=True`.

For shader experiments, edit shaderpack sources and pass `-ShaderRoot`; do not
recapture unless resource or uniform usage changed. If new shader code uses a
resource that was not referenced during capture, recapture that pass or
program type.
