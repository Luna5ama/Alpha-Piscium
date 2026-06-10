param(
    [ValidateSet('gl', 'vk')]
    [string]$Backend = 'gl',
    [string]$Capture = '',
    [string]$ShaderRoot = '',
    [long]$Frames = -1,
    [int]$StartAfterFrames = -1,
    [int]$StartAfterSubmits = -1,
    [int]$StartAfterMs = -1,
    [switch]$StartAfterHotkey,
    [switch]$StartWithNgfxSdk,
    [switch]$StartOnReplayBegin,
    [int]$MaxDurationMs = -1,
    [int]$LimitToFrames = -1,
    [int]$LimitToSubmits = -1,
    [switch]$StopWithNgfxSdk,
    [switch]$StopOnReplayEnd,
    [int]$AllocatedEventBufferMemoryKb = -1,
    [int]$AllocatedHesBufferMemoryKb = -1,
    [int]$AllocatedTimestamps = -1,
    [string]$Architecture = '',
    [string]$MetricSet = '',
    [int]$MetricSetId = -1,
    [string]$PerArchConfigPath = '',
    [string]$OutDir = '',
    [int]$Timeout = -1,
    [switch]$NoTimeEveryAction,
    [switch]$MultiPassMetrics,
    [ValidateSet('', 'unaltered', 'base', 'boost')]
    [string]$SetGpuClocks = '',
    [switch]$ShaderProfile,
    [switch]$PerLineActiveThreads,
    [int]$PcSamplesPerPmIntervalPerSm = -1,
    [int]$PmBandwidthLimit = -1,
    [int]$HesEnabled = -1,
    [int]$CollectScreenshot = -1,
    [switch]$DisableCollectShaderPipelines,
    [switch]$DisableCollectExternalShaderDebugInfo,
    [switch]$DisableTraceShaderBindings,
    [switch]$DisableNvtxRanges,
    [int]$AllowTracingReplayReset = -1,
    [switch]$KeepGoing,
    [int]$TraceTimeout = -1,
    [switch]$UseNgfxTimeout,
    [switch]$VerboseNgfx,
    [switch]$DryRun
)

. "$PSScriptRoot\common.ps1"

$root = Get-VibrisSkillRoot
$config = Get-VibrisConfig -Root $root

function Get-GpuTraceConfigValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [string]$Default = ''
    )

    return Get-VibrisConfigValue -Config $config -Name "gpu_trace_args.$Name" -Default $Default
}

function Convert-VibrisConfigBool {
    param(
        [string]$Value,
        [bool]$Default = $false
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $Default
    }

    switch ($Value.Trim().ToLowerInvariant()) {
        'true' { return $true }
        '1' { return $true }
        'yes' { return $true }
        'on' { return $true }
        'false' { return $false }
        '0' { return $false }
        'no' { return $false }
        'off' { return $false }
        default { throw "Invalid boolean gpu_trace_args value: $Value" }
    }
}

function Get-GpuTraceInt {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ParameterName,
        [Parameter(Mandatory = $true)]
        [string]$ConfigName,
        [Parameter(Mandatory = $true)]
        [int]$Value,
        [int]$Default = -1
    )

    if ($PSBoundParametersGlobal.ContainsKey($ParameterName)) {
        return $Value
    }

    $configValue = Get-GpuTraceConfigValue -Name $ConfigName
    if ([string]::IsNullOrWhiteSpace($configValue)) {
        return $Default
    }
    return [int]$configValue
}

function Get-GpuTraceString {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ParameterName,
        [Parameter(Mandatory = $true)]
        [string]$ConfigName,
        [string]$Value = '',
        [string]$Default = ''
    )

    if ($PSBoundParametersGlobal.ContainsKey($ParameterName)) {
        return $Value
    }

    $configValue = Get-GpuTraceConfigValue -Name $ConfigName
    if ([string]::IsNullOrWhiteSpace($configValue)) {
        return $Default
    }
    return $configValue
}

function Get-GpuTraceSwitch {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ParameterName,
        [Parameter(Mandatory = $true)]
        [string]$ConfigName,
        [Parameter(Mandatory = $true)]
        [switch]$Value,
        [bool]$Default = $false
    )

    if ($PSBoundParametersGlobal.ContainsKey($ParameterName)) {
        return [bool]$Value
    }

    $configValue = Get-GpuTraceConfigValue -Name $ConfigName
    return Convert-VibrisConfigBool -Value $configValue -Default $Default
}

$PSBoundParametersGlobal = $PSBoundParameters

$StartAfterFrames = Get-GpuTraceInt -ParameterName 'StartAfterFrames' -ConfigName 'start_after_frames' -Value $StartAfterFrames
$StartAfterSubmits = Get-GpuTraceInt -ParameterName 'StartAfterSubmits' -ConfigName 'start_after_submits' -Value $StartAfterSubmits
$StartAfterHotkeyValue = Get-GpuTraceSwitch -ParameterName 'StartAfterHotkey' -ConfigName 'start_after_hotkey' -Value $StartAfterHotkey
$StartWithNgfxSdkValue = Get-GpuTraceSwitch -ParameterName 'StartWithNgfxSdk' -ConfigName 'start_with_ngfx_sdk' -Value $StartWithNgfxSdk
$StartOnReplayBeginValue = Get-GpuTraceSwitch -ParameterName 'StartOnReplayBegin' -ConfigName 'start_on_replay_begin' -Value $StartOnReplayBegin

$explicitStart = $PSBoundParameters.ContainsKey('StartAfterFrames') -or
    $PSBoundParameters.ContainsKey('StartAfterSubmits') -or
    $PSBoundParameters.ContainsKey('StartAfterMs') -or
    $PSBoundParameters.ContainsKey('StartAfterHotkey') -or
    $PSBoundParameters.ContainsKey('StartWithNgfxSdk') -or
    $PSBoundParameters.ContainsKey('StartOnReplayBegin')
$StartAfterMs = Get-GpuTraceInt -ParameterName 'StartAfterMs' -ConfigName 'start_after_ms' -Value $StartAfterMs
if ($explicitStart) {
    if (-not $PSBoundParameters.ContainsKey('StartAfterFrames')) { $StartAfterFrames = -1 }
    if (-not $PSBoundParameters.ContainsKey('StartAfterSubmits')) { $StartAfterSubmits = -1 }
    if (-not $PSBoundParameters.ContainsKey('StartAfterMs')) { $StartAfterMs = -1 }
    if (-not $PSBoundParameters.ContainsKey('StartAfterHotkey')) { $StartAfterHotkeyValue = $false }
    if (-not $PSBoundParameters.ContainsKey('StartWithNgfxSdk')) { $StartWithNgfxSdkValue = $false }
    if (-not $PSBoundParameters.ContainsKey('StartOnReplayBegin')) { $StartOnReplayBeginValue = $false }
}
if (
    $StartAfterFrames -lt 0 -and
    $StartAfterSubmits -lt 0 -and
    $StartAfterMs -lt 0 -and
    -not $StartAfterHotkeyValue -and
    -not $StartWithNgfxSdkValue -and
    -not $StartOnReplayBeginValue
) {
    $StartAfterMs = 2000
}

$MaxDurationMs = Get-GpuTraceInt -ParameterName 'MaxDurationMs' -ConfigName 'max_duration_ms' -Value $MaxDurationMs -Default 3000
$LimitToFrames = Get-GpuTraceInt -ParameterName 'LimitToFrames' -ConfigName 'limit_to_frames' -Value $LimitToFrames
$LimitToSubmits = Get-GpuTraceInt -ParameterName 'LimitToSubmits' -ConfigName 'limit_to_submits' -Value $LimitToSubmits
$StopWithNgfxSdkValue = Get-GpuTraceSwitch -ParameterName 'StopWithNgfxSdk' -ConfigName 'stop_with_ngfx_sdk' -Value $StopWithNgfxSdk
$StopOnReplayEndValue = Get-GpuTraceSwitch -ParameterName 'StopOnReplayEnd' -ConfigName 'stop_on_replay_end' -Value $StopOnReplayEnd
$explicitStop = $PSBoundParameters.ContainsKey('LimitToFrames') -or
    $PSBoundParameters.ContainsKey('LimitToSubmits') -or
    $PSBoundParameters.ContainsKey('StopWithNgfxSdk') -or
    $PSBoundParameters.ContainsKey('StopOnReplayEnd')
if ($explicitStop) {
    if (-not $PSBoundParameters.ContainsKey('LimitToFrames')) { $LimitToFrames = -1 }
    if (-not $PSBoundParameters.ContainsKey('LimitToSubmits')) { $LimitToSubmits = -1 }
    if (-not $PSBoundParameters.ContainsKey('StopWithNgfxSdk')) { $StopWithNgfxSdkValue = $false }
    if (-not $PSBoundParameters.ContainsKey('StopOnReplayEnd')) { $StopOnReplayEndValue = $false }
}

$AllocatedEventBufferMemoryKb = Get-GpuTraceInt -ParameterName 'AllocatedEventBufferMemoryKb' -ConfigName 'allocated_event_buffer_memory_kb' -Value $AllocatedEventBufferMemoryKb
$AllocatedHesBufferMemoryKb = Get-GpuTraceInt -ParameterName 'AllocatedHesBufferMemoryKb' -ConfigName 'allocated_hes_buffer_memory_kb' -Value $AllocatedHesBufferMemoryKb
$AllocatedTimestamps = Get-GpuTraceInt -ParameterName 'AllocatedTimestamps' -ConfigName 'allocated_timestamps' -Value $AllocatedTimestamps
$Architecture = Get-GpuTraceString -ParameterName 'Architecture' -ConfigName 'gpu_architecture' -Value $Architecture
$MetricSet = Get-GpuTraceString -ParameterName 'MetricSet' -ConfigName 'metric_set' -Value $MetricSet -Default 'Throughput Metrics'
$MetricSetId = Get-GpuTraceInt -ParameterName 'MetricSetId' -ConfigName 'metric_set_id' -Value $MetricSetId
$PerArchConfigPath = Get-GpuTraceString -ParameterName 'PerArchConfigPath' -ConfigName 'per_arch_config_path' -Value $PerArchConfigPath

$timeEveryAction = $true
if ($PSBoundParameters.ContainsKey('NoTimeEveryAction')) {
    $timeEveryAction = -not [bool]$NoTimeEveryAction
} else {
    $timeEveryAction = Convert-VibrisConfigBool -Value (Get-GpuTraceConfigValue -Name 'time_every_action') -Default $true
}

$MultiPassMetricsValue = Get-GpuTraceSwitch -ParameterName 'MultiPassMetrics' -ConfigName 'multi_pass_metrics' -Value $MultiPassMetrics
$defaultReplayFrames = if ($MultiPassMetricsValue) {
    Get-GpuTraceConfigValue -Name 'multi_pass_replay_frames' -Default '1000'
} else {
    Get-VibrisConfigValue -Config $config -Name 'replay_frames' -Default '8'
}
if (-not $PSBoundParameters.ContainsKey('Frames')) {
    $Frames = [long]$defaultReplayFrames
}
$SetGpuClocks = Get-GpuTraceString -ParameterName 'SetGpuClocks' -ConfigName 'set_gpu_clocks' -Value $SetGpuClocks
$ShaderProfileValue = Get-GpuTraceSwitch -ParameterName 'ShaderProfile' -ConfigName 'real_time_shader_profiler' -Value $ShaderProfile
$PerLineActiveThreadsValue = Get-GpuTraceSwitch -ParameterName 'PerLineActiveThreads' -ConfigName 'per_line_active_threads_per_warp' -Value $PerLineActiveThreads
$PcSamplesPerPmIntervalPerSm = Get-GpuTraceInt -ParameterName 'PcSamplesPerPmIntervalPerSm' -ConfigName 'pc_samples_per_pm_interval_per_sm' -Value $PcSamplesPerPmIntervalPerSm
$PmBandwidthLimit = Get-GpuTraceInt -ParameterName 'PmBandwidthLimit' -ConfigName 'pm_bandwidth_limit' -Value $PmBandwidthLimit
$HesEnabled = Get-GpuTraceInt -ParameterName 'HesEnabled' -ConfigName 'hes_enabled' -Value $HesEnabled
$CollectScreenshot = Get-GpuTraceInt -ParameterName 'CollectScreenshot' -ConfigName 'collect_screenshot' -Value $CollectScreenshot
$DisableCollectShaderPipelinesValue = Get-GpuTraceSwitch -ParameterName 'DisableCollectShaderPipelines' -ConfigName 'disable_collect_shader_pipelines' -Value $DisableCollectShaderPipelines
$DisableCollectExternalShaderDebugInfoValue = Get-GpuTraceSwitch -ParameterName 'DisableCollectExternalShaderDebugInfo' -ConfigName 'disable_collect_external_shader_debug_info' -Value $DisableCollectExternalShaderDebugInfo
$DisableTraceShaderBindingsValue = Get-GpuTraceSwitch -ParameterName 'DisableTraceShaderBindings' -ConfigName 'disable_trace_shader_bindings' -Value $DisableTraceShaderBindings
$DisableNvtxRangesValue = Get-GpuTraceSwitch -ParameterName 'DisableNvtxRanges' -ConfigName 'disable_nvtx_ranges' -Value $DisableNvtxRanges
$AllowTracingReplayReset = Get-GpuTraceInt -ParameterName 'AllowTracingReplayReset' -ConfigName 'allow_tracing_replay_reset' -Value $AllowTracingReplayReset
$KeepGoingValue = Get-GpuTraceSwitch -ParameterName 'KeepGoing' -ConfigName 'keep_going' -Value $KeepGoing
$TraceTimeout = Get-GpuTraceInt -ParameterName 'TraceTimeout' -ConfigName 'trace_timeout' -Value $TraceTimeout
$UseNgfxTimeoutValue = Get-GpuTraceSwitch -ParameterName 'UseNgfxTimeout' -ConfigName 'use_ngfx_timeout' -Value $UseNgfxTimeout
$VerboseNgfxValue = Get-GpuTraceSwitch -ParameterName 'VerboseNgfx' -ConfigName 'verbose' -Value $VerboseNgfx
$Timeout = Get-GpuTraceInt -ParameterName 'Timeout' -ConfigName 'timeout' -Value $Timeout -Default 180

$java = Resolve-VibrisJava -Config $config
$captureDir = Resolve-VibrisCapture -Config $config -Root $root -Capture $Capture
$jar = Resolve-VibrisReplayJar -Root $root -Backend $Backend

if ($ShaderRoot) {
    $ShaderRoot = Resolve-VibrisPath -Root $root -Path $ShaderRoot
}

$argFile = New-VibrisJavaArgFile -Root $root -Backend $Backend -Jar $jar -Capture $captureDir -Frames $Frames -ShaderRoot $ShaderRoot

$nsightScript = Get-VibrisConfigValue -Config $config -Name 'nsight_script' -Default '..\nsight-graphics-analyzer\scripts\nsight.py'
$nsightScript = Resolve-VibrisPath -Root $root -Path $nsightScript
if (-not (Test-Path -LiteralPath $nsightScript -PathType Leaf)) {
    throw "Nsight analyzer script not found: $nsightScript"
}

if (-not $OutDir) {
    $OutDir = Get-VibrisConfigValue -Config $config -Name 'trace_output_path'
}
if (-not $OutDir) {
    $OutDir = Join-Path (Split-Path -Parent $captureDir) 'nsight-traces'
}
$OutDir = Resolve-VibrisPath -Root $root -Path $OutDir
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$out = Join-Path $OutDir ("vibris-$Backend-$stamp.ngfx-gputrace")

$argsList = @(
    $nsightScript,
    'gputrace-capture',
    '--exe', $java,
    '--wd', $root,
    '--args', "@$argFile",
    '--max-duration-ms', [string]$MaxDurationMs,
    '--out', $out,
    '--timeout', [string]$Timeout
)

$startTriggerCount = 0
if ($StartAfterFrames -ge 0) { $startTriggerCount++ }
if ($StartAfterSubmits -ge 0) { $startTriggerCount++ }
if ($StartAfterMs -ge 0) { $startTriggerCount++ }
if ($StartAfterHotkeyValue) { $startTriggerCount++ }
if ($StartWithNgfxSdkValue) { $startTriggerCount++ }
if ($StartOnReplayBeginValue) { $startTriggerCount++ }
if ($startTriggerCount -ne 1) {
    throw "Choose exactly one start trigger; resolved $startTriggerCount."
}
if ($StartAfterFrames -ge 0) {
    $argsList += @('--start-after-frames', [string]$StartAfterFrames)
} elseif ($StartAfterSubmits -ge 0) {
    $argsList += @('--start-after-submits', [string]$StartAfterSubmits)
} elseif ($StartAfterMs -ge 0) {
    $argsList += @('--start-after-ms', [string]$StartAfterMs)
} elseif ($StartAfterHotkeyValue) {
    $argsList += '--start-after-hotkey'
} elseif ($StartWithNgfxSdkValue) {
    $argsList += '--start-with-ngfx-sdk'
} elseif ($StartOnReplayBeginValue) {
    $argsList += '--start-on-replay-begin'
}

$stopLimitCount = 0
if ($LimitToFrames -ge 0) { $stopLimitCount++ }
if ($LimitToSubmits -ge 0) { $stopLimitCount++ }
if ($StopWithNgfxSdkValue) { $stopLimitCount++ }
if ($StopOnReplayEndValue) { $stopLimitCount++ }
if ($stopLimitCount -gt 1) {
    throw 'Choose at most one stop limit.'
}
if ($LimitToFrames -ge 0) {
    $argsList += @('--limit-to-frames', [string]$LimitToFrames)
} elseif ($LimitToSubmits -ge 0) {
    $argsList += @('--limit-to-submits', [string]$LimitToSubmits)
} elseif ($StopWithNgfxSdkValue) {
    $argsList += '--stop-with-ngfx-sdk'
} elseif ($StopOnReplayEndValue) {
    $argsList += '--stop-on-replay-end'
}

if ($AllocatedEventBufferMemoryKb -ge 0) {
    $argsList += @('--allocated-event-buffer-memory-kb', [string]$AllocatedEventBufferMemoryKb)
}
if ($AllocatedHesBufferMemoryKb -ge 0) {
    $argsList += @('--allocated-hes-buffer-memory-kb', [string]$AllocatedHesBufferMemoryKb)
}
if ($AllocatedTimestamps -ge 0) {
    $argsList += @('--allocated-timestamps', [string]$AllocatedTimestamps)
}
if ($Architecture) {
    $argsList += @('--architecture', $Architecture)
}
if ($PerArchConfigPath) {
    $argsList += @('--per-arch-config-path', (Resolve-VibrisPath -Root $root -Path $PerArchConfigPath))
} elseif ($MetricSetId -ge 0) {
    $argsList += @('--metric-set-id', [string]$MetricSetId)
} elseif ($MetricSet) {
    $argsList += @('--metric-set-name', $MetricSet)
}
if ($MultiPassMetricsValue) {
    $argsList += '--multi-pass-metrics'
}
if ($timeEveryAction) {
    $argsList += '--time-every-action'
}
if ($ShaderProfileValue) {
    $argsList += '--real-time-shader-profiler'
}
if ($PerLineActiveThreadsValue) {
    $argsList += '--per-line-active-threads-per-warp'
}
if ($PcSamplesPerPmIntervalPerSm -ge 0) {
    $argsList += @('--pc-samples-per-pm-interval-per-sm', [string]$PcSamplesPerPmIntervalPerSm)
}
if ($PmBandwidthLimit -ge 0) {
    $argsList += @('--pm-bandwidth-limit', [string]$PmBandwidthLimit)
}
if ($HesEnabled -ge 0) {
    $argsList += @('--hes-enabled', [string]$HesEnabled)
}
if ($SetGpuClocks) {
    $argsList += @('--set-gpu-clocks', $SetGpuClocks)
}
if ($CollectScreenshot -ge 0) {
    $argsList += @('--collect-screenshot', [string]$CollectScreenshot)
}
if ($DisableCollectShaderPipelinesValue) {
    $argsList += '--disable-collect-shader-pipelines'
}
if ($DisableCollectExternalShaderDebugInfoValue) {
    $argsList += '--disable-collect-external-shader-debug-info'
}
if ($DisableTraceShaderBindingsValue) {
    $argsList += '--disable-trace-shader-bindings'
}
if ($DisableNvtxRangesValue) {
    $argsList += '--disable-nvtx-ranges'
}
if ($AllowTracingReplayReset -ge 0) {
    $argsList += @('--allow-tracing-replay-reset', [string]$AllowTracingReplayReset)
}
if ($KeepGoingValue) {
    $argsList += '--keep-going'
}
if ($TraceTimeout -ge 0) {
    $argsList += @('--trace-timeout', [string]$TraceTimeout)
}
if ($UseNgfxTimeoutValue) {
    $argsList += '--use-ngfx-timeout'
}
if ($VerboseNgfxValue) {
    $argsList += '--verbose'
}
if ($DryRun) {
    $argsList += '--dry-run'
}

& python @argsList
exit $LASTEXITCODE
