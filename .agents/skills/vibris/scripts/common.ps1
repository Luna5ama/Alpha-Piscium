Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-VibrisSkillRoot {
    return (Split-Path -Parent $PSScriptRoot)
}

function Get-VibrisProjectRoot {
    $skillRoot = Get-VibrisSkillRoot
    return (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $skillRoot)))
}

function Get-VibrisTempRoot {
    $tempRoot = Join-Path (Join-Path (Get-VibrisProjectRoot) '.tmp') 'vibris'
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
    return (Resolve-Path -LiteralPath $tempRoot).Path
}

function Get-VibrisConfig {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    $configPath = Join-Path $Root 'config.json'
    if (-not (Test-Path -LiteralPath $configPath)) {
        $configPath = Join-Path $Root 'config.example.json'
    }
    return Get-Content -Raw -LiteralPath $configPath | ConvertFrom-Json
}

function Get-VibrisConfigValue {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Config,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [string]$Default = ''
    )

    $current = $Config
    foreach ($part in ($Name -split '\.')) {
        if ($null -eq $current) {
            return $Default
        }
        $property = $current.PSObject.Properties[$part]
        if ($null -eq $property) {
            return $Default
        }
        $current = $property.Value
    }
    if ($null -eq $current) {
        return $Default
    }
    return [string]$current
}

function Resolve-VibrisPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return Join-Path $Root $Path
}

function Resolve-VibrisJava {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Config
    )

    $jdk = Get-VibrisConfigValue -Config $Config -Name 'jdk'
    if ($jdk) {
        $java = $jdk
        if ((Test-Path -LiteralPath $jdk -PathType Container)) {
            $java = Join-Path $jdk 'bin\java.exe'
        }
        if (Test-Path -LiteralPath $java -PathType Leaf) {
            return (Resolve-Path -LiteralPath $java).Path
        }
        throw "Configured jdk does not contain java.exe: $jdk"
    }
    return 'java.exe'
}

function Resolve-VibrisCapture {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Config,
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [string]$Capture
    )

    $candidate = $Capture
    if (-not $candidate) {
        $candidate = Get-VibrisConfigValue -Config $Config -Name 'capture_path'
    }
    if (-not $candidate) {
        throw 'No capture path was passed and config.json has no capture_path.'
    }

    $candidate = Resolve-VibrisPath -Root $Root -Path $candidate
    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "Capture path does not exist: $candidate"
    }

    $resolved = (Resolve-Path -LiteralPath $candidate).Path
    $metadata = Join-Path $resolved 'resource_metadata.json'
    if (Test-Path -LiteralPath $metadata -PathType Leaf) {
        return $resolved
    }

    $latest = Get-ChildItem -LiteralPath $resolved -Filter 'resource_metadata.json' -Recurse -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No resource_metadata.json found under capture path: $resolved"
    }
    return $latest.Directory.FullName
}

function Resolve-VibrisCaptureRoot {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Config,
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    $candidate = Get-VibrisConfigValue -Config $Config -Name 'capture_path'
    if (-not $candidate) {
        throw 'config.json has no capture_path.'
    }
    $candidate = Resolve-VibrisPath -Root $Root -Path $candidate
    if (Test-Path -LiteralPath (Join-Path $candidate 'resource_metadata.json') -PathType Leaf) {
        return (Split-Path -Parent (Resolve-Path -LiteralPath $candidate).Path)
    }
    New-Item -ItemType Directory -Force -Path $candidate | Out-Null
    return (Resolve-Path -LiteralPath $candidate).Path
}

function New-VibrisCaptureOutputPath {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Config,
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $captureRoot = Resolve-VibrisCaptureRoot -Config $Config -Root $Root
    $safeName = $Name -replace '[^A-Za-z0-9_.-]', '-'
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    return (Join-Path $captureRoot "$safeName-$stamp")
}

function Resolve-VibrisReplayJar {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [ValidateSet('gl', 'vk')]
        [string]$Backend
    )

    $name = if ($Backend -eq 'gl') {
        'replay-gl.jar'
    } else {
        'replay-vk.jar'
    }
    $jar = Join-Path (Join-Path $Root 'bin') $name
    if (-not (Test-Path -LiteralPath $jar -PathType Leaf)) {
        throw "Replay jar is missing: $jar"
    }
    return (Resolve-Path -LiteralPath $jar).Path
}

function Get-VibrisReplayAotCachePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Jar
    )

    return (Join-Path (Get-VibrisTempRoot) (([System.IO.Path]::GetFileNameWithoutExtension($Jar)) + '.aot'))
}

function Ensure-VibrisJavaAotCache {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Java,
        [Parameter(Mandatory = $true)]
        [string]$Jar,
        [Parameter(Mandatory = $true)]
        [string]$AotCache,
        [Parameter(Mandatory = $true)]
        [string]$ArgFile,
        [string]$Name = 'Replay'
    )

    if (Test-Path -LiteralPath $AotCache -PathType Leaf) {
        if ((Get-Item -LiteralPath $AotCache).LastWriteTimeUtc -ge (Get-Item -LiteralPath $Jar).LastWriteTimeUtc) {
            return 0
        }
        Remove-Item -LiteralPath $AotCache
    }

    Write-Host "$Name AOT cache missing or stale, running once to generate it..."

    $aotArgs = @(
        "-XX:AOTCacheOutput=$AotCache",
        "@$ArgFile"
    )

    & $Java @aotArgs | Out-Host
    return $LASTEXITCODE
}

function New-VibrisJavaArgFile {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet('gl', 'vk')]
        [string]$Backend,
        [Parameter(Mandatory = $true)]
        [string]$Jar,
        [Parameter(Mandatory = $true)]
        [string]$Capture,
        [long]$Frames = 1,
        [string[]]$JvmArg = @(),
        [string]$ShaderRoot,
        [string[]]$ShaderPass = @()
    )

    $tempDir = Get-VibrisTempRoot
    $argFile = Join-Path $tempDir ("replay-$Backend.args")

    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($arg in $JvmArg) {
        if ($arg) {
            $lines.Add($arg)
        }
    }
    $lines.Add('-jar')
    $lines.Add($Jar)
    $lines.Add($Capture)
    $lines.Add([string]$Frames)
    if ($ShaderRoot) {
        $lines.Add('--shader-root')
        $lines.Add($ShaderRoot)
    }
    foreach ($pass in $ShaderPass) {
        if ($pass) {
            $lines.Add('--shader-pass')
            $lines.Add($pass)
        }
    }

    [System.IO.File]::WriteAllLines($argFile, $lines, [System.Text.Encoding]::ASCII)
    return $argFile
}
