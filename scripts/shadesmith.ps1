param()

$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}

$projectRoot = Split-Path -Parent $scriptDir
$configPath = Join-Path $scriptDir 'config.properties'
$shadesmithJarPath = Join-Path $scriptDir 'shadesmith.jar'
$shadesmithAotCachePath = Join-Path $scriptDir 'shadesmith.aot'
$shadersPath = Join-Path $projectRoot 'shaders'

function Read-PropertiesFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $result = @{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $result
    }

    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#') -or $trimmed.StartsWith('!')) {
            continue
        }

        $separatorIndex = $trimmed.IndexOf('=')
        if ($separatorIndex -lt 0) {
            continue
        }

        $key = $trimmed.Substring(0, $separatorIndex).Trim()
        $value = $trimmed.Substring($separatorIndex + 1).Trim()
        if ($key) {
            $result[$key] = $value
        }
    }

    return $result
}

function Get-JavaHome {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Config
    )

    if ($Config.ContainsKey('JAVA_PATH') -and -not [string]::IsNullOrWhiteSpace([string]$Config['JAVA_PATH'])) {
        return [string]$Config['JAVA_PATH']
    }

    $javaCommand = Get-Command java -ErrorAction SilentlyContinue
    if (-not $javaCommand) {
        throw 'JAVA_PATH is not set and java is not available on PATH.'
    }

    $javaOutput = & java -XshowSettings:properties -version 2>&1
    $javaHomeLine = $javaOutput | Select-String -Pattern '^\s*java\.home\s*=\s*(.+)\s*$' | Select-Object -First 1
    if (-not $javaHomeLine) {
        throw 'Unable to determine java.home from the active java command.'
    }

    return $javaHomeLine.Matches[0].Groups[1].Value.Trim()
}

$config = Read-PropertiesFile -Path $configPath
$javaHome = Get-JavaHome -Config $config
$javaExe = Join-Path $javaHome 'bin\java.exe'

if (-not (Test-Path -LiteralPath $javaExe)) {
    throw "Java executable not found: $javaExe"
}

$shadesmithOutputPathStr = if ($config.ContainsKey('SHADESMITH_OUTPUT') -and -not [string]::IsNullOrWhiteSpace([string]$config['SHADESMITH_OUTPUT'])) {
    [string]$config['SHADESMITH_OUTPUT']
} else {
    './shadesmitth'
}

$shadesmithOutputPath = if ([System.IO.Path]::IsPathRooted($shadesmithOutputPathStr)) {
    [System.IO.Path]::GetFullPath($shadesmithOutputPathStr)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $scriptDir $shadesmithOutputPathStr))
}
$shadesmithShadersPath = Join-Path $shadesmithOutputPath 'shaders'

if (-not (Test-Path -LiteralPath $shadesmithAotCachePath)) {
    Write-Host 'Shadesmith AOT cache not found, running once to generate it...'
    $aotArgs = @(
        '--add-modules',
        'jdk.internal.vm.ci',
        "-XX:AOTCacheOutput=$shadesmithAotCachePath",
        '-jar',
        $shadesmithJarPath,
        $shadersPath,
        $shadesmithShadersPath
    )

    & $javaExe @aotArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$runArgs = @(
    "-XX:AOTCache=$shadesmithAotCachePath",
    '-jar',
    $shadesmithJarPath,
    $shadersPath,
    $shadesmithShadersPath
)

& $javaExe @runArgs
exit $LASTEXITCODE
