param(
    [ValidateSet('gl', 'vk')]
    [string]$Backend = 'gl',
    [string]$Capture = '',
    [string]$ShaderRoot = '',
    [string[]]$ShaderPass = @(),
    [long]$Frames = -1,
    [switch]$PrintCommand
)

. "$PSScriptRoot\common.ps1"

$root = Get-VibrisSkillRoot
$config = Get-VibrisConfig -Root $root
$java = Resolve-VibrisJava -Config $config
$captureDir = Resolve-VibrisCapture -Config $config -Root $root -Capture $Capture
$jar = Resolve-VibrisReplayJar -Root $root -Backend $Backend

if (-not $PSBoundParameters.ContainsKey('Frames')) {
    $Frames = [long](Get-VibrisConfigValue -Config $config -Name 'replay_frames' -Default '8')
}

if ($ShaderRoot) {
    $ShaderRoot = Resolve-VibrisPath -Root $root -Path $ShaderRoot
}

$argFile = New-VibrisJavaArgFile -Backend $Backend -Jar $jar -Capture $captureDir -Frames $Frames -ShaderRoot $ShaderRoot -ShaderPass $ShaderPass
$aotCache = Get-VibrisReplayAotCachePath -Jar $jar

if (-not $PrintCommand) {
    $cacheExitCode = Ensure-VibrisJavaAotCache -Java $java -AotCache $aotCache -ArgFile $argFile -Name 'Replayer'
    if ($cacheExitCode -ne 0) {
        exit $cacheExitCode
    }
}

$argFile = New-VibrisJavaArgFile -Backend $Backend -Jar $jar -Capture $captureDir -Frames $Frames -JvmArg @("-XX:AOTCache=$aotCache") -ShaderRoot $ShaderRoot -ShaderPass $ShaderPass

if ($PrintCommand) {
    Write-Output "$java @$argFile"
    exit 0
}

& $java "@$argFile"
exit $LASTEXITCODE
