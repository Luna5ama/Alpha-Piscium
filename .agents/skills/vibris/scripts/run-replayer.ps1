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

$argFile = New-VibrisJavaArgFile -Root $root -Backend $Backend -Jar $jar -Capture $captureDir -Frames $Frames -ShaderRoot $ShaderRoot -ShaderPass $ShaderPass

if ($PrintCommand) {
    Write-Output "$java @$argFile"
    exit 0
}

& $java "@$argFile"
exit $LASTEXITCODE
