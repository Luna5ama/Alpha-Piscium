param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('status', 'reload', 'capture-pass', 'capture-multi')]
    [string]$Action,
    [string]$Pass = '',
    [ValidateSet('prepare', 'begin', 'deferred', 'composite')]
    [string]$Type = 'composite',
    [string]$Path = '',
    [switch]$NoDefaultPath
)

. "$PSScriptRoot\common.ps1"

function Invoke-IrisCaptureRequest {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ControlPath,
        [Parameter(Mandatory = $true)]
        [string]$Endpoint,
        [object]$Payload = $null
    )

    if (-not (Test-Path -LiteralPath $ControlPath -PathType Leaf)) {
        throw "Iris control file does not exist: $ControlPath"
    }

    $control = Get-Content -Raw -LiteralPath $ControlPath | ConvertFrom-Json
    $uri = "http://$($control.host):$($control.port)/$Endpoint"
    $headers = @{
        Authorization = "Bearer $($control.token)"
    }

    if ($null -eq $Payload) {
        return Invoke-RestMethod -Method Get -Uri $uri -Headers $headers -TimeoutSec 30
    }

    return Invoke-RestMethod `
        -Method Post `
        -Uri $uri `
        -Headers $headers `
        -ContentType 'application/json' `
        -Body ($Payload | ConvertTo-Json -Compress -Depth 16) `
        -TimeoutSec 30
}

$root = Get-VibrisSkillRoot
$config = Get-VibrisConfig -Root $root
$controlPath = Get-VibrisConfigValue -Config $config -Name 'iris_control_path'
if (-not $controlPath) {
    throw 'config.json has no iris_control_path.'
}
$controlPath = Resolve-VibrisPath -Root $root -Path $controlPath

switch ($Action) {
    'status' {
        $response = Invoke-IrisCaptureRequest -ControlPath $controlPath -Endpoint 'status'
    }
    'reload' {
        $response = Invoke-IrisCaptureRequest -ControlPath $controlPath -Endpoint 'reload_shader' -Payload @{}
    }
    'capture-pass' {
        if (-not $Pass) {
            throw '-Pass is required for capture-pass.'
        }
        if (-not $Path -and -not $NoDefaultPath) {
            $Path = New-VibrisCaptureOutputPath -Config $config -Root $root -Name $Pass
        }
        $payload = @{
            pass = $Pass
        }
        if ($Path) {
            $payload.path = $Path
        }
        $response = Invoke-IrisCaptureRequest -ControlPath $controlPath -Endpoint 'capture_pass' -Payload $payload
    }
    'capture-multi' {
        if (-not $Path -and -not $NoDefaultPath) {
            $Path = New-VibrisCaptureOutputPath -Config $config -Root $root -Name $Type
        }
        $payload = @{
            type = $Type
        }
        if ($Path) {
            $payload.path = $Path
        }
        $response = Invoke-IrisCaptureRequest -ControlPath $controlPath -Endpoint 'capture_multi' -Payload $payload
    }
}

$response | ConvertTo-Json -Depth 16
