Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '..\scripts\common.ps1')

$testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("vibris-aot-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $testRoot | Out-Null

try {
    # Given an AOT cache older than its replay JAR.
    $jar = Join-Path $testRoot 'replay.jar'
    $aotCache = Join-Path $testRoot 'replay.aot'
    $argFile = Join-Path $testRoot 'replay.args'
    $called = Join-Path $testRoot 'java-called.txt'
    $fakeJava = Join-Path $testRoot 'fake-java.cmd'
    [System.IO.File]::WriteAllText($jar, 'jar')
    [System.IO.File]::WriteAllText($aotCache, 'stale')
    [System.IO.File]::WriteAllText($argFile, '')
    [System.IO.File]::SetLastWriteTimeUtc($aotCache, [datetime]::UtcNow.AddMinutes(-2))
    [System.IO.File]::SetLastWriteTimeUtc($jar, [datetime]::UtcNow)
    [System.IO.File]::WriteAllLines($fakeJava, @(
        '@echo off',
        "> `"$called`" echo called",
        'exit /b 0'
    ))

    # When the cache guard runs.
    $exitCode = Ensure-VibrisJavaAotCache -Java $fakeJava -Jar $jar -AotCache $aotCache -ArgFile $argFile -Name 'Test'

    # Then the stale cache is removed and Java regenerates it.
    if ($exitCode -ne 0) {
        throw "Expected Java exit code 0, got $exitCode."
    }
    if (Test-Path -LiteralPath $aotCache) {
        throw 'Expected the stale AOT cache to be removed.'
    }
    if (-not (Test-Path -LiteralPath $called -PathType Leaf)) {
        throw 'Expected Java to run after stale AOT cache invalidation.'
    }
} finally {
    Remove-Item -LiteralPath $testRoot -Recurse -Force
}
