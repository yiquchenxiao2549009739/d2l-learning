param([string]$PythonPath = 'D:\Anaconda\envs\openmmlab\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'The Python executable was not found. Supply the path of an existing environment.'
}
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLCONFIGDIR = Join-Path $projectRoot '.cache\matplotlib'
$env:TEMP = Join-Path $projectRoot '.cache\tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null

# Verify before writing settings. This script installs nothing.
& $PythonPath tools/check_environment.py
if ($LASTEXITCODE -ne 0) { throw 'The existing environment did not pass verification.' }
$config = @{ python = (Resolve-Path -LiteralPath $PythonPath).Path }
$utf8 = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Join-Path $projectRoot 'environment.local.json'), ($config | ConvertTo-Json), $utf8)
$settingsPath = Join-Path $projectRoot '.vscode\settings.json'
$settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
$settings.'python.defaultInterpreterPath' = $config.python.Replace('\', '/')
[System.IO.File]::WriteAllText($settingsPath, ($settings | ConvertTo-Json -Depth 10), $utf8)
Write-Output 'Existing environment selected. No packages were installed.'
