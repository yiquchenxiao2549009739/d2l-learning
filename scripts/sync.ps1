param([Parameter(Mandatory = $true)][string]$Message)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Invoke-GitChecked {
    param([string[]]$GitArguments)
    & git @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($GitArguments[0]). Read the message above." }
}

$branch = (& git branch --show-current).Trim()
if ($LASTEXITCODE -ne 0 -or -not $branch) { throw 'Open this script in a Git branch first.' }
$remoteUrl = & git remote get-url origin
if ($LASTEXITCODE -ne 0) { throw 'No origin remote is configured.' }
Write-Output "Syncing $branch to $remoteUrl"
Invoke-GitChecked -GitArguments @('fetch', 'origin')
$remoteBranch = "refs/remotes/origin/$branch"
& git show-ref --verify --quiet $remoteBranch
if ($LASTEXITCODE -eq 0) {
    & git merge-base --is-ancestor $remoteBranch HEAD
    if ($LASTEXITCODE -ne 0) {
        throw 'The remote contains commits not in this branch. Ask for help before syncing.'
    }
}
Invoke-GitChecked -GitArguments @('add', '--all')
& git diff --cached --quiet
$diffStatus = $LASTEXITCODE
if ($diffStatus -eq 1) {
    Invoke-GitChecked -GitArguments @('commit', '-m', $Message)
} elseif ($diffStatus -ne 0) {
    throw 'Failed to inspect staged changes.'
} else {
    Write-Output 'No new local changes to commit.'
}
Invoke-GitChecked -GitArguments @('push', '--set-upstream', 'origin', $branch)
Write-Output 'GitHub sync completed.'
