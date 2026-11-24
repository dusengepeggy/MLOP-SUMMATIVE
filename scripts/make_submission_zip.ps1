# Builds bean-disease-ml-pipeline.zip for submission (attempt 1).
# `git archive` exports only tracked files, so .git, virtualenvs, caches, raw data,
# uploads and model backups (all git-ignored) are never included.
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\.."
$zip = Join-Path $root "bean-disease-ml-pipeline.zip"
if (Test-Path $zip) { Remove-Item -Force $zip }

git -C $root archive --format=zip --prefix=bean-disease-ml-pipeline/ -o $zip HEAD
Write-Host "Created $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)"
