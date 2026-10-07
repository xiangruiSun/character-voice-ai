# Installs a portable Redis for Windows into tools\redis (no admin rights, no service).
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_redis_portable.ps1
#
# Only needed when Memurai (the recommended Redis for Windows) cannot be installed, e.g.
# without administrator rights. start_studio.ps1 starts this copy automatically when
# nothing is listening on :6379. tools\ is git-ignored.

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $repo 'tools\redis'
$version = '5.0.14.1'
$url = "https://github.com/tporadowski/redis/releases/download/v$version/Redis-x64-$version.zip"

if (Test-Path (Join-Path $dest 'redis-server.exe')) {
    Write-Host "Portable Redis already installed: $dest" -ForegroundColor Green
    exit 0
}

New-Item -ItemType Directory -Force $dest | Out-Null
$zip = Join-Path $env:TEMP "redis-$version.zip"
Write-Host "Downloading Redis $version for Windows..."
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
Expand-Archive -Path $zip -DestinationPath $dest -Force
Remove-Item $zip

# Local-only, no persistence to disk: Celery's broker queue is transient, and job state
# lives in the Studio database, so nothing is lost when Redis restarts.
@"
bind 127.0.0.1
port 6379
save ""
appendonly no
"@ | Set-Content -Encoding ascii (Join-Path $dest 'studio.conf')

Write-Host "Installed: $dest\redis-server.exe" -ForegroundColor Green
