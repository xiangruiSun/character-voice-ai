# Starts Character AI Studio on this PC and opens it in the browser.
#
#   powershell -ExecutionPolicy Bypass -File scripts\start_studio.ps1
#
# Starts, each in its own window:  the voice engine (GPT-SoVITS, :9880),
# the backend + website (FastAPI, :8000) and the training worker (Celery).
# Expects Ollama (:11434) running; starts Memurai, or the portable Redis in tools\redis, on :6379.
# Rebuilds the website first when its sources are newer than the last build.

param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'User') + ';' + [Environment]::GetEnvironmentVariable('Path', 'Machine')
$env:PYTHONIOENCODING = 'utf-8'

function Test-Port($port) {
    $client = New-Object Net.Sockets.TcpClient
    try { $client.Connect('127.0.0.1', $port); return $true } catch { return $false } finally { $client.Close() }
}

# -- prerequisites ---------------------------------------------------------------------
if (-not (Test-Port 11434)) {
    Write-Host 'Ollama is not running: start "Ollama" from the Start menu (local models will be unavailable).' -ForegroundColor Yellow
}
if (-not (Test-Port 6379)) {
    $memurai = Get-Service -Name 'Memurai*' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($memurai) { Start-Service $memurai.Name; Start-Sleep 2 }
    # No Memurai (it needs admin rights to install): fall back to the portable copy.
    $portable = Join-Path $repo 'tools\redis\redis-server.exe'
    if (-not (Test-Port 6379) -and (Test-Path $portable)) {
        Start-Process $portable -ArgumentList (Join-Path $repo 'tools\redis\studio.conf') -WindowStyle Minimized
        for ($i = 0; $i -lt 10 -and -not (Test-Port 6379); $i++) { Start-Sleep 1 }
    }
    if (-not (Test-Port 6379)) {
        Write-Host 'Redis is not running on :6379 - training and dataset processing will be unavailable.' -ForegroundColor Yellow
        Write-Host '  Install Memurai (admin):   winget install Memurai.MemuraiDeveloper' -ForegroundColor Yellow
        Write-Host '  or portable (no admin):    powershell -ExecutionPolicy Bypass -File scripts\install_redis_portable.ps1' -ForegroundColor Yellow
    }
}

# -- website build (only when sources changed) --------------------------------------------
$studio = Join-Path $repo 'apps\studio'
$out = Join-Path $studio 'out\index.html'
$newest = Get-ChildItem $studio -Recurse -File -Include *.ts, *.tsx, *.css, *.json -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\(node_modules|\.next|out)\\' } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not (Test-Path $out) -or ($newest -and $newest.LastWriteTime -gt (Get-Item $out).LastWriteTime)) {
    Write-Host 'Building the website...'
    Push-Location $studio
    if (-not (Test-Path 'node_modules')) { npm install }
    npm run build
    Pop-Location
}

# -- services ------------------------------------------------------------------------------
if (-not (Test-Port 9880)) {
    Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'start_voice_server.ps1')
}
if (-not (Test-Port 8000)) {
    Start-Process powershell -WorkingDirectory $repo -ArgumentList '-NoExit', '-Command', "& '$python' -m cvai_api.app --port 8000"
}
Start-Process powershell -WorkingDirectory $repo -ArgumentList '-NoExit', '-Command', "& '$python' -m cvai_studio.workers.celery_app"

Write-Host 'Waiting for the backend...'
for ($i = 0; $i -lt 60 -and -not (Test-Port 8000); $i++) { Start-Sleep 1 }
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8000/' }
Write-Host 'Character AI Studio: http://127.0.0.1:8000/' -ForegroundColor Green
