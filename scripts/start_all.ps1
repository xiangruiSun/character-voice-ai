# Starts everything for a conversation and opens the page:
#   GPT-SoVITS voice server  http://127.0.0.1:9880   (her voice, on the GPU)
#   conversation API + page  http://127.0.0.1:8000/  (OpenRouter / OpenAI for replies)
#
# Keys and the chat provider (CVAI_LLM) are read from .env. Pass -Mock to use the
# scripted offline brain instead (no key; replies ignore what you say).

param([switch]$Mock)

$repo = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $repo '.env'
$llm = (Select-String -Path $envFile -Pattern '^CVAI_LLM=(\S+)' -ErrorAction SilentlyContinue).Matches.Groups[1].Value
$keyVar = switch ($llm) { 'openai' { 'OPENAI_API_KEY' } 'openrouter' { 'OPENROUTER_API_KEY' } default { 'GEMINI_API_KEY' } }
if (-not $Mock -and -not (Select-String -Path $envFile -Pattern "^$keyVar=\S" -Quiet -ErrorAction SilentlyContinue)) {
    Write-Host "No $keyVar in $envFile - opening it. Paste your key, save, and run this again." -ForegroundColor Yellow
    notepad $envFile
    exit 1
}

Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'start_voice_server.ps1')

$env:PYTHONIOENCODING = 'utf-8'
if ($Mock) { $env:CVAI_LLM = 'mock' } else { Remove-Item Env:CVAI_LLM -ErrorAction SilentlyContinue }

Write-Host 'Waiting for the voice server (first start loads models, ~30 s)...'
for ($i = 0; $i -lt 60; $i++) {
    try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:9880/docs | Out-Null; break } catch { Start-Sleep 2 }
}

Start-Process 'http://127.0.0.1:8000/'
Set-Location $repo
& .\.venv\Scripts\cvai-api.exe --port 8000
