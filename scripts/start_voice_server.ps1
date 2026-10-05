# Starts the GPT-SoVITS voice server (api_v2) on http://127.0.0.1:9880.
#
# Which voice it speaks with is GPT_SoVITS/configs/tts_infer.yaml -> custom
# (t2s_weights_path / vits_weights_path); see docs/VOICE_TRAINING.md.
#
# The GPT-SoVITS environment uses PyTorch 2.7.1, whose torchaudio reads audio through
# soundfile. Newer torchaudio needs FFmpeg's shared DLLs (winget: Gyan.FFmpeg.Shared);
# they are put on PATH when present, so either setup works.

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$src = Join-Path $repo 'models\gpt_sovits\src'

$ffmpeg = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Directory -Filter 'Gyan.FFmpeg.Shared*' -ErrorAction SilentlyContinue |
    ForEach-Object { Get-ChildItem $_.FullName -Recurse -Filter 'avcodec-*.dll' } |
    Select-Object -First 1
if ($ffmpeg) { $env:Path = $ffmpeg.DirectoryName + ';' + $env:Path }
$env:PYTHONIOENCODING = 'utf-8'

Set-Location $src
& .\.venv\Scripts\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
