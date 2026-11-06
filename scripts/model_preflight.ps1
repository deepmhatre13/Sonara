<#
.SYNOPSIS
    Preflight verification script for Sonara speech and embedding models on Windows.
.DESCRIPTION
    Validates that:
    1. PyTorch CPU runtime is functional.
    2. Faster-Whisper with PyAV can decode audio and transcribe text.
    3. SentenceTransformers can load the embedding model and produce 384-dimensional vectors.
#>
param(
    [string]$AudioFixture = "D:\Sonara\backend\tests\fixtures\spoken_word_fixture.wav"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Sonara AI Speech & Embedding Runtime Preflight Check   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PythonExe = "D:\Sonara\backend\venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment python not found at $PythonExe. Run scripts\setup_windows.ps1 first."
}

if (-not (Test-Path $AudioFixture)) {
    Write-Error "Audio fixture not found at $AudioFixture."
}

$env:PYTHONPATH = "D:\Sonara\backend"

Write-Host "`n[1/3] Verifying PyTorch and audio decoding..." -ForegroundColor Yellow
& $PythonExe -c "
import torch, soundfile, av
print('  PyTorch version:', torch.__version__)
print('  CUDA available:', torch.cuda.is_available())
print('  PyAV version:', av.__version__)
"
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyTorch or audio decoding libraries failed to load."
}

Write-Host "`n[2/3] Verifying Faster-Whisper speech model inference..." -ForegroundColor Yellow
& $PythonExe -c "
import asyncio
from app.transcription.whisper import FasterWhisperProvider
whisper = FasterWhisperProvider(model_size='tiny')
res = asyncio.run(whisper.transcribe('$($AudioFixture.Replace('\', '/'))'))
print(f'  Language: {res.language}, Segments: {len(res.segments)}')
for s in res.segments:
    print(f'    [{s.start_ms}ms - {s.end_ms}ms]: {s.text}')
"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Faster-Whisper model inference failed."
}

Write-Host "`n[3/3] Verifying SentenceTransformers dense vector generation..." -ForegroundColor Yellow
& $PythonExe -c "
import asyncio
from app.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
provider = SentenceTransformerEmbeddingProvider(model_name='BAAI/bge-small-en-v1.5')
embeddings = asyncio.run(provider.embed_texts(['Test sentence for embeddings']))
print(f'  Model: {provider.model_name}, Dimension: {len(embeddings[0])}')
assert len(embeddings[0]) == 384
"
if ($LASTEXITCODE -ne 0) {
    Write-Error "SentenceTransformers embedding generation failed."
}

Write-Host "`nAll Sonara AI models and inference engines are verified and operational!" -ForegroundColor Green

