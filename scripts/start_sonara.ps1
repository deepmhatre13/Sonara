<#
.SYNOPSIS
    Starts the Sonara Windows-native application processes.
.DESCRIPTION
    Launches:
    1. FastAPI Uvicorn Server on http://127.0.0.1:8000
    2. Celery Background Worker using the Windows-compatible solo execution pool (-P solo)
#>
param(
    [switch]$WorkerOnly,
    [switch]$ApiOnly,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$Root = "D:\Sonara"
$BackendDir = "D:\Sonara\backend"
$PythonExe = "$BackendDir\venv\Scripts\python.exe"
$CeleryExe = "$BackendDir\venv\Scripts\celery.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment not found at $PythonExe. Run scripts\setup_windows.ps1 first."
}

# Ensure storage directories
New-Item -ItemType Directory -Path "$Root\data\objects", "$Root\data\models", "$BackendDir\tmp" -Force | Out-Null

$env:PYTHONPATH = $BackendDir

if ($WorkerOnly) {
    Write-Host "Starting Sonara Background Worker on Windows (-P solo)..." -ForegroundColor Cyan
    Set-Location $BackendDir
    & $CeleryExe -A app.core.celery_app.celery_app worker -P solo -l INFO -Q ingest,asr,index,delete,maintenance,eval
    return
}

if ($ApiOnly) {
    Write-Host "Starting Sonara FastAPI Server on http://127.0.0.1:$Port..." -ForegroundColor Cyan
    Set-Location $BackendDir
    & $PythonExe -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload
    return
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "          Starting Sonara Windows-Native Services         " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "FastAPI URL:  http://127.0.0.1:$Port" -ForegroundColor Green
Write-Host "Swagger Docs: http://127.0.0.1:$Port/docs" -ForegroundColor Green
Write-Host "Local Storage: D:\Sonara\data\objects" -ForegroundColor Green
Write-Host "`nLaunching Worker in a separate PowerShell window..." -ForegroundColor Yellow

$WorkerScript = "
`$host.UI.RawUI.WindowTitle = 'Sonara Celery Worker'
Set-Location '$BackendDir'
`$env:PYTHONPATH = '$BackendDir'
& '$CeleryExe' -A app.core.celery_app.celery_app worker -P solo -l INFO -Q ingest,asr,index,delete,maintenance,eval
"

Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $WorkerScript

Write-Host "Launching API Server in this window..." -ForegroundColor Yellow
Set-Location $BackendDir
& $PythonExe -m uvicorn app.main:app --host 127.0.0.1 --port $Port --reload

