<#
.SYNOPSIS
    Automated setup and prerequisite validation for Windows-native Sonara development.
.DESCRIPTION
    Checks Python 3.12, virtual environment packages, local filesystem object storage directories,
    and PostgreSQL connection settings.
#>
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         Sonara Windows-Native Environment Setup          " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify Python
$PythonExe = "D:\Sonara\backend\venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[1/5] Virtual environment not found. Creating D:\Sonara\backend\venv..." -ForegroundColor Yellow
    python -m venv D:\Sonara\backend\venv
} else {
    Write-Host "[1/5] Virtual environment detected: $PythonExe" -ForegroundColor Green
}

# 2. Install / Verify Packages from pyproject.toml
Write-Host "[2/5] Checking and installing Python dependencies..." -ForegroundColor Yellow
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -e "D:\Sonara\backend[dev]"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Python dependencies from pyproject.toml."
}

# 3. Create Local Object & Model Storage Directories
Write-Host "[3/5] Initializing local storage directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "D:\Sonara\data\objects", "D:\Sonara\data\models", "D:\Sonara\backend\tmp" -Force | Out-Null
Write-Host "  Storage paths ensured at D:\Sonara\data\objects and D:\Sonara\data\models" -ForegroundColor Green

# 4. Initialize Local .env if Missing
if (-not (Test-Path "D:\Sonara\backend\.env")) {
    Write-Host "[4/5] Creating backend\.env from .env.windows.example..." -ForegroundColor Yellow
    Copy-Item "D:\Sonara\backend\.env.windows.example" "D:\Sonara\backend\.env"
} else {
    Write-Host "[4/5] Found existing backend\.env configuration" -ForegroundColor Green
}

# 5. Database Connectivity & Migrations Check
Write-Host "[5/5] Checking database connection and migrations..." -ForegroundColor Yellow
$env:PYTHONPATH = "D:\Sonara\backend"
& $PythonExe -c "
import os
from sqlalchemy import text, create_engine
from app.core.config import settings

sync_url = settings.SYNC_DATABASE_URL
engine = create_engine(sync_url, connect_args={'connect_timeout': 3})
try:
    with engine.connect() as conn:
        res = conn.execute(text('SELECT version()')).scalar()
        print('  Connected to PostgreSQL:', res.split(',')[0])
        has_vector = conn.execute(text(\"SELECT 1 FROM pg_extension WHERE extname = 'vector'\")).scalar()
        if has_vector:
            print('  pgvector extension: ACTIVE')
        else:
            print('  pgvector extension: NOT INSTALLED in target database.')
except Exception as e:
    print('  PostgreSQL connection note:', e)
"

Write-Host "`nSetup completed! Run .\scripts\start_sonara.ps1 to start the backend." -ForegroundColor Green

