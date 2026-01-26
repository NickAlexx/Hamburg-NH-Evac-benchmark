@echo off
setlocal

rem Run from repo root regardless of current working dir
pushd "%~dp0"

rem Backend (PowerShell + venv activation)
start "backend" powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-Location app\backend; . .\venv\Scripts\Activate.ps1; python -m uvicorn app.main:app"

rem Frontend (npm dev server)
start "frontend" cmd /k "cd /d app\frontend && npm run dev"

popd
endlocal
