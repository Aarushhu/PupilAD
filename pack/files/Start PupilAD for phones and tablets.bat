@echo off
title PupilAD (phones and tablets)
cd /d "%~dp0"
if not exist "runtime\python\python.exe" (
    echo The PupilAD runtime is missing. Copy the whole PupilAD-Pack folder, not just this file.
    pause
    exit /b 1
)
echo If Windows asks about network access, choose "Private networks" and Allow.
echo.
"runtime\python\python.exe" -m pupilad app --lan %*
if errorlevel 1 pause
