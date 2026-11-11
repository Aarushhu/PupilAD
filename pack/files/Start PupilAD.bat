@echo off
title PupilAD
cd /d "%~dp0"
if not exist "runtime\python\python.exe" (
    echo The PupilAD runtime is missing. Copy the whole PupilAD-Pack folder, not just this file.
    pause
    exit /b 1
)
"runtime\python\python.exe" -m pupilad app %*
if errorlevel 1 pause
