@echo off
cd /d "%~dp0"
if not exist ".venv-app\Scripts\python.exe" (
    echo Setting up PupilAD for the first time...
    python -m venv .venv-app || goto :error
    ".venv-app\Scripts\python.exe" -m pip install --upgrade pip
)
".venv-app\Scripts\python.exe" -c "import flask, cv2, cryptography" 2>nul || ".venv-app\Scripts\python.exe" -m pip install -r requirements-app.txt || goto :error
set PYTHONPATH=%~dp0src
".venv-app\Scripts\python.exe" -m pupilad app --lan %*
goto :eof

:error
echo.
echo Setup failed. Make sure Python 3.11+ is installed and on PATH.
pause
