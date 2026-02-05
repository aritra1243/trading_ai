@echo off
echo ===================================================
echo Starting Trading AI Dashboard...
echo ===================================================

:: Activates the virtual environment and starts the backend
start "Trading AI Backend" cmd /k "cd /d %~dp0 && venv\Scripts\python.exe api/server.py"

:: Starts the Frontend
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

cd /d %~dp0web
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)

start "Trading AI Frontend" cmd /k "cd /d %~dp0web && npm run dev"

echo ===================================================
echo Dashboard launched!
echo Backend API: http://localhost:8000
echo Frontend UI: http://localhost:5173
echo ===================================================
pause
