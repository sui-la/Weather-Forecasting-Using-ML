@echo off
title Weather Prediction System - Full Stack Launcher
color 0A

echo.
echo ========================================
echo    Weather Prediction System Launcher
echo ========================================
echo.
echo Starting both Frontend and Backend servers...
echo.

:: Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    echo.
    echo Download Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found! ✓
echo.

:: Check if requirements.txt exists
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found!
    echo Please make sure you're running this from the project root directory
    pause
    exit /b 1
)

:: Check if Flask is installed
echo Checking dependencies...
pip show flask >nul 2>&1
if errorlevel 1 (
    echo.
    echo ========================================
    echo    Dependencies Not Found
    echo ========================================
    echo.
    echo Some required packages are not installed.
    echo.
    set /p install_choice="Do you want to install dependencies now? (y/n): "
    if /i "%install_choice%"=="y" (
        echo.
        echo Installing dependencies...
        echo This may take a few minutes...
        echo.
        pip install -r requirements.txt
        if errorlevel 1 (
            echo.
            echo ERROR: Failed to install dependencies!
            echo.
            echo Possible solutions:
            echo 1. Check your internet connection
            echo 2. Try running: pip install --upgrade pip
            echo 3. Try running: pip install -r requirements.txt manually
            echo.
            pause
            exit /b 1
        )
        echo.
        echo Dependencies installed successfully! ✓
    ) else (
        echo.
        echo Please install dependencies manually by running:
        echo pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
) else (
    echo Dependencies OK! ✓
)

echo.
echo ========================================
echo    Starting Servers
echo ========================================
echo.

:: Check if run.py exists
if not exist "run.py" (
    echo ERROR: run.py not found!
    echo Please make sure you're running this from the project root directory
    pause
    exit /b 1
)

:: Check if start_frontend.py exists
if not exist "start_frontend.py" (
    echo ERROR: start_frontend.py not found!
    echo Please make sure you're running this from the project root directory
    pause
    exit /b 1
)

:: Start backend server in a new window
echo Starting Backend Server (Flask API)...
start "Weather Prediction Backend" cmd /k "python run.py"

:: Wait a moment for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

:: Start frontend server in a new window
echo Starting Frontend Server (Web Interface)...
start "Weather Prediction Frontend" cmd /k "python start_frontend.py"

:: Wait a moment for frontend to start
echo Waiting for frontend to initialize...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo    Servers Started Successfully!
echo ========================================
echo.
echo Backend API:  http://localhost:5000
echo Frontend Web: http://localhost:8000
echo.
echo The servers are now running in separate windows.
echo.
set /p open_browser="Do you want to open the web interface now? (y/n): "
if /i "%open_browser%"=="y" (
    echo Opening web interface...
    start http://localhost:8000
    echo.
    echo Web interface opened in your browser!
) else (
    echo.
    echo You can manually open: http://localhost:8000
)

echo.
echo ========================================
echo    How to Stop the Servers
echo ========================================
echo.
echo To stop the servers:
echo 1. Close the command windows that opened
echo 2. Or press Ctrl+C in each window
echo.
echo Enjoy your Weather Prediction System!
echo.
pause 