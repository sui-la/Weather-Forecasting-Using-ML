@echo off
title Weather Prediction System - Dependency Installer
color 0B

echo.
echo ========================================
echo    Dependency Installation Script
echo ========================================
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

echo Found requirements.txt ✓
echo.

:: Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo Warning: Failed to upgrade pip, continuing anyway...
    echo.
)

:: Install dependencies
echo.
echo ========================================
echo    Installing Dependencies
echo ========================================
echo.
echo This may take a few minutes...
echo Installing packages from requirements.txt...
echo.

pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ========================================
    echo    Installation Failed!
    echo ========================================
    echo.
    echo Possible solutions:
    echo.
    echo 1. Check your internet connection
    echo 2. Try running with admin privileges
    echo 3. Try installing packages individually:
    echo    pip install flask flask-cors pandas numpy scikit-learn
    echo.
    echo 4. If you're behind a proxy, try:
    echo    pip install -r requirements.txt --proxy http://your-proxy:port
    echo.
    echo 5. Try using a different Python environment
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ========================================
    echo    Installation Successful!
    echo ========================================
    echo.
    echo All dependencies have been installed successfully!
    echo.
    echo You can now run: start_weather_system.bat
    echo.
)

pause 