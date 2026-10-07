@echo off
title Google Maps Business Scraper
echo ========================================
echo Google Maps Business Scraper
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Starting web server...
echo The interface will open in your browser
echo Press Ctrl+C to stop the server
echo.

python app.py
if errorlevel 1 (
    echo.
    echo An error occurred while running the scraper
    echo Check the error messages above
    pause
)
