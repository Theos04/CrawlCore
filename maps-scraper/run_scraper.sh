#!/bin/bash

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "========================================"
echo "Google Maps Business Scraper"
echo "========================================"
echo ""

echo "Starting web server..."
echo "The interface will open in your browser"
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py

if [ $? -ne 0 ]; then
    echo ""
    echo "An error occurred while running the scraper"
    echo "Check the error messages above"
    exit 1
fi
