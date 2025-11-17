#!/bin/bash
# Launcher script for Justice For Short Kings GUI

echo "=========================================="
echo "Justice For Short Kings - Height Equalizer"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "Checking dependencies..."
python3 -c "import PyQt6" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: PyQt6 not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting GUI..."
echo ""

# Run the GUI
python3 main_gui.py

echo ""
echo "GUI closed. Goodbye!"
