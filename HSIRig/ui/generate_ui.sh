#!/bin/sh

# Source - https://stackoverflow.com/questions/43028904/converting-ui-to-py-with-python-3-6-on-pyqt5
# Posted by Danilo Gasques
# Retrieved 2025-11-06, License - CC BY-SA 4.0
#
# Modified by Robert Stevenson
# Created 6 Nov 2025
# old command:
# python3 -m PyQt5.uic.pyuic -xd ./main_window.ui -o main_window.py
# Changes (Rob);
#   script check if windows or linux and adjust the creation command based on this. 
#   if on windows use `GIT Bash` terminal to run this or WSL

# Define filenames
UI_FILE="./main_window.ui"
OUTPUT_FILE="main_window.py"

# Detect Operating System
# Windows (Git Bash/MINGW) identifies as 'MSYS' or 'MINGW'
OS_TYPE="$(uname -s)"

if [[ "$OS_TYPE" == *"MINGW"* || "$OS_TYPE" == *"MSYS"* || "$OS_TYPE" == *"CYGWIN"* ]]; then
    echo "Windows detected..."
    PYTHON_EXEC="python"
else
    echo "Linux/Unix detected..."
    PYTHON_EXEC="python3"
fi

echo "Using command: $PYTHON_EXEC"

# Run the compilation
$PYTHON_EXEC -m PyQt5.uic.pyuic -x "$UI_FILE" -o "$OUTPUT_FILE"

# Result verification
if [ $? -eq 0 ]; then
    echo "Success! $OUTPUT_FILE generated."
else
    echo "Error: Compilation failed."
    # Troubleshooting tip for Windows users
    if [[ "$PYTHON_EXEC" == "python" ]]; then
        echo "Tip: Ensure 'python' is in your PATH or your venv is active."
    fi
    exit 1
fi