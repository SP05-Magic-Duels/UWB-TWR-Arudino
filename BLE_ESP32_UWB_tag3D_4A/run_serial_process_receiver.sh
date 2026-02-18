#!/bin/bash

# Define names - change these if your paths are different
VENV_DIR="venv"
PYTHON_SCRIPT="serial_process_receiver.py"

# 1. Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' not found. Creating one..."
    python3 -m venv "$VENV_DIR"

    # Optional: Install requirements if a requirements file exists
    if [ -f "requirements.txt" ]; then
        echo "Installing requirements..."
        source "$VENV_DIR/bin/activate"
        pip install -r requirements.txt
    fi
else
    # 2. Activate the virtual environment
    echo "Activating venv environment..."
    source "$VENV_DIR/bin/activate"
fi

# 3. Run your Python script
echo "Running $PYTHON_SCRIPT..."
python3 "$PYTHON_SCRIPT"

# 4. Deactivate the environment after the script finishes
echo "Deactivating venv environment..."
deactivate
