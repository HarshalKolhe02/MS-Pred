#!/usr/bin/env bash
# Script to set up a dedicated virtual environment for MS Predictor

set -e

ENV_NAME="ms_env"
PYTHON_BIN=$(which python3)

echo "=========================================================="
echo " Setting up Classical EI-MS Predictor Environment"
echo " Python binary: $PYTHON_BIN"
echo " Target virtual environment: $ENV_NAME"
echo "=========================================================="

if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment '$ENV_NAME'..."
    $PYTHON_BIN -m venv "$ENV_NAME"
else
    echo "Virtual environment '$ENV_NAME' already exists."
fi

echo "Activating virtual environment..."
source "$ENV_NAME/bin/activate"

echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "=========================================================="
echo " Setup complete!"
echo " Activate with:  source $ENV_NAME/bin/activate"
echo " Run tests with: pytest tests/ -v"
echo " Run CLI with:   python predict_eims.py --smiles 'CCCCC=O' --explain"
echo "=========================================================="
