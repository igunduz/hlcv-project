#!/usr/bin/env bash

# Path to the `.py` file you want to run
PYTHON_SCRIPT_PATH="/home/hlcv_team002/team39/src/notebooks/"
# Path to the Python binary of the conda environment
CONDA_PYTHON_BINARY_PATH="/home/hlcv_team002/miniconda3/envs/hlcv-ss23/bin/python"

cd $PYTHON_SCRIPT_PATH
$CONDA_PYTHON_BINARY_PATH "$@"
