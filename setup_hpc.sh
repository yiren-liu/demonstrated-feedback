#!/bin/bash
# Sets up a Python environment for the demonstrated-feedback project on Delta (A100).
# CUDA 12.8, driver 570.x. Torch is installed from PyTorch cu128 wheels.

set -e

module load miniforge3-python

export UV_CACHE_DIR=/projects/bbrz/yirenl2/.uv-cache
# Create venv using HPC Python
uv venv --python "$(which python3)" .venv

# Install pinned deps (--no-deps avoids resolver conflicts from transitive upper bounds)
uv pip install --python .venv/bin/python --no-deps -r requirements-hpc.txt

# Install alignment-handbook at the pinned commit
cd alignment-handbook
git checkout 606d2e954fd17999af40e6fb4f712055ca11b2f0
uv pip install --python ../.venv/bin/python .
cd ..

echo "Done. Activate with: source .venv/bin/activate"
echo "Then load the module before running: module load miniforge3-python"
