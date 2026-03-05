#!/bin/bash
# Source this script in sbatch jobs to activate the project environment.
# Usage: source activate_hpc.sh

module load miniforge3-python
source /projects/bbrz/yirenl2/demonstrated-feedback/.venv/bin/activate
export UV_CACHE_DIR=/projects/bbrz/yirenl2/.uv-cache
