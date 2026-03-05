#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# SBATCH script for a single genre-holdout experiment.
#
# Runs SEEN + UNSEEN-GENRE for one held-out genre (all 10 authors).
# Submit via submit_genre_holdout.sh, or manually:
#   sbatch --export=GENRE_TAG=b genre_holdout.sbatch
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ditto-genre
#SBATCH -o /projects/bbrz/yirenl2/demonstrated-feedback/logs/%x_%j.out  # %x = job-name (e.g. ditto-genre-b), %j = jobID
#SBATCH -e /projects/bbrz/yirenl2/demonstrated-feedback/logs/%x_%j.err
#SBATCH --mail-type=ALL                      # Request status by email
#SBATCH --mail-user=yirenl2@illinois.edu     # Email address to send results to.
#SBATCH --nodes=1
#SBATCH --mem=32g
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbrz-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1

set -euo pipefail

# ── validate input ────────────────────────────────────────────────────────
if [[ -z "${GENRE_TAG:-}" ]]; then
    echo "ERROR: GENRE_TAG not set. Use: sbatch --export=GENRE_TAG=b genre_holdout.sbatch"
    exit 1
fi

echo "=== Genre holdout experiment: GENRE_TAG=${GENRE_TAG} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── activate environment ─────────────────────────────────────────────────
cd /projects/bbrz/yirenl2/demonstrated-feedback
source activate_hpc.sh

# ── run experiment ────────────────────────────────────────────────────────
export TRAIN_GPU=0
export GEN_GPU=0

bash run_genre_holdout_exp.sh --genre "${GENRE_TAG}"

echo "End: $(date)"
