#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# SBATCH script for the SEEN condition only.
#
# Runs SEEN (all 10 authors) as a prerequisite for genre-holdout jobs.
# Submit via submit_genre_holdout.sh (auto-submitted as a dependency).
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ditto-seen
#SBATCH -o /projects/bbrz/yirenl2/demonstrated-feedback/logs/%x_%j.out
#SBATCH -e /projects/bbrz/yirenl2/demonstrated-feedback/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yirenl2@illinois.edu
#SBATCH --nodes=1
#SBATCH --mem=32g
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbrz-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1

set -euo pipefail

echo "=== SEEN condition (prerequisite for genre holdout) ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── activate environment ─────────────────────────────────────────────────
cd /projects/bbrz/yirenl2/demonstrated-feedback
source activate_hpc.sh

# ── run experiment ────────────────────────────────────────────────────────
export TRAIN_GPU=0
export GEN_GPU=0

bash run_genre_holdout_exp.sh --seen-only

echo "End: $(date)"
