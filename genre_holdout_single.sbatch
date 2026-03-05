#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# SBATCH script for a single (condition, author) job.
#
# Trains + generates for ONE author under ONE condition.
# Submit via submit_genre_holdout_parallel.sh, or manually:
#   sbatch --export=ALL,CONDITION=seen,SEED=0,AUTHOR=3 genre_holdout_single.sbatch
#   sbatch --export=ALL,CONDITION=unseen-genre,GENRE_TAG=b,AUTHOR=7 genre_holdout_single.sbatch
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=ditto-single
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
#SBATCH --time=02:00:00
#SBATCH --gpus-per-node=1

set -euo pipefail

# ── validate input ───────────────────────────────────────────────────────
if [[ -z "${CONDITION:-}" ]]; then
    echo "ERROR: CONDITION not set (expected: seen or unseen-genre)"
    exit 1
fi
if [[ -z "${AUTHOR:-}" ]]; then
    echo "ERROR: AUTHOR not set (expected: 0-9)"
    exit 1
fi

echo "=== Single-author job: CONDITION=${CONDITION} AUTHOR=${AUTHOR} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── activate environment ─────────────────────────────────────────────────
cd /projects/bbrz/yirenl2/demonstrated-feedback
source activate_hpc.sh

# ── run experiment ───────────────────────────────────────────────────────
export TRAIN_GPU=0
export GEN_GPU=0

ARGS="--condition ${CONDITION} --author ${AUTHOR}"

case "${CONDITION}" in
    seen)
        ARGS="${ARGS} --seed ${SEED:-0}"
        ;;
    unseen-genre)
        if [[ -z "${GENRE_TAG:-}" ]]; then
            echo "ERROR: GENRE_TAG not set for unseen-genre condition"
            exit 1
        fi
        ARGS="${ARGS} --genre ${GENRE_TAG}"
        ;;
    *)
        echo "ERROR: unknown CONDITION '${CONDITION}'"
        exit 1
        ;;
esac

bash run_single_author.sh ${ARGS}

echo "End: $(date)"
