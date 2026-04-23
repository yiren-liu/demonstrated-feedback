#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# SBATCH script for a single steering-vector baseline job (GPU).
#
# Trains a steering vector from author demonstrations and generates
# steered test outputs. Requires GPU for model inference and activation
# extraction.
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=steer-single
#SBATCH -o /projects/bbrz/yirenl2/demonstrated-feedback/logs/%x_%j.out
#SBATCH -e /projects/bbrz/yirenl2/demonstrated-feedback/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yirenl2@illinois.edu
#SBATCH --nodes=1
#SBATCH --mem=48g
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

MODEL_ID="${MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.2}"
MULTIPLIER="${MULTIPLIER:-0.15}"
LAYERS="${LAYERS:-}"

echo "=== Steering baseline single-author job (GPU): CONDITION=${CONDITION} AUTHOR=${AUTHOR} MODEL_ID=${MODEL_ID} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── activate environment ─────────────────────────────────────────────────
cd /projects/bbrz/yirenl2/demonstrated-feedback
source activate_hpc.sh

# ── run experiment ───────────────────────────────────────────────────────
ARGS="--condition ${CONDITION} --author ${AUTHOR} --model_id ${MODEL_ID} --multiplier ${MULTIPLIER}"

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
    single-genre)
        if [[ -z "${GENRE_TAG:-}" ]]; then
            echo "ERROR: GENRE_TAG not set for single-genre condition"
            exit 1
        fi
        ARGS="${ARGS} --genre ${GENRE_TAG}"
        ;;
    *)
        echo "ERROR: unknown CONDITION '${CONDITION}' (expected: seen, unseen-genre, single-genre)"
        exit 1
        ;;
esac

if [[ -n "${LAYERS}" ]]; then
    ARGS="${ARGS} --layers ${LAYERS}"
fi

bash run_single_author_steering.sh ${ARGS}

echo "End: $(date)"
