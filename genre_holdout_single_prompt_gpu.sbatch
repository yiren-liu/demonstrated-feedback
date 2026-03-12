#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# SBATCH script for a single prompt-based baseline job (GPU / Mistral backend).
#
# Same as genre_holdout_single_prompt.sbatch but requests a GPU for local
# Mistral-7B-Instruct inference.
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=prompt-single
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
#SBATCH --time=01:00:00
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

BACKEND="${BACKEND:-mistral}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
SUMMARIZER_BACKEND="${SUMMARIZER_BACKEND:-}"
SUMMARIZER_OPENAI_MODEL="${SUMMARIZER_OPENAI_MODEL:-}"

echo "=== Prompt baseline single-author job (GPU): CONDITION=${CONDITION} AUTHOR=${AUTHOR} BACKEND=${BACKEND} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── activate environment ─────────────────────────────────────────────────
cd /projects/bbrz/yirenl2/demonstrated-feedback
source activate_hpc.sh

# ── run experiment ───────────────────────────────────────────────────────
ARGS="--condition ${CONDITION} --author ${AUTHOR} --backend ${BACKEND} --openai_model ${OPENAI_MODEL}"

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

if [[ -n "${SUMMARIZER_BACKEND}" ]]; then
    ARGS="${ARGS} --summarizer_backend ${SUMMARIZER_BACKEND}"
fi
if [[ -n "${SUMMARIZER_OPENAI_MODEL}" ]]; then
    ARGS="${ARGS} --summarizer_openai_model ${SUMMARIZER_OPENAI_MODEL}"
fi

bash run_single_author_prompt.sh ${ARGS}

echo "End: $(date)"
