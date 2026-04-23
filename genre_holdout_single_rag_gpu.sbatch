#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# SBATCH script for a single RAG-based baseline job (GPU / Mistral backend).
#
# Same as genre_holdout_single_rag.sbatch but requests a GPU for local
# Mistral-7B-Instruct inference.
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=rag-single
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
TOP_K="${TOP_K:-3}"

echo "=== RAG baseline single-author job (GPU): CONDITION=${CONDITION} AUTHOR=${AUTHOR} BACKEND=${BACKEND} ==="
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

# ── activate environment ─────────────────────────────────────────────────
cd /projects/bbrz/yirenl2/demonstrated-feedback
source activate_hpc.sh

# ── run experiment ───────────────────────────────────────────────────────
ARGS="--condition ${CONDITION} --author ${AUTHOR} --backend ${BACKEND} --openai_model ${OPENAI_MODEL} --top_k ${TOP_K}"

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

bash run_single_author_rag.sh ${ARGS}

echo "End: $(date)"
