#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# SBATCH script for a single RAG-based baseline (condition, author) job.
#
# Generates for ONE author under ONE condition using the RAG baseline.
# Submit via submit_genre_holdout_parallel_rag.sh, or manually:
#   sbatch --export=ALL,CONDITION=seen,SEED=0,AUTHOR=3,BACKEND=openai,OPENAI_MODEL=gpt-4o \
#          genre_holdout_single_rag.sbatch
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=rag-single
#SBATCH -o /projects/bbrz/yirenl2/demonstrated-feedback/logs/%x_%j.out
#SBATCH -e /projects/bbrz/yirenl2/demonstrated-feedback/logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yirenl2@illinois.edu
#SBATCH --nodes=1
#SBATCH --mem=16g
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --account=bbrz-delta-cpu
#SBATCH --time=01:00:00

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

BACKEND="${BACKEND:-openai}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o}"
TOP_K="${TOP_K:-3}"

echo "=== RAG baseline single-author job: CONDITION=${CONDITION} AUTHOR=${AUTHOR} BACKEND=${BACKEND} ==="
echo "Node: $(hostname)"
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
