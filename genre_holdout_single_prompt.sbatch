#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────
# SBATCH script for a single prompt-based baseline (condition, author) job.
#
# Generates for ONE author under ONE condition using the prompt baseline.
# Submit via submit_genre_holdout_parallel_prompt.sh, or manually:
#   sbatch --export=ALL,CONDITION=seen,SEED=0,AUTHOR=3,BACKEND=openai,OPENAI_MODEL=gpt-4o \
#          genre_holdout_single_prompt.sbatch
# ──────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=prompt-single
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
SUMMARIZER_BACKEND="${SUMMARIZER_BACKEND:-}"
SUMMARIZER_OPENAI_MODEL="${SUMMARIZER_OPENAI_MODEL:-}"

echo "=== Prompt baseline single-author job: CONDITION=${CONDITION} AUTHOR=${AUTHOR} BACKEND=${BACKEND} ==="
echo "Node: $(hostname)"
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
