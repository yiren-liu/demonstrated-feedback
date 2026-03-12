#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Submit per-author parallel steering-vector baseline jobs for the
# genre-holdout experiment.
#
# Submits one sbatch job per (condition, author) pair — all independent.
# Default: 10 SEEN + 50 UNSEEN-GENRE = 60 jobs.
#
# Usage:
#   bash submit_genre_holdout_parallel_steering.sh                     # all 60 jobs
#   bash submit_genre_holdout_parallel_steering.sh --dry               # print without submitting
#   bash submit_genre_holdout_parallel_steering.sh --no-seen --genre b --authors 0 1
#   bash submit_genre_holdout_parallel_steering.sh --model_id meta-llama/Llama-3.1-8B-Instruct
#   bash submit_genre_holdout_parallel_steering.sh --multiplier 0.2 --layers 10 11 12 13 14 15
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_GENRES=(b c d e s)
ALL_AUTHORS=(0 1 2 3 4 5 6 7 8 9)

DRY_RUN=false
SKIP_SEEN=false
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
MULTIPLIER=0.3
LAYERS=""
SELECTED_GENRES=()
SELECTED_AUTHORS=()

# ── parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry)
            DRY_RUN=true; shift ;;
        --no-seen)
            SKIP_SEEN=true; shift ;;
        --model_id)
            MODEL_ID="$2"; shift 2 ;;
        --multiplier)
            MULTIPLIER="$2"; shift 2 ;;
        --layers)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                LAYERS="${LAYERS:+${LAYERS} }$1"; shift
            done
            ;;
        --genre)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                SELECTED_GENRES+=("$1"); shift
            done
            ;;
        --authors)
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                SELECTED_AUTHORS+=("$1"); shift
            done
            ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ${#SELECTED_GENRES[@]} -eq 0 ]]; then
    SELECTED_GENRES=("${ALL_GENRES[@]}")
fi
if [[ ${#SELECTED_AUTHORS[@]} -eq 0 ]]; then
    SELECTED_AUTHORS=("${ALL_AUTHORS[@]}")
fi

SBATCH_SCRIPT="${SCRIPT_DIR}/genre_holdout_single_steering_gpu.sbatch"

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

NUM_JOBS=0

echo "Submitting per-author parallel steering-vector baseline jobs (model=${MODEL_ID}, multiplier=${MULTIPLIER})..."
echo ""

# Build common export vars
EXPORT_BASE="MODEL_ID=${MODEL_ID},MULTIPLIER=${MULTIPLIER},LAYERS=${LAYERS}"

# ── SEEN condition ───────────────────────────────────────────────────────
if ! $SKIP_SEEN; then
    SEED=0
    for author in "${SELECTED_AUTHORS[@]}"; do
        JOB_NAME="steer-seen-s${SEED}-a${author}"
        CMD="sbatch --job-name=${JOB_NAME} --export=ALL,CONDITION=seen,SEED=${SEED},AUTHOR=${author},${EXPORT_BASE} ${SBATCH_SCRIPT}"

        if $DRY_RUN; then
            echo "[DRY] ${CMD}"
        else
            echo "Submitting ${JOB_NAME}..."
            eval "${CMD}"
        fi
        NUM_JOBS=$((NUM_JOBS + 1))
    done
    echo ""
fi

# ── UNSEEN-GENRE conditions ─────────────────────────────────────────────
for genre in "${SELECTED_GENRES[@]}"; do
    for author in "${SELECTED_AUTHORS[@]}"; do
        JOB_NAME="steer-genre-${genre}-a${author}"
        CMD="sbatch --job-name=${JOB_NAME} --export=ALL,CONDITION=unseen-genre,GENRE_TAG=${genre},AUTHOR=${author},${EXPORT_BASE} ${SBATCH_SCRIPT}"

        if $DRY_RUN; then
            echo "[DRY] ${CMD}"
        else
            echo "Submitting ${JOB_NAME}..."
            eval "${CMD}"
        fi
        NUM_JOBS=$((NUM_JOBS + 1))
    done
done

echo ""
echo "Total jobs: ${NUM_JOBS}"
echo "Use 'squeue -u \$USER' to monitor jobs."
