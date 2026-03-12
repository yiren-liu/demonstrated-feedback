#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Submit per-author parallel prompt-baseline jobs for the genre-holdout experiment.
#
# Submits one sbatch job per (condition, author) pair — all independent.
# Default: 10 SEEN + 50 UNSEEN-GENRE = 60 jobs.
#
# Usage:
#   bash submit_genre_holdout_parallel_prompt.sh                                        # all 60 jobs (openai)
#   bash submit_genre_holdout_parallel_prompt.sh --dry                                  # print without submitting
#   bash submit_genre_holdout_parallel_prompt.sh --backend mistral                      # use Mistral (GPU jobs)
#   bash submit_genre_holdout_parallel_prompt.sh --no-seen --genre b --authors 0 1      # selective
#   bash submit_genre_holdout_parallel_prompt.sh --openai_model gpt-4o-mini             # specify model
#   bash submit_genre_holdout_parallel_prompt.sh --summarizer_backend openai --summarizer_openai_model gpt-4o
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_GENRES=(b c d e s)
ALL_AUTHORS=(0 1 2 3 4 5 6 7 8 9)

DRY_RUN=false
SKIP_SEEN=false
BACKEND="openai"
OPENAI_MODEL="gpt-5.2"
SUMMARIZER_BACKEND="openai"
SUMMARIZER_OPENAI_MODEL="gpt-5.2"
SELECTED_GENRES=()
SELECTED_AUTHORS=()

# ── parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry)
            DRY_RUN=true; shift ;;
        --no-seen)
            SKIP_SEEN=true; shift ;;
        --backend)
            BACKEND="$2"; shift 2 ;;
        --openai_model)
            OPENAI_MODEL="$2"; shift 2 ;;
        --summarizer_backend)
            SUMMARIZER_BACKEND="$2"; shift 2 ;;
        --summarizer_openai_model)
            SUMMARIZER_OPENAI_MODEL="$2"; shift 2 ;;
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

# Pick the right sbatch script based on backend
if [[ "${BACKEND}" == "mistral" ]]; then
    SBATCH_SCRIPT="${SCRIPT_DIR}/genre_holdout_single_prompt_gpu.sbatch"
else
    SBATCH_SCRIPT="${SCRIPT_DIR}/genre_holdout_single_prompt.sbatch"
fi

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

NUM_JOBS=0

echo "Submitting per-author parallel prompt-baseline jobs (backend=${BACKEND})..."
echo ""

# Build common export vars
EXPORT_BASE="BACKEND=${BACKEND},OPENAI_MODEL=${OPENAI_MODEL}"
if [[ -n "${SUMMARIZER_BACKEND}" ]]; then
    EXPORT_BASE="${EXPORT_BASE},SUMMARIZER_BACKEND=${SUMMARIZER_BACKEND}"
fi
if [[ -n "${SUMMARIZER_OPENAI_MODEL}" ]]; then
    EXPORT_BASE="${EXPORT_BASE},SUMMARIZER_OPENAI_MODEL=${SUMMARIZER_OPENAI_MODEL}"
fi

# ── SEEN condition ───────────────────────────────────────────────────────
if ! $SKIP_SEEN; then
    SEED=0
    for author in "${SELECTED_AUTHORS[@]}"; do
        JOB_NAME="prompt-seen-s${SEED}-a${author}"
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
        JOB_NAME="prompt-genre-${genre}-a${author}"
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
