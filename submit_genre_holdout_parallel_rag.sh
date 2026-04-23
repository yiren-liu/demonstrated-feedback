#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Submit per-author parallel RAG-baseline jobs for the genre-holdout experiment.
#
# Submits one sbatch job per (condition, author) pair — all independent.
# Default: 10 SEEN + 50 UNSEEN-GENRE + 50 SINGLE-GENRE = 110 jobs.
#
# Usage:
#   bash submit_genre_holdout_parallel_rag.sh                                  # all 60 jobs (openai)
#   bash submit_genre_holdout_parallel_rag.sh --dry                            # print without submitting
#   bash submit_genre_holdout_parallel_rag.sh --backend mistral                # use Mistral (GPU jobs)
#   bash submit_genre_holdout_parallel_rag.sh --no-seen --genre b --authors 0 1
#   bash submit_genre_holdout_parallel_rag.sh --genre b --authors 0 1 --backend openai --openai_model gpt-5.2
#   bash submit_genre_holdout_parallel_rag.sh --top_k 5                        # more retrieved examples
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ALL_GENRES=(b c d e s)
ALL_AUTHORS=(0 1 2 3 4 5 6 7 8 9)

DRY_RUN=false
SKIP_SEEN=false
SKIP_UNSEEN_GENRE=false
SKIP_SINGLE_GENRE=false
BACKEND="openai"
OPENAI_MODEL="gpt-4o"
TOP_K=3
SELECTED_GENRES=()
SELECTED_AUTHORS=()

# ── parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry)
            DRY_RUN=true; shift ;;
        --no-seen)
            SKIP_SEEN=true; shift ;;
        --no-unseen-genre)
            SKIP_UNSEEN_GENRE=true; shift ;;
        --no-single-genre)
            SKIP_SINGLE_GENRE=true; shift ;;
        --backend)
            BACKEND="$2"; shift 2 ;;
        --openai_model)
            OPENAI_MODEL="$2"; shift 2 ;;
        --top_k)
            TOP_K="$2"; shift 2 ;;
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
    SBATCH_SCRIPT="${SCRIPT_DIR}/genre_holdout_single_rag_gpu.sbatch"
else
    SBATCH_SCRIPT="${SCRIPT_DIR}/genre_holdout_single_rag.sbatch"
fi

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

NUM_JOBS=0

echo "Submitting per-author parallel RAG-baseline jobs (backend=${BACKEND}, top_k=${TOP_K})..."
echo ""

# Build common export vars
EXPORT_BASE="BACKEND=${BACKEND},OPENAI_MODEL=${OPENAI_MODEL},TOP_K=${TOP_K}"

# ── SEEN condition ───────────────────────────────────────────────────────
if ! $SKIP_SEEN; then
    SEED=0
    for author in "${SELECTED_AUTHORS[@]}"; do
        JOB_NAME="rag-seen-s${SEED}-a${author}"
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
if ! $SKIP_UNSEEN_GENRE; then
    for genre in "${SELECTED_GENRES[@]}"; do
        for author in "${SELECTED_AUTHORS[@]}"; do
            JOB_NAME="rag-genre-${genre}-a${author}"
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
fi

# ── SINGLE-GENRE conditions ─────────────────────────────────────────────
if ! $SKIP_SINGLE_GENRE; then
    for genre in "${SELECTED_GENRES[@]}"; do
        for author in "${SELECTED_AUTHORS[@]}"; do
            JOB_NAME="rag-single-${genre}-a${author}"
            CMD="sbatch --job-name=${JOB_NAME} --export=ALL,CONDITION=single-genre,GENRE_TAG=${genre},AUTHOR=${author},${EXPORT_BASE} ${SBATCH_SCRIPT}"

            if $DRY_RUN; then
                echo "[DRY] ${CMD}"
            else
                echo "Submitting ${JOB_NAME}..."
                eval "${CMD}"
            fi
            NUM_JOBS=$((NUM_JOBS + 1))
        done
    done
fi

echo ""
echo "Total jobs: ${NUM_JOBS}"
echo "Use 'squeue -u \$USER' to monitor jobs."
