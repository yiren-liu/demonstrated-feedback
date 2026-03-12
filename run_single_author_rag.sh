#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# RAG-based baseline: generate for a single (condition, author) pair.
#
# Usage:
#   bash run_single_author_rag.sh --condition seen --seed 0 --author 3 --backend openai
#   bash run_single_author_rag.sh --condition unseen-genre --genre b --author 7 --backend mistral
#   bash run_single_author_rag.sh --condition seen --seed 0 --author 3 --dry
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── parse arguments ──────────────────────────────────────────────────────
CONDITION=""
SEED=0
GENRE=""
AUTHOR=""
BACKEND="mistral"
OPENAI_MODEL="gpt-4o"
TOP_K=3
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --condition)    CONDITION="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --genre)        GENRE="$2"; shift 2 ;;
        --author)       AUTHOR="$2"; shift 2 ;;
        --backend)      BACKEND="$2"; shift 2 ;;
        --openai_model) OPENAI_MODEL="$2"; shift 2 ;;
        --top_k)        TOP_K="$2"; shift 2 ;;
        --dry)          DRY_RUN=true; shift ;;
        *)              echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${CONDITION}" ]]; then
    echo "ERROR: --condition is required (seen or unseen-genre)"; exit 1
fi
if [[ -z "${AUTHOR}" ]]; then
    echo "ERROR: --author is required (0-9)"; exit 1
fi

# ── configuration ────────────────────────────────────────────────────────
NUM_GEN_SAMPLES=3
DATA_DIR="${DATA_DIR:-benchmarks/cmcc/processed/genre_holdout}"

# Model subfolder: use openai model name or "mistral" for local backend
if [[ "${BACKEND}" == "openai" ]]; then
    MODEL_TAG="${OPENAI_MODEL}"
else
    MODEL_TAG="mistral"
fi
BASE_OUTPUT="outputs/genre_holdout_rag_exp/${MODEL_TAG}"

# ── determine paths based on condition ───────────────────────────────────
case "${CONDITION}" in
    seen)
        TRAIN_PKL="${DATA_DIR}/cmcc_seen_s${SEED}_train.pkl"
        TEST_PKL="${DATA_DIR}/cmcc_seen_s${SEED}_test.pkl"
        TAG="seen-s${SEED}-a${AUTHOR}"
        ;;
    unseen-genre)
        if [[ -z "${GENRE}" ]]; then
            echo "ERROR: --genre is required for unseen-genre condition"; exit 1
        fi
        TRAIN_PKL="${DATA_DIR}/cmcc_unseen_genre_${GENRE}_train.pkl"
        TEST_PKL="${DATA_DIR}/cmcc_unseen_genre_${GENRE}_test.pkl"
        TAG="unseen-genre-${GENRE}-a${AUTHOR}"
        ;;
    *)
        echo "ERROR: unknown condition '${CONDITION}' (expected: seen, unseen-genre)"; exit 1
        ;;
esac

OUT_DIR="${BASE_OUTPUT}/${TAG}"
GEN_JSON="${OUT_DIR}/generations.json"

# ── helper: run or echo ─────────────────────────────────────────────────
run_cmd() {
    if $DRY_RUN; then
        echo "[DRY] $*"
    else
        echo ">>> $*"
        eval "$@"
    fi
}

# ── skip if already completed ────────────────────────────────────────────
if [[ -f "${GEN_JSON}" ]]; then
    echo "  [skip] ${TAG} — generations.json exists"
    exit 0
fi

echo ""
echo "============================================================"
echo "  RAG BASELINE  ${CONDITION^^}  tag=${TAG}  -> ${OUT_DIR}"
echo "============================================================"

# ── validate input data ─────────────────────────────────────────────────
if [[ ! -f "${TRAIN_PKL}" ]]; then
    echo "ERROR: ${TRAIN_PKL} not found. Run preprocessing first."
    exit 1
fi

# ── generate ─────────────────────────────────────────────────────────────
run_cmd python generate_rag_baseline.py \
    --train_pkl "${TRAIN_PKL}" \
    --test_pkl "${TEST_PKL}" \
    --author_key "${AUTHOR}" \
    --output_json "${GEN_JSON}" \
    --num_samples "${NUM_GEN_SAMPLES}" \
    --top_k "${TOP_K}" \
    --backend "${BACKEND}" \
    --openai_model "${OPENAI_MODEL}"

echo ""
echo "  DONE: ${TAG}"
