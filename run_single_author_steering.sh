#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Steering-vector baseline: generate for a single (condition, author) pair.
#
# Usage:
#   bash run_single_author_steering.sh --condition seen --seed 0 --author 3
#   bash run_single_author_steering.sh --condition unseen-genre --genre b --author 7
#   bash run_single_author_steering.sh --condition seen --seed 0 --author 3 --dry
#   bash run_single_author_steering.sh --condition seen --seed 0 --author 3 --model_id meta-llama/Llama-3.1-8B-Instruct
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── parse arguments ──────────────────────────────────────────────────────
CONDITION=""
SEED=0
GENRE=""
AUTHOR=""
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"
MULTIPLIER=0.15
LAYERS=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --condition)  CONDITION="$2"; shift 2 ;;
        --seed)       SEED="$2"; shift 2 ;;
        --genre)      GENRE="$2"; shift 2 ;;
        --author)     AUTHOR="$2"; shift 2 ;;
        --model_id)   MODEL_ID="$2"; shift 2 ;;
        --multiplier) MULTIPLIER="$2"; shift 2 ;;
        --layers)     LAYERS="$2"; shift 2 ;;
        --dry)        DRY_RUN=true; shift ;;
        *)            echo "Unknown argument: $1"; exit 1 ;;
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

# Model subfolder: derive short tag from model_id
# e.g. "meta-llama/Llama-3.1-8B-Instruct" -> "llama-3.1-8b"
#      "mistralai/Mistral-7B-Instruct-v0.2" -> "mistral-7b"
case "${MODEL_ID}" in
    *Mistral-7B*)     MODEL_TAG="mistral-7b" ;;
    *Llama-3.1-8B*)   MODEL_TAG="llama-3.1-8b" ;;
    *)                MODEL_TAG=$(basename "${MODEL_ID}" | tr '[:upper:]' '[:lower:]') ;;
esac
BASE_OUTPUT="outputs/genre_holdout_steering_exp/${MODEL_TAG}"

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
echo "  STEERING BASELINE  ${CONDITION^^}  tag=${TAG}  -> ${OUT_DIR}"
echo "============================================================"

# ── validate input data ─────────────────────────────────────────────────
if [[ ! -f "${TRAIN_PKL}" ]]; then
    echo "ERROR: ${TRAIN_PKL} not found. Run preprocessing first."
    exit 1
fi

# ── build command ────────────────────────────────────────────────────────
CMD="python generate_steering_baseline.py \
    --train_pkl \"${TRAIN_PKL}\" \
    --test_pkl \"${TEST_PKL}\" \
    --author_key \"${AUTHOR}\" \
    --output_json \"${GEN_JSON}\" \
    --num_samples \"${NUM_GEN_SAMPLES}\" \
    --model_id \"${MODEL_ID}\" \
    --multiplier \"${MULTIPLIER}\""

if [[ -n "${LAYERS}" ]]; then
    CMD="${CMD} --layers ${LAYERS}"
fi

# ── generate ─────────────────────────────────────────────────────────────
run_cmd ${CMD}

echo ""
echo "  DONE: ${TAG}"
