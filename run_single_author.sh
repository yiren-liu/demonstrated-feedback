#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Train + generate for a single (condition, author) pair.
#
# Usage:
#   bash run_single_author.sh --condition seen --seed 0 --author 3
#   bash run_single_author.sh --condition unseen-genre --genre b --author 7
#   bash run_single_author.sh --condition seen --seed 0 --author 3 --dry
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── parse arguments ──────────────────────────────────────────────────────
CONDITION=""
SEED=0
GENRE=""
AUTHOR=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --condition) CONDITION="$2"; shift 2 ;;
        --seed)      SEED="$2"; shift 2 ;;
        --genre)     GENRE="$2"; shift 2 ;;
        --author)    AUTHOR="$2"; shift 2 ;;
        --dry)       DRY_RUN=true; shift ;;
        *)           echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${CONDITION}" ]]; then
    echo "ERROR: --condition is required (seen or unseen-genre)"; exit 1
fi
if [[ -z "${AUTHOR}" ]]; then
    echo "ERROR: --author is required (0-9)"; exit 1
fi

# ── configuration ────────────────────────────────────────────────────────
CONFIG=configs/ditto-mistral-7b-instruct.yaml
ACCEL_CONFIG=configs/multi_gpu.yaml
NUM_GEN_SAMPLES=3
GEN_BATCH_SIZE=8
TRAIN_GPU="${TRAIN_GPU:-0}"
GEN_GPU="${GEN_GPU:-0}"
BASE_OUTPUT=outputs/genre_holdout_exp
DATA_DIR="${DATA_DIR:-benchmarks/cmcc/processed/genre_holdout}"

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
MODEL_DIR="${OUT_DIR}/ditto"
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
echo "  ${CONDITION^^}  tag=${TAG}  -> ${OUT_DIR}"
echo "============================================================"

# ── validate input data ─────────────────────────────────────────────────
if [[ ! -f "${TRAIN_PKL}" ]]; then
    echo "ERROR: ${TRAIN_PKL} not found. Run preprocessing first."
    exit 1
fi

# ── train ────────────────────────────────────────────────────────────────
run_cmd rm -rf "${OUT_DIR}"

run_cmd CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python scripts/run_ditto.py "${CONFIG}" \
    --train_pkl="${TRAIN_PKL}" \
    --train_author_key="${AUTHOR}" \
    --output_dir="${OUT_DIR}"

# ── generate ─────────────────────────────────────────────────────────────
run_cmd CUDA_VISIBLE_DEVICES="${GEN_GPU}" python generate_scenario.py \
    --model_dir "${MODEL_DIR}" \
    --test_pkl "${TEST_PKL}" \
    --author_key "${AUTHOR}" \
    --output_json "${GEN_JSON}" \
    --num_samples "${NUM_GEN_SAMPLES}" \
    --batch_size "${GEN_BATCH_SIZE}"

echo ""
echo "  DONE: ${TAG}"
