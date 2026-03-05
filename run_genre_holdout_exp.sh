#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Genre-holdout experiment for DITTO on CMCC (5-genre).
#
# Modes:
#   --genre <tag>   Run only one genre holdout (e.g. --genre b). Runs
#                   SEEN + UNSEEN-GENRE for that tag, all authors.
#   --all           Run all 5 genre holdouts (b c d e s) sequentially.
#   (no flag)       Same as --all.
#   --seen-only     Run only the SEEN condition (no genre holdouts).
#   --dry           Print commands without executing (combinable with above).
#
# Environment overrides:
#   DATA_DIR   — processed pickle directory (default: benchmarks/cmcc/processed/genre_holdout)
#   TRAIN_GPU  — CUDA device for training  (default: 0)
#   GEN_GPU    — CUDA device for generation (default: 0)
#
# Usage:
#   bash run_genre_holdout_exp.sh --genre b          # single genre
#   bash run_genre_holdout_exp.sh --genre b --dry    # dry run
#   bash run_genre_holdout_exp.sh --all              # all genres
#   bash run_genre_holdout_exp.sh                    # same as --all
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── parse arguments ──────────────────────────────────────────────────────
DRY_RUN=false
GENRE_TAG=""
RUN_ALL=false
SEEN_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry)       DRY_RUN=true; shift ;;
        --genre)     GENRE_TAG="$2"; shift 2 ;;
        --all)       RUN_ALL=true; shift ;;
        --seen-only) SEEN_ONLY=true; shift ;;
        *)           echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ALL_GENRE_TAGS=(b c d e s)

SKIP_SEEN=false
SKIP_UNSEEN=false
if $SEEN_ONLY; then
    GENRE_TAGS=()
    SKIP_UNSEEN=true
elif [[ -n "${GENRE_TAG}" ]]; then
    GENRE_TAGS=("${GENRE_TAG}")
    SKIP_SEEN=true  # When running a single genre via sbatch, skip SEEN to avoid races
elif $RUN_ALL || true; then
    # Default: run all genres
    GENRE_TAGS=("${ALL_GENRE_TAGS[@]}")
fi

if $DRY_RUN; then
    echo "=== DRY RUN — printing commands without executing ==="
fi

# ── configuration ─────────────────────────────────────────────────────────
AUTHORS=(0 1 2 3 4 5 6 7 8 9)
SEEN_SEEDS=(0)

CONFIG=configs/ditto-mistral-7b-instruct.yaml
ACCEL_CONFIG=configs/multi_gpu.yaml

NUM_GEN_SAMPLES=3      # generations per test prompt
GEN_BATCH_SIZE=8       # prompts per pipeline batch during generation
TRAIN_GPU="${TRAIN_GPU:-0}"
GEN_GPU="${GEN_GPU:-0}"
BASE_OUTPUT=outputs/genre_holdout_exp

DATA_DIR="${DATA_DIR:-benchmarks/cmcc/processed/genre_holdout}"

# ── helper: run or echo ───────────────────────────────────────────────────
run_cmd() {
    if $DRY_RUN; then
        echo "[DRY] $*"
    else
        echo ">>> $*"
        eval "$@"
    fi
}

# ── SEEN condition (only once, not per-genre) ─────────────────────────────
if $SKIP_SEEN; then
    echo "  [skip] SEEN condition (use --all or run without --genre to include it)"
fi
if ! $SKIP_SEEN; then
for seed in "${SEEN_SEEDS[@]}"; do
    TRAIN_PKL="${DATA_DIR}/cmcc_seen_s${seed}_train.pkl"
    TEST_PKL="${DATA_DIR}/cmcc_seen_s${seed}_test.pkl"

    for author in "${AUTHORS[@]}"; do
        TAG="seen-s${seed}-a${author}"
        OUT_DIR="${BASE_OUTPUT}/${TAG}"
        MODEL_DIR="${OUT_DIR}/ditto"
        GEN_JSON="${OUT_DIR}/generations.json"

        # Skip if already completed
        if [[ -f "${GEN_JSON}" ]]; then
            echo "  [skip] ${TAG} — generations.json exists"
            continue
        fi

        echo ""
        echo "============================================================"
        echo "  SEEN  seed=${seed}  author=${author}  -> ${OUT_DIR}"
        echo "============================================================"

        run_cmd rm -rf "${OUT_DIR}"

        run_cmd CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python scripts/run_ditto.py "${CONFIG}" \
            --train_pkl="${TRAIN_PKL}" \
            --train_author_key="${author}" \
            --output_dir="${OUT_DIR}"

        run_cmd CUDA_VISIBLE_DEVICES="${GEN_GPU}" python generate_scenario.py \
            --model_dir "${MODEL_DIR}" \
            --test_pkl "${TEST_PKL}" \
            --author_key "${author}" \
            --output_json "${GEN_JSON}" \
            --num_samples "${NUM_GEN_SAMPLES}" \
            --batch_size "${GEN_BATCH_SIZE}"
    done
done
fi  # end SKIP_SEEN guard

# ── UNSEEN-GENRE (genre holdout) condition ────────────────────────────────
if ! $SKIP_UNSEEN; then
for g in "${GENRE_TAGS[@]}"; do
    TRAIN_PKL="${DATA_DIR}/cmcc_unseen_genre_${g}_train.pkl"
    TEST_PKL="${DATA_DIR}/cmcc_unseen_genre_${g}_test.pkl"

    if [[ ! -f "${TRAIN_PKL}" ]]; then
        echo "ERROR: ${TRAIN_PKL} not found. Run proc_cmcc_scenario.py first."
        exit 1
    fi

    for author in "${AUTHORS[@]}"; do
        TAG="unseen-genre-${g}-a${author}"
        OUT_DIR="${BASE_OUTPUT}/${TAG}"
        MODEL_DIR="${OUT_DIR}/ditto"
        GEN_JSON="${OUT_DIR}/generations.json"

        # Skip if already completed
        if [[ -f "${GEN_JSON}" ]]; then
            echo "  [skip] ${TAG} — generations.json exists"
            continue
        fi

        echo ""
        echo "============================================================"
        echo "  UNSEEN-GENRE  holdout=${g}  author=${author}  -> ${OUT_DIR}"
        echo "============================================================"

        run_cmd rm -rf "${OUT_DIR}"

        run_cmd CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python scripts/run_ditto.py "${CONFIG}" \
            --train_pkl="${TRAIN_PKL}" \
            --train_author_key="${author}" \
            --output_dir="${OUT_DIR}"

        run_cmd CUDA_VISIBLE_DEVICES="${GEN_GPU}" python generate_scenario.py \
            --model_dir "${MODEL_DIR}" \
            --test_pkl "${TEST_PKL}" \
            --author_key "${author}" \
            --output_json "${GEN_JSON}" \
            --num_samples "${NUM_GEN_SAMPLES}" \
            --batch_size "${GEN_BATCH_SIZE}"
    done
done
fi  # end SKIP_UNSEEN guard

echo ""
echo "============================================================"
echo "  ALL DONE — generation JSONs are in ${BASE_OUTPUT}/"
echo "============================================================"