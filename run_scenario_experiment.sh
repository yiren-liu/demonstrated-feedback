#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Held-out scenario experiment for DITTO on CMCC.
#
# Runs DITTO training + generation for:
#   - 10 authors  (IDs 0-9)
#   - 2 conditions: "seen" (random split) and "unseen" (topic holdout)
#   - 3 variants per condition:
#       seen:   seeds 0, 1, 2
#       unseen: holdout sets  is (I,S),  cg (C,G),  mp (P,M)
#
# Optional:
#   - "unseen-gt" (genre-topic atomic tasks), if you generated
#     cmcc_unseen_gt_*_{train,test}.pkl with benchmarks/proc_cmcc_scenario.py
#
# Prerequisites:
#   1. Activate the venv:  source .venv/bin/activate
#   2. Run benchmarks/proc_cmcc_scenario.py for all holdout/seed combos
#      (already done if you followed the plan).
#
# Usage:
#   bash run_scenario_experiment.sh          # full experiment
#   bash run_scenario_experiment.sh  --dry   # print commands only
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN — printing commands without executing ==="
fi

# ── configuration ─────────────────────────────────────────────────────────
AUTHORS=(0 1 2 3 4 5 6 7 8 9)
# AUTHORS=(1)   # quick smoke test

# SEEN_SEEDS=(0 1 2)
# SEEN_SEEDS=(0)  # quick smoke test
# UNSEEN_HOLDOUTS=("is" "cg" "mp")
# UNSEEN_HOLDOUTS=("cg" "mp")

CONFIG=configs/ditto-mistral-7b-instruct.yaml
ACCEL_CONFIG=configs/multi_gpu.yaml

NUM_GEN_SAMPLES=3      # generations per test prompt (paper uses 3)
GEN_BATCH_SIZE=8       # prompts per pipeline batch during generation
TRAIN_GPU="${TRAIN_GPU:-1}"   # GPU for training (GPU 0 has a hardware fault)
GEN_GPU="${GEN_GPU:-1}"       # GPU for generation
BASE_OUTPUT=outputs/scenario_experiment

# Allow overriding from environment:
#   DATA_DIR=benchmarks/cmcc/processed/genre_topic_5 bash run_scenario_experiment.sh
DATA_DIR="${DATA_DIR:-benchmarks/cmcc/processed}"

# Optional: genre-topic holdout tags (the middle part of cmcc_unseen_gt_<TAG>_train.pkl)
# Example tag from proc_cmcc_scenario.py: cmcc_unseen_gt_bi-cm-dp-sg_train.pkl -> "bi-cm-dp-sg"
#
# Set from env as a space-separated string:
#   UNSEEN_GT_TAGS="bi-cm-dp-sg" DATA_DIR=benchmarks/cmcc/processed/genre_topic_5 bash run_scenario_experiment.sh
# UNSEEN_GT_TAGS="${UNSEEN_GT_TAGS:-}"
# UNSEEN_GT_TAGS="bi-cm-dp-sg"
# DATA_DIR="benchmarks/cmcc/processed/genre_topic_5"
# UNSEEN_GT_HOLDOUTS=()
# if [[ -n "${UNSEEN_GT_TAGS}" ]]; then
#     read -r -a UNSEEN_GT_HOLDOUTS <<< "${UNSEEN_GT_TAGS}"
# fi

# ── helper: run or echo ───────────────────────────────────────────────────
run_cmd() {
    if $DRY_RUN; then
        echo "[DRY] $*"
    else
        echo ">>> $*"
        eval "$@"
    fi
}

# ── SEEN condition ────────────────────────────────────────────────────────
# for seed in "${SEEN_SEEDS[@]}"; do
#     TRAIN_PKL="${DATA_DIR}/cmcc_seen_s${seed}_train.pkl"
#     TEST_PKL="${DATA_DIR}/cmcc_seen_s${seed}_test.pkl"

#     for author in "${AUTHORS[@]}"; do
#         TAG="seen-s${seed}-a${author}"
#         OUT_DIR="${BASE_OUTPUT}/${TAG}"
#         MODEL_DIR="${OUT_DIR}/ditto"
#         GEN_JSON="${OUT_DIR}/generations.json"

#         echo ""
#         echo "============================================================"
#         echo "  SEEN  seed=${seed}  author=${author}  -> ${OUT_DIR}"
#         echo "============================================================"

#         # ── train ──
#         run_cmd rm -rf "${OUT_DIR}"

#         run_cmd CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python scripts/run_ditto.py "${CONFIG}" \
#             --train_pkl="${TRAIN_PKL}" \
#             --train_author_key="${author}" \
#             --output_dir="${OUT_DIR}"

#         # ── generate (single GPU) ──
#         run_cmd CUDA_VISIBLE_DEVICES="${GEN_GPU}" python generate_scenario.py \
#             --model_dir "${MODEL_DIR}" \
#             --test_pkl "${TEST_PKL}" \
#             --author_key "${author}" \
#             --output_json "${GEN_JSON}" \
#             --num_samples "${NUM_GEN_SAMPLES}" \
#             --batch_size "${GEN_BATCH_SIZE}"
#     done
# done

# ── UNSEEN condition ──────────────────────────────────────────────────────
# for ht in "${UNSEEN_HOLDOUTS[@]}"; do
#     TRAIN_PKL="${DATA_DIR}/cmcc_unseen_${ht}_train.pkl"
#     TEST_PKL="${DATA_DIR}/cmcc_unseen_${ht}_test.pkl"

#     for author in "${AUTHORS[@]}"; do
#         TAG="unseen-${ht}-a${author}"
#         OUT_DIR="${BASE_OUTPUT}/${TAG}"
#         MODEL_DIR="${OUT_DIR}/ditto"
#         GEN_JSON="${OUT_DIR}/generations.json"

#         echo ""
#         echo "============================================================"
#         echo "  UNSEEN  holdout=${ht}  author=${author}  -> ${OUT_DIR}"
#         echo "============================================================"

#         # ── train ──
#         run_cmd rm -rf "${OUT_DIR}"

#         run_cmd CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python scripts/run_ditto.py "${CONFIG}" \
#             --train_pkl="${TRAIN_PKL}" \
#             --train_author_key="${author}" \
#             --output_dir="${OUT_DIR}"

#         # ── generate (single GPU) ──
#         run_cmd CUDA_VISIBLE_DEVICES="${GEN_GPU}" python generate_scenario.py \
#             --model_dir "${MODEL_DIR}" \
#             --test_pkl "${TEST_PKL}" \
#             --author_key "${author}" \
#             --output_json "${GEN_JSON}" \
#             --num_samples "${NUM_GEN_SAMPLES}" \
#             --batch_size "${GEN_BATCH_SIZE}"
#     done
# done

# ── UNSEEN-GENRE (genre holdout) condition ──────────────────────────────────
# Set from env as a space-separated string of genre code tags:
#   UNSEEN_GENRE_TAGS="b c" DATA_DIR=benchmarks/cmcc/processed bash run_scenario_experiment.sh
UNSEEN_GENRE_TAGS="${UNSEEN_GENRE_TAGS:-}"
UNSEEN_GENRE_HOLDOUTS=()
if [[ -n "${UNSEEN_GENRE_TAGS}" ]]; then
    read -r -a UNSEEN_GENRE_HOLDOUTS <<< "${UNSEEN_GENRE_TAGS}"
fi

if ((${#UNSEEN_GENRE_HOLDOUTS[@]})); then
    for g in "${UNSEEN_GENRE_HOLDOUTS[@]}"; do
        TRAIN_PKL="${DATA_DIR}/cmcc_unseen_genre_${g}_train.pkl"
        TEST_PKL="${DATA_DIR}/cmcc_unseen_genre_${g}_test.pkl"

        for author in "${AUTHORS[@]}"; do
            TAG="unseen-genre-${g}-a${author}"
            OUT_DIR="${BASE_OUTPUT}/${TAG}"
            MODEL_DIR="${OUT_DIR}/ditto"
            GEN_JSON="${OUT_DIR}/generations.json"

            echo ""
            echo "============================================================"
            echo "  UNSEEN-GENRE  holdout=${g}  author=${author}  -> ${OUT_DIR}"
            echo "============================================================"

            # ── train ──
            run_cmd rm -rf "${OUT_DIR}"

            run_cmd CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python scripts/run_ditto.py "${CONFIG}" \
                --train_pkl="${TRAIN_PKL}" \
                --train_author_key="${author}" \
                --output_dir="${OUT_DIR}"

            # ── generate (single GPU) ──
            run_cmd CUDA_VISIBLE_DEVICES="${GEN_GPU}" python generate_scenario.py \
                --model_dir "${MODEL_DIR}" \
                --test_pkl "${TEST_PKL}" \
                --author_key "${author}" \
                --output_json "${GEN_JSON}" \
                --num_samples "${NUM_GEN_SAMPLES}" \
                --batch_size "${GEN_BATCH_SIZE}"
        done
    done
fi

# ── UNSEEN (genre-topic atomic task) condition ─────────────────────────────
# if ((${#UNSEEN_GT_HOLDOUTS[@]})); then
#     for gt in "${UNSEEN_GT_HOLDOUTS[@]}"; do
#         TRAIN_PKL="${DATA_DIR}/cmcc_unseen_gt_${gt}_train.pkl"
#         TEST_PKL="${DATA_DIR}/cmcc_unseen_gt_${gt}_test.pkl"

#         for author in "${AUTHORS[@]}"; do
#             TAG="unseen-gt-${gt}-a${author}"
#             OUT_DIR="${BASE_OUTPUT}/${TAG}"
#             MODEL_DIR="${OUT_DIR}/ditto"
#             GEN_JSON="${OUT_DIR}/generations.json"

#             echo ""
#             echo "============================================================"
#             echo "  UNSEEN-GT  holdout=${gt}  author=${author}  -> ${OUT_DIR}"
#             echo "============================================================"

#             # ── train ──
#             run_cmd rm -rf "${OUT_DIR}"

#             run_cmd CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" python scripts/run_ditto.py "${CONFIG}" \
#                 --train_pkl="${TRAIN_PKL}" \
#                 --train_author_key="${author}" \
#                 --output_dir="${OUT_DIR}"

#             # ── generate (single GPU) ──
#             run_cmd CUDA_VISIBLE_DEVICES="${GEN_GPU}" python generate_scenario.py \
#                 --model_dir "${MODEL_DIR}" \
#                 --test_pkl "${TEST_PKL}" \
#                 --author_key "${author}" \
#                 --output_json "${GEN_JSON}" \
#                 --num_samples "${NUM_GEN_SAMPLES}" \
#                 --batch_size "${GEN_BATCH_SIZE}"
#         done
#     done
# fi

echo ""
echo "============================================================"
echo "  ALL DONE — generation JSONs are in ${BASE_OUTPUT}/"
echo "============================================================"