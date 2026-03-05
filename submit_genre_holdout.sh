#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Submit genre-holdout experiment jobs.
#
# Submits a SEEN job first, then one job per genre holdout (dependent on SEEN).
# Total: 1 SEEN + N genre jobs (6 jobs by default).
#
# Usage:
#   bash submit_genre_holdout.sh           # submit all 5 genres (+SEEN)
#   bash submit_genre_holdout.sh b c       # submit only genres b and c (+SEEN)
#   bash submit_genre_holdout.sh --dry     # print sbatch commands without submitting
#   bash submit_genre_holdout.sh --no-seen # skip SEEN job (if already done)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/genre_holdout.sbatch"
SEEN_SBATCH_SCRIPT="${SCRIPT_DIR}/genre_holdout_seen.sbatch"

ALL_GENRES=(b c d e s)
GENRE_NAMES=(Blogs Chat Discussion Emails Essays)

DRY_RUN=false
SKIP_SEEN=false
SELECTED_GENRES=()

for arg in "$@"; do
    if [[ "$arg" == "--dry" ]]; then
        DRY_RUN=true
    elif [[ "$arg" == "--no-seen" ]]; then
        SKIP_SEEN=true
    else
        SELECTED_GENRES+=("$arg")
    fi
done

if [[ ${#SELECTED_GENRES[@]} -eq 0 ]]; then
    SELECTED_GENRES=("${ALL_GENRES[@]}")
fi

# Create logs directory
mkdir -p "${SCRIPT_DIR}/logs"

echo "Submitting genre holdout jobs..."
echo ""

# ── Submit SEEN job first ────────────────────────────────────────────────
SEEN_JOB_ID=""
DEPENDENCY_FLAG=""

if ! $SKIP_SEEN; then
    SEEN_CMD="sbatch --parsable ${SEEN_SBATCH_SCRIPT}"
    if $DRY_RUN; then
        echo "[DRY] ${SEEN_CMD}"
        DEPENDENCY_FLAG="--dependency=afterok:SEEN_JOB_ID"
    else
        echo "Submitting SEEN condition..."
        SEEN_JOB_ID=$(eval "${SEEN_CMD}")
        echo "  SEEN job ID: ${SEEN_JOB_ID}"
        DEPENDENCY_FLAG="--dependency=afterok:${SEEN_JOB_ID}"
    fi
    echo ""
fi

# ── Submit genre-holdout jobs (dependent on SEEN) ────────────────────────
for i in "${!ALL_GENRES[@]}"; do
    g="${ALL_GENRES[$i]}"
    name="${GENRE_NAMES[$i]}"

    # Skip if not in selected list
    found=false
    for sel in "${SELECTED_GENRES[@]}"; do
        if [[ "$sel" == "$g" ]]; then found=true; break; fi
    done
    $found || continue

    CMD="sbatch --job-name=ditto-genre-${g} ${DEPENDENCY_FLAG} --export=ALL,GENRE_TAG=${g} ${SBATCH_SCRIPT}"

    if $DRY_RUN; then
        echo "[DRY] ${CMD}"
    else
        echo "Submitting genre=${g} (${name})..."
        eval "${CMD}"
    fi
done

echo ""
echo "Done. Use 'squeue -u \$USER' to monitor jobs."
