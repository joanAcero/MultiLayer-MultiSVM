#!/usr/bin/env bash
# =======================================================================
# run_all.sh — ML-MSVM Full Experiment Suite Runner
# =======================================================================
# Runs all 10 experiments sequentially, each writing to its own log file
# and CSV. The suite is safe to interrupt: partial results survive because
# each result row is flushed to disk immediately after evaluation.
#
# Usage:
#   bash run_all.sh                    # run all 10 experiments
#   bash run_all.sh 9 1 3 7            # run specific experiments
#   PYTHON=python3.11 bash run_all.sh  # use a specific Python binary
#
# Recommended run order for a 24h+ overnight session (longest first):
#   9  1  3  7  4  5  6  8  10  2
#
# Output directories (created automatically):
#   logs/      one .txt log file per experiment per run
#   results/   one .csv per experiment (appended to if re-run)
#   data/      cached large datasets (SUSY, HIGGS .npz files)
# =======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CSV_DIR="$SCRIPT_DIR/results"
DATA_DIR="$SCRIPT_DIR/data"
PYTHON="${PYTHON:-python3}"

mkdir -p "$LOG_DIR" "$CSV_DIR" "$DATA_DIR"

# Experiment metadata
declare -A SCRIPT DESC
SCRIPT[1]="exp1_width_analysis.py";        DESC[1]="Width Analysis (m sweep, both kernels, all datasets)"
SCRIPT[2]="exp2_main_benchmark.py";        DESC[2]="Main Head-to-Head Benchmark (with published baselines)"
SCRIPT[3]="exp3_scalability.py";           DESC[3]="Scalability on Large Datasets (SUSY, CoverType Full, HIGGS)"
SCRIPT[4]="exp4_learning_curves.py";       DESC[4]="Learning Curves — accuracy and time vs n_train"
SCRIPT[5]="exp5_depth_p_analysis.py";      DESC[5]="Depth × P Interaction Grid"
SCRIPT[6]="exp6_c_spread.py";              DESC[6]="C-Spread Ablation — regularisation diversity"
SCRIPT[7]="exp7_nystroem.py";              DESC[7]="Nystroem vs RFF Comparison"
SCRIPT[8]="exp8_arccosine_degree.py";      DESC[8]="Arc-Cosine Degree Ablation (degree 0, 1, 2)"
SCRIPT[9]="exp9_scalability_timing.py";    DESC[9]="HERO: Timing Curves (acc + time vs n, MNIST + SUSY)"
SCRIPT[10]="exp10_final_c.py";             DESC[10]="Head SVM final_C Sensitivity"

# Experiments that need --data_dir (large dataset downloads)
NEEDS_DATA_DIR="3 4 9"

# Determine which experiments to run
if [ "$#" -eq 0 ]; then
    # Recommended overnight order: longest/most important first
    TO_RUN=(9 1 3 7 4 5 6 8 10 2)
else
    TO_RUN=("$@")
fi

SUITE_START=$(date +%s)
echo "========================================================================"
echo "  ML-MSVM Experiment Suite"
echo "  Started  : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Running  : experiments ${TO_RUN[*]}"
echo "  Python   : $($PYTHON --version 2>&1)"
echo "  Logs     : $LOG_DIR"
echo "  Results  : $CSV_DIR"
echo "  Data     : $DATA_DIR"
echo "========================================================================"

for EXP in "${TO_RUN[@]}"; do
    if [[ -z "${SCRIPT[$EXP]+x}" ]]; then
        echo "  [WARN] Unknown experiment number: $EXP (valid: 1–10). Skipping."
        continue
    fi

    echo ""
    echo "────────────────────────────────────────────────────────────────────"
    echo "  Experiment $EXP — ${DESC[$EXP]}"
    echo "  Script  : ${SCRIPT[$EXP]}"
    echo "  Start   : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────────────────────────"

    # Build argument list
    ARGS=(--log_dir "$LOG_DIR" --csv_dir "$CSV_DIR")
    if echo "$NEEDS_DATA_DIR" | grep -qw "$EXP"; then
        ARGS+=(--data_dir "$DATA_DIR")
    fi

    EXP_START=$(date +%s)
    if "$PYTHON" "$SCRIPT_DIR/${SCRIPT[$EXP]}" "${ARGS[@]}"; then
        EXP_END=$(date +%s)
        EL=$(( EXP_END - EXP_START ))
        printf "  ✓ Experiment %d finished in %02dh%02dm%02ds\n" \
            "$EXP" $((EL/3600)) $(((EL%3600)/60)) $((EL%60))
    else
        echo "  ✗ Experiment $EXP FAILED (exit $?). Continuing with next."
    fi
done

SUITE_END=$(date +%s)
TOTAL=$(( SUITE_END - SUITE_START ))
echo ""
echo "========================================================================"
echo "  Suite complete."
printf "  Total time : %02dh%02dm%02ds\n" \
    $((TOTAL/3600)) $(((TOTAL%3600)/60)) $((TOTAL%60))
echo "  Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""
echo "CSV results in: $CSV_DIR"
ls -lh "$CSV_DIR"/*.csv 2>/dev/null || echo "  (no CSV files found)"
