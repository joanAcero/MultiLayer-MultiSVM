#!/usr/bin/env bash
# ============================================================
# run_all_experiments.sh
# Sequential runner for the full ML-MSVM experiment suite.
# ============================================================
# Usage:
#   bash run_all_experiments.sh           # run all
#   bash run_all_experiments.sh 2 3 4     # run only experiments 2, 3, 4
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CSV_DIR="$SCRIPT_DIR/results"
PYTHON="${PYTHON:-python3}"

mkdir -p "$LOG_DIR" "$CSV_DIR"

# Which experiments to run (default: all)
TO_RUN=("${@:-1 2 3 4 5}")
if [ "$#" -eq 0 ]; then
    TO_RUN=(1 2 3 4 5)
else
    TO_RUN=("$@")
fi

SUITE_START=$(date +%s)
echo "================================================================"
echo "  ML-MSVM Experiment Suite"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Running : experiments ${TO_RUN[*]}"
echo "  Logs    : $LOG_DIR"
echo "  Results : $CSV_DIR"
echo "================================================================"
echo

for EXP in "${TO_RUN[@]}"; do
    case $EXP in
        1) SCRIPT="exp1_width_analysis.py";  DESC="Width Analysis (m sweep, both kernels)" ;;
        2) SCRIPT="exp2_main_benchmark.py";  DESC="Main Head-to-Head Benchmark" ;;
        3) SCRIPT="exp3_scalability.py";     DESC="Scalability on Large Datasets" ;;
        4) SCRIPT="exp4_learning_curves.py"; DESC="Learning Curves (n-scaling)" ;;
        5) SCRIPT="exp5_depth_p_analysis.py";DESC="Depth × P Interaction" ;;
        *) echo "Unknown experiment: $EXP (valid: 1-5)"; exit 1 ;;
    esac

    echo "────────────────────────────────────────────────────────────────"
    echo "  EXPERIMENT $EXP — $DESC"
    echo "  Script : $SCRIPT"
    echo "  Start  : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────────────────────"

    EXP_START=$(date +%s)
    if $PYTHON "$SCRIPT_DIR/$SCRIPT" \
        --log_dir "$LOG_DIR" \
        --csv_dir "$CSV_DIR"; then
        EXP_END=$(date +%s)
        ELAPSED=$(( EXP_END - EXP_START ))
        H=$(( ELAPSED/3600 )); M=$(( (ELAPSED%3600)/60 )); S=$(( ELAPSED%60 ))
        echo "  ✓ Experiment $EXP done in $(printf '%02dh%02dm%02ds' $H $M $S)"
    else
        echo "  ✗ Experiment $EXP FAILED (exit code $?). Continuing..."
    fi
    echo
done

SUITE_END=$(date +%s)
TOTAL=$(( SUITE_END - SUITE_START ))
H=$(( TOTAL/3600 )); M=$(( (TOTAL%3600)/60 )); S=$(( TOTAL%60 ))

echo "================================================================"
echo "  Suite complete."
echo "  Total time : $(printf '%02dh%02dm%02ds' $H $M $S)"
echo "  Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo
echo "Results CSVs in: $CSV_DIR"
echo "Log files in   : $LOG_DIR"
