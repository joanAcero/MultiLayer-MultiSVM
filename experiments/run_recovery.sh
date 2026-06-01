#!/usr/bin/env bash
# =======================================================================
# run_recovery.sh — Run only what's missing from the previous session.
#
# What completed:
#   exp9 MNIST:  DONE — do not re-run
#   exp1 R1 (4): DONE — script skips Wine/BC/Ionosphere/Sonar automatically
#
# What's missing:
#   exp1  Glass + all Regime 2 + MNIST + Fashion (continues from where stopped)
#   exp9  SUSY timing curves (MNIST already done)
#   exp2  Main benchmark (entire run)
#   exp3  Scalability (everything except CoverType Linear SVM)
#   exp4  Learning curves (MNIST done via exp9; SUSY missing)
#   exp5  Depth × P (entire run)
#   exp6  C-spread ablation (entire run)
#   exp7  Nystroem comparison (entire run)
#   exp8  ArcCos degree (entire run)
#   exp10 Head final_C (entire run)
#
# =======================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
CSV_DIR="$SCRIPT_DIR/results"
DATA_DIR="$SCRIPT_DIR/data"
PYTHON="${PYTHON:-python3}"
mkdir -p "$LOG_DIR" "$CSV_DIR" "$DATA_DIR"

run_exp() {
    local num="$1" script="$2" desc="$3"
    shift 3
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  EXP $num — $desc"
    echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "────────────────────────────────────────────────────────────"
    local t0=$(date +%s)
    if "$PYTHON" "$SCRIPT_DIR/$script" \
        --log_dir "$LOG_DIR" --csv_dir "$CSV_DIR" "$@"; then
        local el=$(( $(date +%s) - t0 ))
        printf "  ✓ Done in %02dh%02dm%02ds\n" $((el/3600)) $(((el%3600)/60)) $((el%60))
    else
        echo "  ✗ FAILED (exit $?). Continuing."
    fi
}

echo "========================================================================"
echo "  ML-MSVM Recovery Run"
echo "  Started : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Skipping: exp9 MNIST (done), exp1 Wine/BC/Ionosphere/Sonar (done)"
echo "========================================================================"

# Order: most critical / longest first
run_exp 9  "exp9_scalability_timing.py" "SUSY timing curves"         --data_dir "$DATA_DIR"
run_exp 1  "exp1_width_analysis.py"     "Width analysis (remaining datasets)"
run_exp 3  "exp3_scalability.py"        "Scalability large datasets" --data_dir "$DATA_DIR"
run_exp 2  "exp2_main_benchmark.py"     "Main benchmark"
run_exp 7  "exp7_nystroem.py"           "Nystroem comparison"
run_exp 5  "exp5_depth_p_analysis.py"   "Depth × P grid"
run_exp 6  "exp6_c_spread.py"           "C-spread ablation"
run_exp 8  "exp8_arccosine_degree.py"   "ArcCos degree ablation"
run_exp 10 "exp10_final_c.py"           "Head final_C sensitivity"
run_exp 4  "exp4_learning_curves.py"    "Learning curves (MNIST+SUSY)" --data_dir "$DATA_DIR"

SUITE_END=$(date +%s)
echo ""
echo "========================================================================"
echo "  Recovery run complete."
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  CSVs in : $CSV_DIR"
echo "========================================================================"
ls -lh "$CSV_DIR"/*.csv 2>/dev/null || echo "  (no CSV files)"
