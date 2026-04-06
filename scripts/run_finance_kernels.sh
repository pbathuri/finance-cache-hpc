#!/bin/bash
#SBATCH --job-name=finance_cache
#SBATCH --account=c01999
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=finance_cache_%j.out
#SBATCH --error=finance_cache_%j.err

echo "=== Finance Kernel Cache Measurement Suite ==="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Date:      $(date -Iseconds)"
echo "CPU:       $(lscpu | grep 'Model name' | sed 's/.*: *//')"
echo "L1d:       $(lscpu | grep 'L1d' | head -1 | sed 's/.*: *//')"
echo "L2:        $(lscpu | grep 'L2'  | head -1 | sed 's/.*: *//')"
echo "L3:        $(lscpu | grep 'L3'  | head -1 | sed 's/.*: *//')"
echo ""

# ---- modules ----------------------------------------------------
module purge 2>/dev/null || true
module load papi 2>&1 || echo "WARNING: 'module load papi' failed"
echo "Loaded modules:"
module list 2>&1
echo ""
echo "CC:        $(gcc --version | head -1)"
echo "PAPI:      $(papi_avail 2>/dev/null | head -3 | tail -1 || echo 'papi_avail not in PATH')"
echo "pkg-config: $(pkg-config --cflags --libs papi 2>&1 || echo 'no pkg-config for papi')"
echo ""

# ---- working directory -------------------------------------------
cd "$SLURM_SUBMIT_DIR"
echo "WORKDIR:   $(pwd)"
echo ""

# ---- rebuild with latest source ----------------------------------
rm -rf bin
make finance 2>&1
BUILD_RC=$?
echo "=== Build exit code: ${BUILD_RC} ==="
if [ $BUILD_RC -ne 0 ]; then
    echo "FATAL: make finance failed."
    exit 1
fi
echo "Binaries built:"
ls bin/
echo ""

BINDIR=bin

# ---- output files ------------------------------------------------
CHOL_CSV="results_cholesky.csv"
MC_CSV="results_mcpaths.csv"
GARCH_CSV="results_garch.csv"

# ---- quick sanity test -------------------------------------------
echo "=== Sanity test ==="
TEST_OUT=$("${BINDIR}/cholesky_ROW_MAJOR_ALGO_BANACHIEWICZ" 100 2>&1)
echo "Output: $TEST_OUT"
L1_VAL=$(echo "$TEST_OUT" | head -1 | cut -d, -f5)
echo "L1 miss value from test: $L1_VAL"
if [ "$L1_VAL" = "0" ] || [ "$L1_VAL" = "-1" ]; then
    echo "WARNING: L1 miss value is $L1_VAL -- PAPI counters may not be working"
fi
echo ""

# ============================================================
# 1. CHOLESKY DECOMPOSITION
# ============================================================
echo "layout,algo,d,gflops,l1_miss,seconds" > "$CHOL_CSV"
CHOL_DIMS="10 20 50 100 200 500 1000 2000 3000"

echo "=== Cholesky sweep ==="
for layout in ROW_MAJOR COL_MAJOR; do
    for algo in ALGO_BANACHIEWICZ ALGO_CROUT; do
        BIN="${BINDIR}/cholesky_${layout}_${algo}"
        for d in $CHOL_DIMS; do
            echo "  Running: ${layout} ${algo} d=${d}"
            "$BIN" "$d" >> "$CHOL_CSV"
        done
    done
done
echo "  -> ${CHOL_CSV}: $(tail -n +2 "$CHOL_CSV" | wc -l) data points"
echo "  Sample: $(tail -1 "$CHOL_CSV")"
echo ""

# ============================================================
# 2. MONTE CARLO PATH GENERATION
# ============================================================
echo "layout,kernel,d,P,gflops,l1_miss,seconds" > "$MC_CSV"
MC_DIMS="10 50 100 200 500 1000"
MC_PATHS="10000 100000"

echo "=== MC path generation sweep ==="
for layout in ROW_MAJOR COL_MAJOR; do
    BIN="${BINDIR}/mc_paths_${layout}"
    for d in $MC_DIMS; do
        for P in $MC_PATHS; do
            echo "  Running: ${layout} d=${d} P=${P}"
            "$BIN" "$d" "$P" >> "$MC_CSV"
        done
    done
done
echo "  -> ${MC_CSV}: $(tail -n +2 "$MC_CSV" | wc -l) data points"
echo "  Sample: $(tail -1 "$MC_CSV")"
echo ""

# ============================================================
# 3. GARCH(1,1) MLE
# ============================================================
echo "kernel,T,n_eval,gflops,l1_miss,seconds" > "$GARCH_CSV"
GARCH_T="100 500 1000 5000 10000 50000 100000"
GARCH_NEVAL="125 1000 8000"

echo "=== GARCH MLE sweep ==="
BIN="${BINDIR}/garch_mle"
for T in $GARCH_T; do
    for nev in $GARCH_NEVAL; do
        echo "  Running: T=${T} n_eval=${nev}"
        "$BIN" "$T" "$nev" >> "$GARCH_CSV"
    done
done
echo "  -> ${GARCH_CSV}: $(tail -n +2 "$GARCH_CSV" | wc -l) data points"
echo "  Sample: $(tail -1 "$GARCH_CSV")"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=== All sweeps complete ==="
echo "Cholesky:   $(tail -n +2 "$CHOL_CSV" | wc -l) data points"
echo "MC Paths:   $(tail -n +2 "$MC_CSV"   | wc -l) data points"
echo "GARCH MLE:  $(tail -n +2 "$GARCH_CSV"| wc -l) data points"
echo ""

echo "=== Spot-check L1 miss values ==="
echo "Cholesky d=3000:"
grep "3000" "$CHOL_CSV" | head -2
echo "MC Paths d=1000:"
grep ",1000," "$MC_CSV" | head -2
echo "GARCH T=100000:"
grep "100000" "$GARCH_CSV" | head -2
echo ""

echo "Done at $(date -Iseconds)"
