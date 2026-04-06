# Cache-Aware Computation for Quantitative Finance Workloads

Empirical characterisation of L1 data cache behaviour across four representative quantitative finance kernels -- dense matrix multiplication (GEMM), Cholesky factorisation, correlated Monte Carlo path generation, and GARCH(1,1) maximum likelihood estimation -- measured on an AMD EPYC 7742 (Zen 2) processor using PAPI hardware performance counters on Indiana University's Big Red 200 supercomputer.

## Key Findings

- **Cholesky: layout dominates algorithm.** Storage layout (row-major vs column-major) produces a 28x variation in L1 cache misses, while algorithm choice (Banachiewicz vs Crout) contributes less than 3%.
- **Monte Carlo paths: sharp L1 phase transition.** A 1,657x increase in L1 misses occurs between portfolio dimension d=50 and d=100, coinciding with the triangular factor matrix crossing the 32 KB L1d boundary.
- **GARCH: compute-bound despite cache misses.** A 500x increase in L1 miss rate causes only 3% throughput loss due to loop-carried dependencies in the GARCH recurrence.

## Prerequisites

- Linux with GCC (tested with GCC 7.5.0)
- [PAPI](https://icl.utk.edu/papi/) library (tested with 7.2.0.1)
- Slurm job scheduler (optional, for batch execution)
- Python 3 with `matplotlib` and `pandas` (for plotting)

## Quick Start

```bash
# On a system with PAPI installed:
module load papi          # if using environment modules
cd src
make finance              # builds cholesky, mc_paths, garch binaries

# Run individual benchmarks:
./bin/cholesky_ROW_MAJOR_ALGO_BANACHIEWICZ 1000
./bin/mc_paths_ROW_MAJOR 100 100000
./bin/garch_mle 10000 1000

# Or run full sweep via Slurm:
cd ../scripts
sbatch run_finance_kernels.sh

# Generate plots from collected data:
python3 plot_comparison.py
```

## Directory Structure

```
.
├── README.md
├── LICENSE
├── src/
│   ├── Makefile              # Parametric build (2 layouts x 2 algos etc.)
│   ├── cholesky_papi.c       # Cholesky factorisation with PAPI counters
│   ├── mc_paths_papi.c       # Correlated MC path generation
│   ├── garch_mle_papi.c      # GARCH(1,1) MLE via grid search
│   └── mm_papi.c             # Dense GEMM (validation benchmark)
├── scripts/
│   ├── run_finance_kernels.sh  # Slurm batch script for full sweep
│   └── plot_comparison.py      # Generates publication figures
└── data/
    ├── results.csv             # GEMM measurements (260 configs)
    ├── results_cholesky.csv    # Cholesky measurements (36 configs)
    ├── results_mcpaths.csv     # MC paths measurements (24 configs)
    └── results_garch.csv       # GARCH measurements (21 configs)
```

## Platform

- **CPU**: AMD EPYC 7742, 2.25 GHz (Zen 2)
- **L1d**: 32 KB per core, 8-way associative, 64B lines
- **L2**: 512 KB per core
- **L3**: 256 MB shared
- **PAPI**: 7.2.0.1, using native `perf::L1-DCACHE-LOAD-MISSES` event
- **System**: Indiana University Big Red 200

## Citation

```bibtex
@misc{bathuri2026cache,
  author       = {Pradyot Bathuri},
  title        = {Cache-Aware Computation for Quantitative Finance Workloads on {AMD} {EPYC}},
  year         = {2026},
  institution  = {Indiana University Bloomington},
  howpublished = {\url{https://github.com/pbathuri/finance-cache-hpc}}
}
```

## License

MIT License. See [LICENSE](LICENSE).
