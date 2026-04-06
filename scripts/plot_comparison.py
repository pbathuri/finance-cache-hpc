#!/usr/bin/env python3
"""
plot_comparison.py

Cross-kernel cache sensitivity analysis for finance HPC research.

Reads:
  results.csv             — GEMM (existing)
  results_cholesky.csv    — Cholesky factorisation
  results_mcpaths.csv     — Monte Carlo path generation
  results_garch.csv       — GARCH(1,1) MLE

Produces:
  1. gflops_all_kernels.png        — GFLOPS vs problem size, all kernels
  2. l1miss_all_kernels.png        — L1 misses vs problem size, all kernels
  3. l1miss_per_flop.png           — L1 misses per GFLOP (cache efficiency)
  4. cholesky_layout_algo.png      — Cholesky: layout × algo comparison
  5. mc_paths_phase_transition.png — MC paths: L1 misses vs d (cache spill)
  6. garch_scaling.png             — GARCH: L1 misses vs T (streaming scan)
  7. kernel_comparison_table.txt   — Summary table for paper
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Load all CSV results files."""
    data = {}

    # GEMM (existing)
    gemm_path = os.path.join(SCRIPT_DIR, 'results.csv')
    if os.path.exists(gemm_path):
        df = pd.read_csv(gemm_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        data['gemm'] = df
        print(f"  GEMM:     {len(df)} rows")
    else:
        print(f"  WARNING: {gemm_path} not found")

    # Cholesky
    chol_path = os.path.join(SCRIPT_DIR, 'results_cholesky.csv')
    if os.path.exists(chol_path):
        df = pd.read_csv(chol_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        data['cholesky'] = df
        print(f"  Cholesky: {len(df)} rows")

    # MC Paths
    mc_path = os.path.join(SCRIPT_DIR, 'results_mcpaths.csv')
    if os.path.exists(mc_path):
        df = pd.read_csv(mc_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        data['mc_paths'] = df
        print(f"  MC Paths: {len(df)} rows")

    # GARCH
    garch_path = os.path.join(SCRIPT_DIR, 'results_garch.csv')
    if os.path.exists(garch_path):
        df = pd.read_csv(garch_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        data['garch'] = df
        print(f"  GARCH:    {len(df)} rows")

    return data


# ==================================================================
# Plot 1: GFLOPS vs problem size — all kernels on one figure
# ==================================================================
def plot_gflops_all(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    # GEMM: best order per layout at each N
    if 'gemm' in data:
        df = data['gemm']
        for layout, ls in [('rm', '-'), ('cm', '--')]:
            sub = df[df['layout'] == layout]
            best = sub.groupby('N')['gflops'].max().reset_index()
            ax.plot(best['N'], best['gflops'],
                    marker='o', ms=3, ls=ls, label=f'GEMM ({layout}, best order)')

    # Cholesky: best algo per layout at each d
    if 'cholesky' in data:
        df = data['cholesky']
        for layout, ls in [('rm', '-'), ('cm', '--')]:
            sub = df[df['layout'] == layout]
            best = sub.groupby('d')['gflops'].max().reset_index()
            ax.plot(best['d'], best['gflops'],
                    marker='s', ms=4, ls=ls, label=f'Cholesky ({layout}, best algo)')

    # MC Paths: fixed P=100000, vary d
    if 'mc_paths' in data:
        df = data['mc_paths']
        for layout, ls in [('rm', '-'), ('cm', '--')]:
            sub = df[(df['layout'] == layout) & (df['P'] == df['P'].max())]
            if len(sub) > 0:
                ax.plot(sub['d'], sub['gflops'],
                        marker='^', ms=4, ls=ls, label=f'MC Paths ({layout}, P={int(sub["P"].iloc[0])})')

    # GARCH: fixed n_eval, vary T
    if 'garch' in data:
        df = data['garch']
        nev_max = df['n_eval'].max()
        sub = df[df['n_eval'] == nev_max]
        if len(sub) > 0:
            ax.plot(sub['T'], sub['gflops'],
                    marker='D', ms=4, ls='-', color='red',
                    label=f'GARCH MLE (n_eval={int(nev_max)})')

    ax.set_xlabel('Problem Size (N, d, or T)')
    ax.set_ylabel('GFLOPS')
    ax.set_title('Throughput Across Finance Kernels on Big Red 200')
    ax.set_xscale('log')
    ax.legend(loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(SCRIPT_DIR, 'gflops_all_kernels.png'))
    plt.close(fig)
    print("  -> gflops_all_kernels.png")


# ==================================================================
# Plot 2: L1 misses vs problem size — all kernels (log-log)
# ==================================================================
def plot_l1miss_all(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    def _filter_positive(x, y):
        """Drop zero L1 miss entries that break log scale."""
        mask = y > 0
        return x[mask], y[mask]

    if 'gemm' in data:
        df = data['gemm']
        sub = df[df['layout'] == 'rm']
        best = sub.loc[sub.groupby('N')['gflops'].idxmax()]
        xv, yv = _filter_positive(best['N'].values, best['l1_miss'].values)
        ax.plot(xv, yv, marker='o', ms=3, label='GEMM (rm, best order)')

    if 'cholesky' in data:
        df = data['cholesky']
        sub = df[df['layout'] == 'rm']
        best = sub.loc[sub.groupby('d')['gflops'].idxmax()]
        xv, yv = _filter_positive(best['d'].values, best['l1_miss'].values)
        ax.plot(xv, yv, marker='s', ms=4, label='Cholesky (rm, best algo)')

    if 'mc_paths' in data:
        df = data['mc_paths']
        sub = df[(df['layout'] == 'rm') & (df['P'] == df['P'].max())]
        if len(sub) > 0:
            xv, yv = _filter_positive(sub['d'].values, sub['l1_miss'].values)
            ax.plot(xv, yv, marker='^', ms=4,
                    label=f'MC Paths (rm, P={int(sub["P"].iloc[0])})')

    if 'garch' in data:
        df = data['garch']
        nev_max = df['n_eval'].max()
        sub = df[df['n_eval'] == nev_max]
        if len(sub) > 0:
            xv, yv = _filter_positive(sub['T'].values, sub['l1_miss'].values)
            ax.plot(xv, yv, marker='D', ms=4, color='red',
                    label=f'GARCH MLE (n_eval={int(nev_max)})')

    ax.set_xlabel('Problem Size (N, d, or T)')
    ax.set_ylabel('L1 Data Cache Misses')
    ax.set_title('L1 Cache Misses Across Finance Kernels')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')
    fig.savefig(os.path.join(SCRIPT_DIR, 'l1miss_all_kernels.png'))
    plt.close(fig)
    print("  -> l1miss_all_kernels.png")


# ==================================================================
# Plot 3: L1 misses per GFLOP — cache efficiency metric
# ==================================================================
def plot_l1miss_per_gflop(data):
    fig, ax = plt.subplots(figsize=(10, 6))

    def _plot_nonzero(ax, x, y, **kwargs):
        mask = y > 0
        ax.plot(x[mask], y[mask], **kwargs)

    if 'gemm' in data:
        df = data['gemm']
        sub = df[df['layout'] == 'rm'].copy()
        best = sub.loc[sub.groupby('N')['gflops'].idxmax()].copy()
        flops = 2.0 * best['N'].astype(float)**3
        best['miss_per_gflop'] = best['l1_miss'] / (flops * 1e-9)
        _plot_nonzero(ax, best['N'].values, best['miss_per_gflop'].values,
                      marker='o', ms=3, label='GEMM')

    if 'cholesky' in data:
        df = data['cholesky']
        sub = df[df['layout'] == 'rm'].copy()
        best = sub.loc[sub.groupby('d')['gflops'].idxmax()].copy()
        flops = best['d'].astype(float)**3 / 3.0
        best['miss_per_gflop'] = best['l1_miss'] / (flops * 1e-9)
        _plot_nonzero(ax, best['d'].values, best['miss_per_gflop'].values,
                      marker='s', ms=4, label='Cholesky')

    if 'mc_paths' in data:
        df = data['mc_paths']
        sub = df[(df['layout'] == 'rm') & (df['P'] == df['P'].max())].copy()
        if len(sub) > 0:
            P = sub['P'].iloc[0]
            flops = float(P) * sub['d'].astype(float) * (sub['d'].astype(float) + 1.0)
            sub['miss_per_gflop'] = sub['l1_miss'] / (flops * 1e-9)
            _plot_nonzero(ax, sub['d'].values, sub['miss_per_gflop'].values,
                          marker='^', ms=4, label='MC Paths')

    if 'garch' in data:
        df = data['garch']
        nev_max = df['n_eval'].max()
        sub = df[df['n_eval'] == nev_max].copy()
        if len(sub) > 0:
            flops = 7.0 * sub['T'].astype(float) * float(nev_max)
            sub['miss_per_gflop'] = sub['l1_miss'] / (flops * 1e-9)
            _plot_nonzero(ax, sub['T'].values, sub['miss_per_gflop'].values,
                          marker='D', ms=4, color='red', label='GARCH MLE')

    ax.set_xlabel('Problem Size')
    ax.set_ylabel('L1 Misses per GFLOP')
    ax.set_title('Cache Efficiency: L1 Misses Normalised by Work')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')
    fig.savefig(os.path.join(SCRIPT_DIR, 'l1miss_per_gflop.png'))
    plt.close(fig)
    print("  -> l1miss_per_gflop.png")


# ==================================================================
# Plot 4: Cholesky — layout × algorithm comparison
# ==================================================================
def plot_cholesky_detail(data):
    if 'cholesky' not in data:
        return
    df = data['cholesky']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for (layout, algo), grp in df.groupby(['layout', 'algo']):
        grp = grp.sort_values('d')
        label = f'{layout} / {algo}'
        ax1.plot(grp['d'], grp['gflops'], marker='o', ms=4, label=label)
        pos = grp[grp['l1_miss'] > 0]
        ax2.plot(pos['d'], pos['l1_miss'], marker='s', ms=4, label=label)

    ax1.set_xlabel('Portfolio Dimension d')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title('Cholesky: Throughput vs d')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Portfolio Dimension d')
    ax2.set_ylabel('L1 Data Cache Misses')
    ax2.set_title('Cholesky: L1 Misses vs d')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle('Cholesky Factorisation: Layout × Algorithm Interaction', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'cholesky_layout_algo.png'))
    plt.close(fig)
    print("  -> cholesky_layout_algo.png")


# ==================================================================
# Plot 5: MC Paths — phase transition in L1 misses at d ≈ 128
# ==================================================================
def plot_mc_phase_transition(data):
    if 'mc_paths' not in data:
        return
    df = data['mc_paths']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for layout, grp in df.groupby('layout'):
        for P, sub in grp.groupby('P'):
            sub = sub.sort_values('d')
            label = f'{layout}, P={int(P)}'
            pos = sub[sub['l1_miss'] > 0]
            ax1.plot(pos['d'], pos['l1_miss'], marker='o', ms=4, label=label)
            ax2.plot(sub['d'], sub['gflops'], marker='s', ms=4, label=label)

    # Mark predicted L1 spill point
    l1_size = 32768  # 32 KB L1d on EPYC 7713
    d_spill = int(np.sqrt(2.0 * l1_size / 4.0))  # L has d^2/2 entries
    ax1.axvline(d_spill, color='gray', ls=':', lw=1.5,
                label=f'd≈{d_spill} (L spills L1)')

    ax1.set_xlabel('Portfolio Dimension d')
    ax1.set_ylabel('L1 Data Cache Misses')
    ax1.set_title('MC Path Generation: L1 Misses vs d')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Portfolio Dimension d')
    ax2.set_ylabel('GFLOPS')
    ax2.set_title('MC Path Generation: Throughput vs d')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Monte Carlo Correlated Path Generation: Cache Phase Transition', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'mc_paths_phase_transition.png'))
    plt.close(fig)
    print("  -> mc_paths_phase_transition.png")


# ==================================================================
# Plot 6: GARCH — L1 misses vs T (streaming scan behaviour)
# ==================================================================
def plot_garch_scaling(data):
    if 'garch' not in data:
        return
    df = data['garch']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for nev, grp in df.groupby('n_eval'):
        grp = grp.sort_values('T')
        label = f'n_eval={int(nev)}'
        pos = grp[grp['l1_miss'] > 0]
        ax1.plot(pos['T'], pos['l1_miss'], marker='o', ms=4, label=label)
        ax2.plot(grp['T'], grp['gflops'], marker='s', ms=4, label=label)

    # Overlay predicted: 1 miss per cache line per evaluation per pass
    # cache_line = 64 bytes, double = 8 bytes => 8 doubles per line
    # misses ≈ T / 8 * n_eval (if r[] doesn't fit in cache)
    if len(df) > 0:
        T_range = np.logspace(np.log10(df['T'].min()), np.log10(df['T'].max()), 50)
        nev_max = df['n_eval'].max()
        predicted = T_range / 8.0 * nev_max
        ax1.plot(T_range, predicted, 'k--', lw=1, alpha=0.5,
                 label=f'predicted: T/8 × {int(nev_max)}')

    ax1.set_xlabel('Time Series Length T')
    ax1.set_ylabel('L1 Data Cache Misses')
    ax1.set_title('GARCH MLE: L1 Misses vs T')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Time Series Length T')
    ax2.set_ylabel('GFLOPS')
    ax2.set_title('GARCH MLE: Throughput vs T')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('GARCH(1,1) Maximum Likelihood: Sequential Scan Cache Behaviour', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(SCRIPT_DIR, 'garch_scaling.png'))
    plt.close(fig)
    print("  -> garch_scaling.png")


# ==================================================================
# Table: Summary comparison at representative problem sizes
# ==================================================================
def print_summary_table(data):
    lines = []
    lines.append("=" * 90)
    lines.append("CROSS-KERNEL CACHE SENSITIVITY SUMMARY")
    lines.append("=" * 90)
    lines.append(f"{'Kernel':<20} {'Size':>10} {'GFLOPS':>10} {'L1 Misses':>15} "
                 f"{'Time (s)':>12} {'Miss/GFLOP':>12}")
    lines.append("-" * 90)

    # GEMM at N=1000 (rm, best order)
    if 'gemm' in data:
        df = data['gemm']
        sub = df[(df['layout'] == 'rm') & (df['N'] == 1000)]
        if len(sub) > 0:
            best = sub.loc[sub['gflops'].idxmax()]
            flops_g = 2e9 * 1e-9  # 2*1000^3 / 1e9
            miss_per = best['l1_miss'] / (2e9 * 1e-9)
            lines.append(f"{'GEMM (rm,'+best['order']+')':<20} {'N=1000':>10} "
                         f"{best['gflops']:>10.3f} {best['l1_miss']:>15.0f} "
                         f"{best['seconds']:>12.6f} {miss_per:>12.1f}")

    # Cholesky at d=1000 (rm, best algo)
    if 'cholesky' in data:
        df = data['cholesky']
        sub = df[(df['layout'] == 'rm') & (df['d'] == 1000)]
        if len(sub) > 0:
            best = sub.loc[sub['gflops'].idxmax()]
            flops_g = 1e9 / 3.0 * 1e-9
            miss_per = best['l1_miss'] / flops_g
            lines.append(f"{'Cholesky (rm)':<20} {'d=1000':>10} "
                         f"{best['gflops']:>10.3f} {best['l1_miss']:>15.0f} "
                         f"{best['seconds']:>12.6f} {miss_per:>12.1f}")

    # MC Paths at d=100, max P (rm)
    if 'mc_paths' in data:
        df = data['mc_paths']
        sub = df[(df['layout'] == 'rm') & (df['d'] == 100) & (df['P'] == df['P'].max())]
        if len(sub) > 0:
            row = sub.iloc[0]
            P = row['P']
            flops_g = P * 100 * 101 * 1e-9
            miss_per = row['l1_miss'] / flops_g
            lines.append(f"{'MC Paths (rm)':<20} {f'd=100,P={int(P)}':>10} "
                         f"{row['gflops']:>10.3f} {row['l1_miss']:>15.0f} "
                         f"{row['seconds']:>12.6f} {miss_per:>12.1f}")

    # GARCH at T=10000, max n_eval
    if 'garch' in data:
        df = data['garch']
        nev_max = df['n_eval'].max()
        sub = df[(df['T'] == 10000) & (df['n_eval'] == nev_max)]
        if len(sub) > 0:
            row = sub.iloc[0]
            flops_g = 7.0 * 10000 * nev_max * 1e-9
            miss_per = row['l1_miss'] / flops_g
            lines.append(f"{'GARCH MLE':<20} {f'T=10k,e={int(nev_max)}':>10} "
                         f"{row['gflops']:>10.3f} {row['l1_miss']:>15.0f} "
                         f"{row['seconds']:>12.6f} {miss_per:>12.1f}")

    lines.append("-" * 90)
    lines.append("")
    lines.append("Key observations:")
    lines.append("  - GEMM: high L1 misses, strong layout/order sensitivity")
    lines.append("  - Cholesky: triangular access, ~50% footprint, different sensitivity profile")
    lines.append("  - MC Paths: phase transition at d ~ 128 (L matrix spills L1)")
    lines.append("  - GARCH: streaming scan, minimal L1 misses, compute-bound")
    lines.append("")

    table_text = "\n".join(lines)
    print(table_text)

    outpath = os.path.join(SCRIPT_DIR, 'kernel_comparison_table.txt')
    with open(outpath, 'w') as f:
        f.write(table_text)
    print(f"  -> {outpath}")


# ==================================================================
def main():
    print("Loading data...")
    data = load_data()

    if not data:
        print("ERROR: No CSV files found. Run experiments first.")
        sys.exit(1)

    print("\nGenerating plots...")
    plot_gflops_all(data)
    plot_l1miss_all(data)
    plot_l1miss_per_gflop(data)
    plot_cholesky_detail(data)
    plot_mc_phase_transition(data)
    plot_garch_scaling(data)

    print("\nSummary table:")
    print_summary_table(data)

    print("\nDone.")


if __name__ == '__main__':
    main()
