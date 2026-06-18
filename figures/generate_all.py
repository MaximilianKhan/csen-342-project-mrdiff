"""
Generate all figures for the CSEN-342 Final Report.
Outputs PNGs to the current directory.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'paper': '#2c3e50',
    'baseline': '#e74c3c',
    'ci_decomp': '#2ecc71',
    'itrans': '#3498db',
    'twoscale': '#9b59b6',
    'sweep': '#f39c12',
    'ensemble': '#1abc9c',
    'diffusion_only': '#95a5a6',
    'direct_only': '#e67e22',
}

benchmarks = ['ETTh1\nMulti', 'ETTh1\nUni', 'ETTm1\nMulti', 'ETTm1\nUni']
benchmarks_short = ['ETTh1 M', 'ETTh1 U', 'ETTm1 M', 'ETTm1 U']

OUT = '.'

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Baseline vs Paper MAE comparison (grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_baseline_vs_paper():
    paper_mae = [0.42, 0.34, 0.37, 0.15]
    our_mae   = [0.4744, 0.2535, 0.4204, 0.2011]

    x = np.arange(len(benchmarks))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, paper_mae, w, label='Paper (mr-Diff)', color=COLORS['paper'], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + w/2, our_mae,   w, label='Our Baseline',     color=COLORS['baseline'], edgecolor='white', linewidth=0.5)

    # annotate
    for bar, val in zip(bars1, paper_mae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, f'{val:.2f}', ha='center', va='bottom', fontsize=9, color=COLORS['paper'])
    for bar, val, pv in zip(bars2, our_mae, paper_mae):
        delta = (val - pv) / pv * 100
        sign = '+' if delta > 0 else ''
        color = '#27ae60' if delta < 0 else '#c0392b'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{val:.4f}\n({sign}{delta:.1f}%)', ha='center', va='bottom', fontsize=8, color=color)

    ax.set_ylabel('MAE (lower is better)')
    ax.set_title('Figure 1: Baseline Replication vs. Paper')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylim(0, 0.58)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(f'{OUT}/fig1_baseline_vs_paper.png')
    plt.close(fig)
    print('  fig1_baseline_vs_paper.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Diffusion contribution ablation (stacked / side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_diffusion_ablation():
    direct   = [0.4721, 0.2539, 0.4175, 0.2008]
    full     = [0.4744, 0.2535, 0.4204, 0.2011]
    diff_contrib = [f - d for f, d in zip(full, direct)]

    x = np.arange(len(benchmarks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, direct, w, label='DLinear Only (113K params)', color=COLORS['direct_only'], edgecolor='white')
    ax.bar(x + w/2, full,   w, label='DLinear + Diffusion (843K params)', color=COLORS['diffusion_only'], edgecolor='white')

    for i in range(len(benchmarks)):
        d = diff_contrib[i]
        sign = '+' if d > 0 else ''
        ax.annotate(f'{sign}{d:.4f}', xy=(x[i] + w/2, full[i] + 0.005),
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#c0392b' if d > 0 else '#27ae60')

    ax.set_ylabel('MAE')
    ax.set_title('Figure 2: Diffusion Contributes < 0.3% to Forecast Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylim(0, 0.55)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(f'{OUT}/fig2_diffusion_ablation.png')
    plt.close(fig)
    print('  fig2_diffusion_ablation.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Architecture progression — Exps 15→16→17→18 + baseline
# ══════════════════════════════════════════════════════════════════════════════
def fig3_architecture_progression():
    data = {
        'DLinear BL':       [0.4744, 0.2535, 0.4204, 0.2011],
        'Tiny Transformer': [0.5607, 0.2538, 0.5514, 0.2002],
        'CI Transformer':   [0.5485, 0.2741, 0.4293, 0.1885],
        'CI + Decomp':      [0.5101, 0.2580, 0.4159, 0.2011],
        'HP Sweep Best':    [0.4880, 0.2514, 0.4094, 0.1881],
    }
    colors = [COLORS['baseline'], '#bdc3c7', COLORS['ci_decomp'], '#27ae60', COLORS['sweep']]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    fig.suptitle('Figure 3: Transformer Architecture Progression (MAE by Benchmark)', fontsize=14, y=1.02)

    for idx, (ax, bname) in enumerate(zip(axes, benchmarks_short)):
        vals = [data[k][idx] for k in data]
        names = list(data.keys())
        bars = ax.barh(range(len(vals)), vals, color=colors, edgecolor='white', height=0.6)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(names if idx == 0 else ['' for _ in names])
        ax.set_xlabel('MAE')
        ax.set_title(bname)
        ax.invert_yaxis()
        # mark the best
        best_idx = np.argmin(vals)
        ax.barh(best_idx, vals[best_idx], color=colors[best_idx], edgecolor='#2c3e50', linewidth=2, height=0.6)
        for i, v in enumerate(vals):
            ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig3_architecture_progression.png')
    plt.close(fig)
    print('  fig3_architecture_progression.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Parameter efficiency — MAE vs Params scatter
# ══════════════════════════════════════════════════════════════════════════════
def fig4_param_efficiency():
    # models: (name, params_K, ETTm1_Uni_MAE, ETTm1_Multi_MAE)
    models = [
        ('mr-Diff\n(Baseline)',  843, 0.2011, 0.4204),
        ('Tiny\nTransformer',    295, 0.2002, 0.5514),
        ('CI Transformer',       73,  0.1885, 0.4293),
        ('CI+Decomp',            109, 0.2011, 0.4159),
        ('Sweep Best\n(ETTm1 U)',77,  0.1881, None),
        ('Sweep Best\n(ETTm1 M)',182, None,   0.4094),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 4: Parameter Efficiency — MAE vs Model Size', fontsize=14, y=1.02)

    # ETTm1 Uni
    for name, p, uni, multi in models:
        if uni is not None:
            c = COLORS['baseline'] if p > 500 else COLORS['ci_decomp']
            ax1.scatter(p, uni, s=120, c=c, edgecolors='#2c3e50', linewidths=1, zorder=3)
            ax1.annotate(name, (p, uni), textcoords='offset points', xytext=(8, 6), fontsize=8)
    ax1.set_xlabel('Parameters (K)')
    ax1.set_ylabel('MAE')
    ax1.set_title('ETTm1 Univariate')
    ax1.set_xscale('log')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axhline(0.15, color='#2c3e50', ls='--', lw=0.8, alpha=0.5, label='Paper claim (0.15)')
    ax1.legend(fontsize=8)

    # ETTm1 Multi
    for name, p, uni, multi in models:
        if multi is not None:
            c = COLORS['baseline'] if p > 500 else COLORS['ci_decomp']
            ax2.scatter(p, multi, s=120, c=c, edgecolors='#2c3e50', linewidths=1, zorder=3)
            ax2.annotate(name, (p, multi), textcoords='offset points', xytext=(8, 6), fontsize=8)
    ax2.set_xlabel('Parameters (K)')
    ax2.set_ylabel('MAE')
    ax2.set_title('ETTm1 Multivariate')
    ax2.set_xscale('log')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.axhline(0.37, color='#2c3e50', ls='--', lw=0.8, alpha=0.5, label='Paper claim (0.37)')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig4_param_efficiency.png')
    plt.close(fig)
    print('  fig4_param_efficiency.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Training time comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig5_training_time():
    models = ['mr-Diff\nBaseline', 'Tiny\nTransformer', 'CI\nTransformer', 'CI+Decomp', 'CI+Decomp\n(sweep, 30 cfg)']
    times  = [34, 4.7, 8.6, 11.9, 150]  # minutes
    colors_t = [COLORS['baseline'], '#bdc3c7', COLORS['ci_decomp'], '#27ae60', COLORS['sweep']]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, times, color=colors_t, edgecolor='white', width=0.6)
    for bar, t in zip(bars, times):
        label = f'{t:.0f} min' if t >= 10 else f'{t:.1f} min'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Training Time (minutes)')
    ax.set_title('Figure 5: Training Time Comparison (All 4 Benchmarks)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 175)
    # add annotation for sweep
    ax.annotate('30 configs × 4 benchmarks\n= 120 model trainings', xy=(4, 150), xytext=(2.8, 160),
                fontsize=8, ha='center', va='bottom', color='#7f8c8d')
    fig.savefig(f'{OUT}/fig5_training_time.png')
    plt.close(fig)
    print('  fig5_training_time.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: HP Sweep landscape — all 8 logged configs across 4 benchmarks
# ══════════════════════════════════════════════════════════════════════════════
def fig6_sweep_landscape():
    # From sweep_partial_results.txt — first 8 of 30 configs
    configs = ['exp17_bl', 'cfg01', 'cfg02', 'cfg03', 'cfg04', 'cfg05', 'cfg06', 'cfg07']
    h1m = [0.5069, 0.4908, 0.4881, 0.4914, 0.4946, 0.4962, 0.5106, 0.4880]
    h1u = [0.2939, 0.2514, 0.2631, 0.2538, 0.2756, 0.2542, 0.2602, None]  # cfg07 cut off
    m1m = [0.4181, 0.4217, 0.4134, 0.4201, 0.4210, 0.4181, 0.4192, None]
    m1u = [0.1917, 0.1887, 0.2066, 0.2056, 0.1935, 0.2145, 0.1881, None]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Figure 6: Hyperparameter Sweep — MAE Across Configurations', fontsize=14)

    datasets = [('ETTh1 Multi', h1m), ('ETTh1 Uni', h1u), ('ETTm1 Multi', m1m), ('ETTm1 Uni', m1u)]

    for ax, (name, vals) in zip(axes.flat, datasets):
        valid = [(c, v) for c, v in zip(configs, vals) if v is not None]
        cs, vs = zip(*valid)
        colors_s = [COLORS['sweep'] if v == min(vs) else '#bdc3c7' for v in vs]
        bars = ax.bar(range(len(cs)), vs, color=colors_s, edgecolor='white', width=0.7)
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels(cs, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('MAE')
        ax.set_title(name)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # highlight best
        best_i = np.argmin(vs)
        ax.text(best_i, vs[best_i] - 0.003, f'{vs[best_i]:.4f}', ha='center', va='top',
                fontsize=9, fontweight='bold', color=COLORS['sweep'])
        # baseline line
        baselines = {'ETTh1 Multi': 0.4744, 'ETTh1 Uni': 0.2535, 'ETTm1 Multi': 0.4204, 'ETTm1 Uni': 0.2011}
        ax.axhline(baselines[name], color=COLORS['baseline'], ls='--', lw=1, alpha=0.7, label='DLinear BL')
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig6_sweep_landscape.png')
    plt.close(fig)
    print('  fig6_sweep_landscape.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: All improvements comparison — grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def fig7_all_improvements():
    methods = ['Paper', 'DLinear\nBaseline', 'CI+Decomp\n(Best Single)', 'iTransformer\nEnsemble', 'Two-Scale\nDecomp']
    data = {
        'ETTh1 Multi': [0.42,  0.4744, 0.4829, 0.4773, 0.4858],
        'ETTh1 Uni':   [0.34,  0.2535, 0.2505, 0.2471, 0.2574],
        'ETTm1 Multi': [0.37,  0.4204, 0.4094, 0.4103, 0.4081],
        'ETTm1 Uni':   [0.15,  0.2011, 0.1881, 0.1911, 0.1914],
    }
    method_colors = [COLORS['paper'], COLORS['baseline'], COLORS['ci_decomp'], COLORS['itrans'], COLORS['twoscale']]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Figure 7: MAE Comparison Across All Methods', fontsize=14)

    for ax, bname in zip(axes.flat, data.keys()):
        vals = data[bname]
        bars = ax.bar(range(len(methods)), vals, color=method_colors, edgecolor='white', width=0.65)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel('MAE')
        ax.set_title(bname, fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # annotate values
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=8)
        # star the best non-paper result
        best_ours = min(vals[1:])
        best_i = vals.index(best_ours)
        ax.bar(best_i, vals[best_i], color=method_colors[best_i], edgecolor='#2c3e50', linewidth=2.5, width=0.65)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig7_all_improvements.png')
    plt.close(fig)
    print('  fig7_all_improvements.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Final best results vs paper — clean summary
# ══════════════════════════════════════════════════════════════════════════════
def fig8_final_vs_paper():
    paper = [0.42, 0.34, 0.37, 0.15]
    ours  = [0.4773, 0.2471, 0.4081, 0.1865]
    bench = ['ETTh1\nMulti', 'ETTh1\nUni', 'ETTm1\nMulti', 'ETTm1\nUni']

    x = np.arange(len(bench))
    w = 0.32
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w/2, paper, w, label='Paper (mr-Diff) [1]', color=COLORS['paper'], edgecolor='white')
    b2 = ax.bar(x + w/2, ours,  w, label='Our Best Result',     color=COLORS['ci_decomp'], edgecolor='white')

    for i in range(len(bench)):
        delta = (ours[i] - paper[i]) / paper[i] * 100
        sign = '+' if delta > 0 else ''
        color = '#27ae60' if delta < 0 else '#e74c3c'
        ax.text(x[i] + w/2, ours[i] + 0.008,
                f'{ours[i]:.4f}\n({sign}{delta:.1f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
        ax.text(x[i] - w/2, paper[i] + 0.008, f'{paper[i]:.2f}',
                ha='center', va='bottom', fontsize=9, color=COLORS['paper'])

    ax.set_ylabel('MAE (lower is better)')
    ax.set_title('Figure 8: Final Best Results vs. Paper')
    ax.set_xticks(x)
    ax.set_xticklabels(bench)
    ax.set_ylim(0, 0.58)
    ax.legend(loc='upper right', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # highlight wins
    for i in range(len(bench)):
        if ours[i] < paper[i]:
            ax.annotate('BEATS\nPAPER', xy=(x[i] + w/2, ours[i]),
                        xytext=(x[i] + w/2, ours[i] - 0.04),
                        ha='center', fontsize=7, fontweight='bold', color='#27ae60',
                        arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))

    fig.savefig(f'{OUT}/fig8_final_vs_paper.png')
    plt.close(fig)
    print('  fig8_final_vs_paper.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: Experiment timeline — MAE progression across 27 experiments
# ══════════════════════════════════════════════════════════════════════════════
def fig9_experiment_timeline():
    # Key experiments with ETTm1 Uni MAE (most consistent improvement trajectory)
    exps = {
        'BL':    {'m1u': 0.2011, 'm1m': 0.4204, 'h1u': 0.2535, 'h1m': 0.4744},
        'Exp1':  {'m1u': 0.1988, 'm1m': 0.4224, 'h1u': 0.2543, 'h1m': 0.4765},
        'Exp2':  {'m1u': 0.1999, 'm1m': 0.4218, 'h1u': 0.2523, 'h1m': 0.4719},
        'Exp4':  {'m1u': 0.2049, 'm1m': 0.4216, 'h1u': 0.2531, 'h1m': 0.4790},
        'Exp7':  {'m1u': 0.1913, 'm1m': 0.4819, 'h1u': 0.2558, 'h1m': 0.5653},
        'Exp10': {'m1u': 0.1969, 'm1m': 0.4194, 'h1u': 0.2508, 'h1m': 0.4842},
        'Exp15': {'m1u': 0.2002, 'm1m': 0.5514, 'h1u': 0.2538, 'h1m': 0.5607},
        'Exp16': {'m1u': 0.1885, 'm1m': 0.4293, 'h1u': 0.2741, 'h1m': 0.5485},
        'Exp17': {'m1u': 0.2011, 'm1m': 0.4159, 'h1u': 0.2580, 'h1m': 0.5101},
        'Exp18': {'m1u': 0.1881, 'm1m': 0.4094, 'h1u': 0.2514, 'h1m': 0.4880},
        'Exp26': {'m1u': 0.1904, 'm1m': 0.4197, 'h1u': 0.2645, 'h1m': 0.4875},
        'Exp27': {'m1u': 0.1924, 'm1m': 0.4151, 'h1u': 0.2505, 'h1m': 0.4829},
    }

    exp_names = list(exps.keys())
    x = range(len(exp_names))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('Figure 9: MAE Progression Across Key Experiments', fontsize=14)

    metrics = [('ETTh1 Multi', 'h1m'), ('ETTh1 Uni', 'h1u'), ('ETTm1 Multi', 'm1m'), ('ETTm1 Uni', 'm1u')]

    for ax, (title, key) in zip(axes.flat, metrics):
        vals = [exps[e][key] for e in exp_names]
        # color by phase
        phase_colors = []
        for e in exp_names:
            if e == 'BL':
                phase_colors.append(COLORS['baseline'])
            elif e.startswith('Exp') and int(e[3:]) <= 10:
                phase_colors.append('#e74c3c')  # diffusion phase
            elif e.startswith('Exp') and int(e[3:]) <= 18:
                phase_colors.append(COLORS['ci_decomp'])  # transformer phase
            else:
                phase_colors.append(COLORS['itrans'])  # refinement phase

        ax.plot(x, vals, 'o-', color='#7f8c8d', lw=1, markersize=7, zorder=1)
        for i, (xi, v, c) in enumerate(zip(x, vals, phase_colors)):
            ax.scatter(xi, v, c=c, s=60, edgecolors='#2c3e50', linewidths=0.8, zorder=2)

        # best line
        best = min(vals)
        ax.axhline(best, color=COLORS['ci_decomp'], ls=':', lw=1, alpha=0.5)
        ax.text(len(x)-1, best - 0.002, f'Best: {best:.4f}', fontsize=8, ha='right', color=COLORS['ci_decomp'])

        # baseline line
        bl = exps['BL'][key]
        ax.axhline(bl, color=COLORS['baseline'], ls='--', lw=0.8, alpha=0.5)

        ax.set_xticks(list(x))
        ax.set_xticklabels(exp_names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('MAE')
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['baseline'], markersize=8, label='Baseline'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Diffusion Improvements'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['ci_decomp'], markersize=8, label='Transformer Phase'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['itrans'], markersize=8, label='Refinement/Ensemble'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig9_experiment_timeline.png')
    plt.close(fig)
    print('  fig9_experiment_timeline.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Parameter count comparison (log scale)
# ══════════════════════════════════════════════════════════════════════════════
def fig10_param_comparison():
    models = ['Paper\n(implied)', 'Our\nmr-Diff BL', 'DLinear\nOnly', 'CI+Decomp\n(best uni)', 'CI+Decomp\n(best multi)']
    params = [17500, 843, 113, 54, 182]
    colors_p = ['#2c3e50', COLORS['baseline'], COLORS['direct_only'], COLORS['ci_decomp'], COLORS['ci_decomp']]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, params, color=colors_p, edgecolor='white', width=0.6)
    ax.set_yscale('log')
    ax.set_ylabel('Parameters (K, log scale)')
    ax.set_title('Figure 10: Model Size Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, p in zip(bars, params):
        if p >= 1000:
            label = f'{p/1000:.1f}M'
        else:
            label = f'{p}K'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # reduction annotations
    ax.annotate('20× smaller', xy=(1.5, 500), fontsize=9, ha='center', color='#7f8c8d',
                arrowprops=dict(arrowstyle='->', color='#7f8c8d'), xytext=(1.5, 3000))
    ax.annotate('324× smaller\nthan paper', xy=(3, 54), fontsize=9, ha='center', color='#27ae60',
                xytext=(3, 12), arrowprops=dict(arrowstyle='->', color='#27ae60'))

    fig.savefig(f'{OUT}/fig10_param_comparison.png')
    plt.close(fig)
    print('  fig10_param_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: Ensemble composition — individual vs ensemble MAE
# ══════════════════════════════════════════════════════════════════════════════
def fig11_ensemble_composition():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Figure 11: Ensemble Composition — Individual Member vs. Ensemble MAE', fontsize=13)

    # ETTh1 Multi — ensemble helps
    members_h1m = [0.4929, 0.4927, 0.4875]
    labels_h1m = ['AttnRes+Aug', 'cfg07\n(d=64)', 'cfg02\n(low drop)']
    ensemble_h1m = 0.4829
    baseline_h1m = 0.4744

    x = range(len(members_h1m))
    ax1.bar(x, members_h1m, color=[COLORS['ci_decomp']]*3, edgecolor='white', width=0.5, label='Individual members')
    ax1.axhline(ensemble_h1m, color=COLORS['ensemble'], lw=2.5, ls='-', label=f'Ensemble: {ensemble_h1m:.4f}')
    ax1.axhline(baseline_h1m, color=COLORS['baseline'], lw=1.5, ls='--', label=f'DLinear BL: {baseline_h1m:.4f}')
    for i, v in enumerate(members_h1m):
        ax1.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels_h1m)
    ax1.set_ylabel('MAE')
    ax1.set_title('ETTh1 Multi (Ensemble HELPS)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0.475, 0.50)

    # ETTm1 Multi — ensemble hurts
    members_m1m = [0.4214, 0.4210, 0.4130]
    labels_m1m = ['cfg10\n(champion)', 'cfg02', 'AttnRes']
    ensemble_m1m = 0.4151
    single_best_m1m = 0.4094

    x = range(len(members_m1m))
    ax2.bar(x, members_m1m, color=[COLORS['ci_decomp']]*3, edgecolor='white', width=0.5, label='Individual members')
    ax2.axhline(ensemble_m1m, color=COLORS['ensemble'], lw=2.5, ls='-', label=f'Ensemble: {ensemble_m1m:.4f}')
    ax2.axhline(single_best_m1m, color=COLORS['sweep'], lw=1.5, ls='--', label=f'Sweep best: {single_best_m1m:.4f}')
    for i, v in enumerate(members_m1m):
        ax2.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels_m1m)
    ax2.set_ylabel('MAE')
    ax2.set_title('ETTm1 Multi (Ensemble HURTS)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0.405, 0.43)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig11_ensemble_composition.png')
    plt.close(fig)
    print('  fig11_ensemble_composition.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Diffusion experiments heatmap — 10 experiments, 4 benchmarks
# ══════════════════════════════════════════════════════════════════════════════
def fig12_diffusion_heatmap():
    exp_names = ['Exp1\nJoint E2E', 'Exp2\nSelf-Cond', 'Exp3\nCosine Sched',
                 'Exp4\nv-Pred', 'Exp5\nDecomp Loss', 'Exp7\nMG-TSD',
                 'Exp8\nCross-Ch', 'Exp9\nFFT Denoise',
                 'Exp10\nx0+Decomp']

    # Delta vs baseline (%) — positive = worse
    deltas = np.array([
        [+0.4, +0.3, +0.5, -1.1],   # Exp1
        [-0.5, -0.5, +0.3, -0.6],   # Exp2
        [+42.2, +11.5, +52.5, +7.1], # Exp3
        [+1.5, +0.3, +0.0, +2.5],   # Exp4
        [+0.8, -0.6, +0.7, -2.0],   # Exp5
        [+19.2, +0.9, +14.6, -4.9], # Exp7
        [+43.9, +3.3, +35.1, +3.0], # Exp8
        [+2.3, -0.5, +2.1, +0.5],   # Exp9
        [+2.1, -1.1, -0.2, -2.1],   # Exp10
    ])

    fig, ax = plt.subplots(figsize=(10, 6))
    # clamp for display
    display = np.clip(deltas, -10, 55)
    im = ax.imshow(display, cmap='RdYlGn_r', aspect='auto', vmin=-5, vmax=20)

    ax.set_xticks(range(4))
    ax.set_xticklabels(benchmarks_short)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names, fontsize=9)

    # annotate cells
    for i in range(len(exp_names)):
        for j in range(4):
            v = deltas[i, j]
            sign = '+' if v > 0 else ''
            color = 'white' if abs(v) > 15 else 'black'
            ax.text(j, i, f'{sign}{v:.1f}%', ha='center', va='center', fontsize=9, color=color, fontweight='bold')

    ax.set_title('Figure 12: Diffusion Experiment Results (% Change vs Baseline)\nGreen = improvement, Red = regression', fontsize=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('% Change vs Baseline')

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig12_diffusion_heatmap.png')
    plt.close(fig)
    print('  fig12_diffusion_heatmap.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: MSE comparison — our best vs paper
# ══════════════════════════════════════════════════════════════════════════════
def fig13_mse_comparison():
    paper_mse = [0.411, 0.066, 0.340, 0.039]
    our_mse   = [0.4516, 0.1183, 0.3223, 0.0670]

    x = np.arange(len(benchmarks))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, paper_mse, w, label='Paper (mr-Diff)', color=COLORS['paper'], edgecolor='white')
    ax.bar(x + w/2, our_mse,   w, label='Our Baseline',     color=COLORS['baseline'], edgecolor='white')

    for i in range(len(benchmarks)):
        ax.text(x[i] - w/2, paper_mse[i] + 0.008, f'{paper_mse[i]:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(x[i] + w/2, our_mse[i] + 0.008, f'{our_mse[i]:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('MSE (lower is better)')
    ax.set_title('Figure 13: MSE Comparison — Our Baseline vs. Paper')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(f'{OUT}/fig13_mse_comparison.png')
    plt.close(fig)
    print('  fig13_mse_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Radar chart — improvement methods on 4 benchmarks
# ══════════════════════════════════════════════════════════════════════════════
def fig14_radar():
    categories = ['ETTh1\nMulti', 'ETTh1\nUni', 'ETTm1\nMulti', 'ETTm1\nUni']
    N = len(categories)

    # Normalize: lower MAE = higher score. Use (baseline - method) / baseline * 100
    baseline = [0.4744, 0.2535, 0.4204, 0.2011]
    methods = {
        'CI+Decomp':       [0.4829, 0.2505, 0.4094, 0.1881],
        'iTransformer Ens':[0.4773, 0.2471, 0.4103, 0.1911],
        'Two-Scale':       [0.4858, 0.2574, 0.4081, 0.1914],
    }

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title('Figure 14: Improvement Over Baseline by Method\n(higher = better)', fontsize=12, pad=20)

    method_colors_r = [COLORS['ci_decomp'], COLORS['itrans'], COLORS['twoscale']]
    for (name, vals), color in zip(methods.items(), method_colors_r):
        scores = [(b - v) / b * 100 for b, v in zip(baseline, vals)]
        scores += scores[:1]
        ax.plot(angles, scores, 'o-', linewidth=2, label=name, color=color)
        ax.fill(angles, scores, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.axhline(0, color='#e74c3c', ls='--', lw=1, alpha=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.savefig(f'{OUT}/fig14_radar.png')
    plt.close(fig)
    print('  fig14_radar.png')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: Summary dashboard — key takeaways
# ══════════════════════════════════════════════════════════════════════════════
def fig15_summary_dashboard():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figure 15: Project Summary Dashboard', fontsize=14, y=1.02)

    # Panel 1: Best MAE wins
    ax = axes[0]
    benchmarks_s = ['ETTh1 M', 'ETTh1 U', 'ETTm1 M', 'ETTm1 U']
    paper = [0.42, 0.34, 0.37, 0.15]
    ours  = [0.4773, 0.2471, 0.4081, 0.1865]
    wins = [o < p for o, p in zip(ours, paper)]
    bar_colors = ['#27ae60' if w else '#e74c3c' for w in wins]
    pct = [(o-p)/p*100 for o, p in zip(ours, paper)]
    ax.barh(benchmarks_s, pct, color=bar_colors, edgecolor='white', height=0.5)
    ax.axvline(0, color='black', lw=1)
    for i, (p, w) in enumerate(zip(pct, wins)):
        label = f'{p:+.1f}%'
        ax.text(p + (2 if p > 0 else -2), i, label, ha='left' if p > 0 else 'right',
                va='center', fontsize=10, fontweight='bold', color=bar_colors[i])
    ax.set_xlabel('% vs Paper MAE')
    ax.set_title('vs Paper: 2 Wins, 2 Losses')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Speed
    ax = axes[1]
    categories = ['Diffusion\nBaseline', 'CI+Decomp\nTransformer']
    train_time = [34, 5]
    inference_steps = [60, 1]
    x = range(len(categories))
    ax.bar([0], [34], color=COLORS['baseline'], width=0.35, label='Training (min)')
    ax.bar([1], [5],  color=COLORS['ci_decomp'], width=0.35)
    ax.set_ylabel('Minutes')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(categories)
    ax.set_title(f'Training: 7× Faster')
    ax.text(0, 34+1, '34 min', ha='center', fontsize=10, fontweight='bold')
    ax.text(1, 5+1, '~5 min', ha='center', fontsize=10, fontweight='bold', color=COLORS['ci_decomp'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 3: Model size
    ax = axes[2]
    ax.bar(['Diffusion\nBaseline'], [843], color=COLORS['baseline'], width=0.35)
    ax.bar(['CI+Decomp\n(best)'], [54], color=COLORS['ci_decomp'], width=0.35)
    ax.set_ylabel('Parameters (K)')
    ax.set_title('Model Size: 15× Smaller')
    ax.text(0, 843+20, '843K', ha='center', fontsize=10, fontweight='bold')
    ax.text(1, 54+20, '54K', ha='center', fontsize=10, fontweight='bold', color=COLORS['ci_decomp'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig15_summary_dashboard.png')
    plt.close(fig)
    print('  fig15_summary_dashboard.png')


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating figures...')
    fig1_baseline_vs_paper()
    fig2_diffusion_ablation()
    fig3_architecture_progression()
    fig4_param_efficiency()
    fig5_training_time()
    fig6_sweep_landscape()
    fig7_all_improvements()
    fig8_final_vs_paper()
    fig9_experiment_timeline()
    fig10_param_comparison()
    fig11_ensemble_composition()
    fig12_diffusion_heatmap()
    fig13_mse_comparison()
    fig14_radar()
    fig15_summary_dashboard()
    print('\nDone! 15 figures generated.')
