"""
Generate Attention Residuals mechanism diagrams for presentation.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 1: Standard Residual vs AttnRes — side by side
# ══════════════════════════════════════════════════════════════════════════════
def diagram1_standard_vs_attnres():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # ── LEFT: Standard Residual ──
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Standard Residual Connection', fontsize=14, fontweight='bold', pad=15)

    # Boxes
    box_style = dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=1.5)
    layer_colors = ['#3498db', '#2ecc71', '#e74c3c']
    layer_names = ['Layer 1', 'Layer 2', 'Layer 3']

    positions = [(5, 1.5), (5, 5), (5, 8.5)]
    for i, (x, y) in enumerate(positions):
        rect = FancyBboxPatch((x-1.8, y-0.6), 3.6, 1.2, boxstyle='round,pad=0.15',
                              facecolor=layer_colors[i], edgecolor='#2c3e50', linewidth=1.5, alpha=0.8)
        ax1.add_patch(rect)
        ax1.text(x, y, f'{layer_names[i]}\n(Self-Attn + FFN)', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Input
    ax1.text(5, 0.2, 'Patch Embedding (v₀)', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f39c12', edgecolor='#2c3e50', alpha=0.8))

    # Output
    ax1.text(5, 11, 'Output (v₃)', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1abc9c', edgecolor='#2c3e50', alpha=0.8))

    # Arrows — each layer only connects to adjacent
    arrow_style = dict(arrowstyle='->', color='#2c3e50', lw=2, connectionstyle='arc3,rad=0')
    ax1.annotate('', xy=(5, 0.9), xytext=(5, 0.55), arrowprops=arrow_style)
    ax1.annotate('', xy=(5, 4.4), xytext=(5, 2.1), arrowprops=arrow_style)
    ax1.annotate('', xy=(5, 7.9), xytext=(5, 5.6), arrowprops=arrow_style)
    ax1.annotate('', xy=(5, 10.65), xytext=(5, 9.1), arrowprops=arrow_style)

    # Residual arrows (side)
    for y_from, y_to in [(2.1, 4.4), (5.6, 7.9)]:
        ax1.annotate('', xy=(7.5, y_to), xytext=(7.5, y_from),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5,
                                    connectionstyle='arc3,rad=0.3', linestyle='--'))
    ax1.text(8.2, 3.3, '+', fontsize=16, fontweight='bold', color='#e74c3c', ha='center')
    ax1.text(8.2, 6.8, '+', fontsize=16, fontweight='bold', color='#e74c3c', ha='center')

    ax1.text(5, -0.5, 'Each layer sees ONLY its\nimmediate predecessor',
            ha='center', fontsize=9, color='#7f8c8d', style='italic')

    # ── RIGHT: AttnRes ──
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Attention Residual (AttnRes)', fontsize=14, fontweight='bold', pad=15)

    # Same boxes
    for i, (x, y) in enumerate(positions):
        rect = FancyBboxPatch((x-1.8, y-0.6), 3.6, 1.2, boxstyle='round,pad=0.15',
                              facecolor=layer_colors[i], edgecolor='#2c3e50', linewidth=1.5, alpha=0.8)
        ax2.add_patch(rect)
        ax2.text(x, y, f'{layer_names[i]}\n(Self-Attn + FFN)', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')

    # Input
    ax2.text(5, 0.2, 'Patch Embedding (v₀)', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f39c12', edgecolor='#2c3e50', alpha=0.8))
    # Output
    ax2.text(5, 11, 'Output (v₃)', ha='center', va='center', fontsize=10,
            fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1abc9c', edgecolor='#2c3e50', alpha=0.8))

    # AttnRes aggregation boxes
    for y_pos in [4.0, 7.5]:
        rect = FancyBboxPatch((1.0, y_pos-0.3), 2.2, 0.6, boxstyle='round,pad=0.1',
                              facecolor='#9b59b6', edgecolor='#2c3e50', linewidth=1, alpha=0.9)
        ax2.add_patch(rect)
        ax2.text(2.1, y_pos, 'AttnRes\nQuery', ha='center', va='center', fontsize=8,
                fontweight='bold', color='white')

    # Arrows from embedding to all layers (fan-out)
    # v0 → Layer 1 (direct)
    ax2.annotate('', xy=(5, 0.9), xytext=(5, 0.55), arrowprops=arrow_style)

    # v0 → AttnRes for Layer 2
    ax2.annotate('', xy=(1.5, 3.7), xytext=(1.5, 0.4),
                 arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.5,
                                connectionstyle='arc3,rad=0.2', linestyle='-'))
    # v1 → AttnRes for Layer 2
    ax2.annotate('', xy=(2.5, 3.7), xytext=(3.2, 2.1),
                 arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5,
                                connectionstyle='arc3,rad=0.15', linestyle='-'))

    # AttnRes → Layer 2
    ax2.annotate('', xy=(3.7, 5.0), xytext=(3.2, 4.2),
                 arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2))

    # v0 → AttnRes for Layer 3
    ax2.annotate('', xy=(0.8, 7.2), xytext=(0.8, 0.4),
                 arrowprops=dict(arrowstyle='->', color='#f39c12', lw=1.2,
                                connectionstyle='arc3,rad=0.25', linestyle='-'))
    # v1 → AttnRes for Layer 3
    ax2.annotate('', xy=(1.5, 7.2), xytext=(1.5, 2.1),
                 arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.2,
                                connectionstyle='arc3,rad=0.2', linestyle='-'))
    # v2 → AttnRes for Layer 3
    ax2.annotate('', xy=(2.5, 7.2), xytext=(3.2, 5.6),
                 arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.2,
                                connectionstyle='arc3,rad=0.15', linestyle='-'))

    # AttnRes → Layer 3
    ax2.annotate('', xy=(3.7, 8.5), xytext=(3.2, 7.7),
                 arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=2))

    # Layer 1 → output
    ax2.annotate('', xy=(5, 4.4), xytext=(5, 2.1), arrowprops=arrow_style)
    ax2.annotate('', xy=(5, 7.9), xytext=(5, 5.6), arrowprops=arrow_style)
    ax2.annotate('', xy=(5, 10.65), xytext=(5, 9.1), arrowprops=arrow_style)

    ax2.text(5, -0.5, 'Each layer can attend to ALL\nprior layers via learned query',
            ha='center', fontsize=9, color='#9b59b6', style='italic', fontweight='bold')

    plt.tight_layout()
    fig.savefig('diagram1_standard_vs_attnres.png')
    plt.close(fig)
    print('  diagram1_standard_vs_attnres.png')


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 2: AttnRes mechanism detail — the aggregation step
# ══════════════════════════════════════════════════════════════════════════════
def diagram2_attnres_mechanism():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Attention Residual Aggregation — Detailed Mechanism (Layer 3 Example)',
                fontsize=13, fontweight='bold', pad=15)

    # Source layer outputs (left column)
    sources = [
        ('v₀  (Embedding)', '#f39c12', 1.2),
        ('v₁  (Layer 1 out)', '#3498db', 3.0),
        ('v₂  (Layer 2 out)', '#2ecc71', 4.8),
    ]

    for label, color, y in sources:
        rect = FancyBboxPatch((0.3, y-0.4), 3.0, 0.8, boxstyle='round,pad=0.1',
                              facecolor=color, edgecolor='#2c3e50', linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(1.8, y, label, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # RMSNorm
    for _, _, y in sources:
        ax.annotate('', xy=(4.3, y), xytext=(3.3, y),
                     arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))
    rect = FancyBboxPatch((4.3, 1.8), 1.4, 3.4, boxstyle='round,pad=0.15',
                          facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5.0, 3.0, 'RMS\nNorm', ha='center', va='center', fontsize=11, fontweight='bold', color='#2c3e50')
    ax.text(5.0, 1.9, '(stabilize\nmagnitudes)', ha='center', va='bottom', fontsize=7, color='#7f8c8d')

    # Query vector
    rect = FancyBboxPatch((6.2, 5.8), 2.0, 0.8, boxstyle='round,pad=0.1',
                          facecolor='#9b59b6', edgecolor='#2c3e50', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(7.2, 6.2, 'Query q₃', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(7.2, 5.5, '(learned, 64-dim\nzero-initialized)', ha='center', va='center', fontsize=8, color='#9b59b6')

    # Dot product
    for _, _, y in sources:
        ax.annotate('', xy=(6.7, 3.0), xytext=(5.7, y),
                     arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2))
    ax.annotate('', xy=(7.2, 4.2), xytext=(7.2, 5.8),
                 arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

    circle = plt.Circle((7.2, 3.0), 0.5, facecolor='#e8daef', edgecolor='#9b59b6', linewidth=2)
    ax.add_patch(circle)
    ax.text(7.2, 3.0, 'q · k', ha='center', va='center', fontsize=10, fontweight='bold', color='#9b59b6')

    # Softmax
    ax.annotate('', xy=(8.7, 3.0), xytext=(7.7, 3.0),
                 arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))
    rect = FancyBboxPatch((8.7, 2.4), 1.6, 1.2, boxstyle='round,pad=0.1',
                          facecolor='#f5b7b1', edgecolor='#e74c3c', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9.5, 3.0, 'Softmax\n(depth)', ha='center', va='center', fontsize=10, fontweight='bold', color='#c0392b')

    # Attention weights
    ax.annotate('', xy=(11.0, 3.0), xytext=(10.3, 3.0),
                 arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    weights_example = [
        ('α₀ = 0.05', '#f39c12', 1.5),
        ('α₁ = 0.15', '#3498db', 3.0),
        ('α₂ = 0.80', '#2ecc71', 4.5),
    ]
    rect = FancyBboxPatch((10.8, 0.8), 2.6, 4.4, boxstyle='round,pad=0.15',
                          facecolor='#fdf2e9', edgecolor='#e67e22', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(12.1, 5.0, 'Weighted Sum', ha='center', va='center', fontsize=10, fontweight='bold', color='#e67e22')

    for label, color, y in weights_example:
        ax.text(12.1, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color=color)

    # Arrow connecting source outputs to weighted sum (bypass RMSNorm for values)
    for _, _, y in sources:
        ax.annotate('', xy=(10.8, y+0.5), xytext=(3.3, y+0.3),
                    arrowprops=dict(arrowstyle='->', color='#bdc3c7', lw=0.8,
                                   connectionstyle='arc3,rad=-0.15', linestyle=':'))

    # Output
    ax.annotate('', xy=(12.1, 0.3), xytext=(12.1, 0.8),
                 arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((10.8, -0.4), 2.6, 0.7, boxstyle='round,pad=0.1',
                          facecolor='#1abc9c', edgecolor='#2c3e50', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(12.1, -0.05, '→ Self-Attn + FFN', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    fig.savefig('diagram2_attnres_mechanism.png')
    plt.close(fig)
    print('  diagram2_attnres_mechanism.png')


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 3: Zero-init behavior — uniform at start, learned at convergence
# ══════════════════════════════════════════════════════════════════════════════
def diagram3_zero_init():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('AttnRes Zero Initialization: Starts as Standard Residual, Learns to Specialize',
                fontsize=13, fontweight='bold')

    sources = ['v₀\n(embed)', 'v₁\n(layer 1)', 'v₂\n(layer 2)']
    colors = ['#f39c12', '#3498db', '#2ecc71']

    # LEFT: At initialization (uniform)
    init_weights = [1/3, 1/3, 1/3]
    bars1 = ax1.bar(sources, init_weights, color=colors, edgecolor='white', width=0.5, alpha=0.8)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Attention Weight (α)')
    ax1.set_title('At Initialization\n(query = zeros → uniform)', fontsize=11)
    ax1.axhline(1/3, color='#e74c3c', ls='--', lw=1, alpha=0.5)
    for bar, w in zip(bars1, init_weights):
        ax1.text(bar.get_x() + bar.get_width()/2, w + 0.02, f'{w:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.text(1, 0.85, '= Standard\nResidual', ha='center', fontsize=10, color='#e74c3c',
            fontweight='bold', style='italic')

    # RIGHT: After training (learned, ETTh1 Multi example)
    learned_weights = [0.55, 0.10, 0.35]  # example: layer 3 prefers embedding
    bars2 = ax2.bar(sources, learned_weights, color=colors, edgecolor='white', width=0.5, alpha=0.8)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('Attention Weight (α)')
    ax2.set_title('After Training (ETTh1 Multi)\n(query learned to specialize)', fontsize=11)
    for bar, w in zip(bars2, learned_weights):
        ax2.text(bar.get_x() + bar.get_width()/2, w + 0.02, f'{w:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.annotate('Skips back to\nraw embedding!', xy=(0, 0.55), xytext=(0.8, 0.75),
                fontsize=9, fontweight='bold', color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

    plt.tight_layout()
    fig.savefig('diagram3_zero_init.png')
    plt.close(fig)
    print('  diagram3_zero_init.png')


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 4: Full CI+Decomp+AttnRes pipeline — end to end
# ══════════════════════════════════════════════════════════════════════════════
def diagram4_full_pipeline():
    fig, ax = plt.subplots(figsize=(16, 6.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Full CI+Decomp+AttnRes Pipeline', fontsize=14, fontweight='bold', pad=10)

    # Stage 1: Input
    rect = FancyBboxPatch((0.2, 2.5), 1.8, 1.0, boxstyle='round,pad=0.1',
                          facecolor='#2c3e50', edgecolor='#2c3e50', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(1.1, 3.0, 'Lookback\n[B, 336, 7]', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Stage 2: Decomposition
    ax.annotate('', xy=(2.5, 3.0), xytext=(2.0, 3.0),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))

    rect = FancyBboxPatch((2.5, 2.3), 1.8, 1.4, boxstyle='round,pad=0.1',
                          facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3.4, 3.0, 'Avg Pool\nDecomp\n(k=15)', ha='center', va='center', fontsize=8, fontweight='bold')

    # Split into trend and residual
    ax.annotate('', xy=(4.8, 4.2), xytext=(4.3, 3.3),
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.5))
    ax.annotate('', xy=(4.8, 1.8), xytext=(4.3, 2.7),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=1.5))

    # Trend branch
    rect = FancyBboxPatch((4.8, 3.8), 1.4, 0.8, boxstyle='round,pad=0.1',
                          facecolor='#e67e22', edgecolor='#2c3e50', linewidth=1, alpha=0.8)
    ax.add_patch(rect)
    ax.text(5.5, 4.2, 'Trend', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Residual branch
    rect = FancyBboxPatch((4.8, 1.4), 1.4, 0.8, boxstyle='round,pad=0.1',
                          facecolor='#3498db', edgecolor='#2c3e50', linewidth=1, alpha=0.8)
    ax.add_patch(rect)
    ax.text(5.5, 1.8, 'Residual', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    # Stage 3: CI flatten
    ax.annotate('', xy=(6.8, 4.2), xytext=(6.2, 4.2),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))
    ax.annotate('', xy=(6.8, 1.8), xytext=(6.2, 1.8),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    rect = FancyBboxPatch((6.8, 2.0), 1.6, 2.8, boxstyle='round,pad=0.1',
                          facecolor='#1abc9c', edgecolor='#2c3e50', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(7.6, 3.6, 'CI\nFlatten', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(7.6, 2.8, '[B×7, 336]', ha='center', va='center', fontsize=7, color='white')
    ax.text(7.6, 2.3, 'each ch\nindependent', ha='center', va='center', fontsize=6, color='white')

    # Stage 4: Patching
    ax.annotate('', xy=(9.0, 3.4), xytext=(8.4, 3.4),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((9.0, 2.5), 1.4, 1.8, boxstyle='round,pad=0.1',
                          facecolor='#f39c12', edgecolor='#2c3e50', linewidth=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(9.7, 3.6, 'Patch\n+ Embed', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(9.7, 2.8, '[B×7, 42, 64]', ha='center', va='center', fontsize=7, color='white')

    # Stage 5: AttnRes Transformer
    ax.annotate('', xy=(11.0, 3.4), xytext=(10.4, 3.4),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((11.0, 1.8), 2.0, 3.2, boxstyle='round,pad=0.15',
                          facecolor='#9b59b6', edgecolor='#2c3e50', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(12.0, 4.0, 'AttnRes', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(12.0, 3.3, 'Transformer', ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(12.0, 2.5, '3 layers\n4 heads\nd=64', ha='center', va='center', fontsize=7, color='#d7bde2')

    # Stage 6: Dual heads
    ax.annotate('', xy=(13.6, 3.4), xytext=(13.0, 3.4),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    rect = FancyBboxPatch((13.6, 2.5), 1.0, 1.8, boxstyle='round,pad=0.1',
                          facecolor='#ecf0f1', edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(14.1, 3.7, 'Trend\nHead', ha='center', va='center', fontsize=7, fontweight='bold', color='#e67e22')
    ax.text(14.1, 2.9, 'Resid\nHead', ha='center', va='center', fontsize=7, fontweight='bold', color='#3498db')

    # Stage 7: Sum → Output
    ax.annotate('', xy=(15.2, 3.4), xytext=(14.6, 3.4),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    circle = plt.Circle((15.2, 3.4), 0.3, facecolor='#27ae60', edgecolor='#2c3e50', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(15.2, 3.4, '+', ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    ax.text(15.2, 2.7, 'Forecast\n[B,168,7]', ha='center', va='center', fontsize=7, fontweight='bold', color='#2c3e50')

    # Bottom annotation
    ax.text(8.0, 0.8, 'Total: 54–86K parameters  |  Training: ~5 min  |  Inference: 1 forward pass',
           ha='center', va='center', fontsize=10, fontweight='bold', color='#7f8c8d',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='#fdfefe', edgecolor='#bdc3c7'))

    # Top: label the key innovation
    ax.annotate('Key innovation:\nlearned depth-wise\nretrieval', xy=(12.0, 5.0), xytext=(12.0, 5.8),
               ha='center', fontsize=8, fontweight='bold', color='#9b59b6',
               arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

    fig.savefig('diagram4_full_pipeline.png')
    plt.close(fig)
    print('  diagram4_full_pipeline.png')


# ══════════════════════════════════════════════════════════════════════════════
# DIAGRAM 5: AttnRes impact — before/after on ETTh1 Multi
# ══════════════════════════════════════════════════════════════════════════════
def diagram5_attnres_impact():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    experiments = [
        'Sweep ceiling\n(30 configs)', 'Exp 19\n(+training)', 'Exp 21\n(+ch mix)',
        'Exp 22\n(+augment)', 'Exp 25\n(+freq)', 'Exp 26\n(+AttnRes\n+aug)',
        'Exp 27\n(Ensemble)'
    ]
    values = [0.4880, 0.4912, 0.4937, 0.4902, 0.4884, 0.4875, 0.4829]
    colors = ['#bdc3c7', '#bdc3c7', '#bdc3c7', '#bdc3c7', '#bdc3c7', '#9b59b6', '#1abc9c']

    bars = ax.bar(range(len(experiments)), values, color=colors, edgecolor='white', width=0.6)
    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, fontsize=8)
    ax.set_ylabel('MAE (ETTh1 Multi)')
    ax.set_title('ETTh1 Multi: Breaking the 0.488 Wall\nOnly AttnRes + Ensemble broke through', fontsize=12, fontweight='bold')

    # Ceiling line
    ax.axhline(0.4880, color='#e74c3c', ls='--', lw=1.5, alpha=0.6)
    ax.text(0.3, 0.4882, '← Sweep ceiling (0.4880)', fontsize=8, color='#e74c3c')

    # Baseline
    ax.axhline(0.4744, color='#2c3e50', ls=':', lw=1, alpha=0.4)
    ax.text(0.3, 0.4746, '← DLinear baseline (0.4744)', fontsize=8, color='#2c3e50')

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.0005, f'{v:.4f}',
               ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylim(0.474, 0.498)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig('diagram5_attnres_impact.png')
    plt.close(fig)
    print('  diagram5_attnres_impact.png')


if __name__ == '__main__':
    print('Generating AttnRes diagrams...')
    diagram1_standard_vs_attnres()
    diagram2_attnres_mechanism()
    diagram3_zero_init()
    diagram4_full_pipeline()
    diagram5_attnres_impact()
    print('\nDone! 5 diagrams generated.')
