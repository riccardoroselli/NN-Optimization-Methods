"""
Shared matplotlib configuration for publication-quality plots.
"""

import matplotlib
import matplotlib.pyplot as plt


def setup_plotting():
    """Configure matplotlib for clean, readable plots."""
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# Consistent colour scheme across all plots
COLORS = {
    'A1-Sub-CM':   '#1f77b4',
    'A1-Sub-NAG':  '#ff7f0e',
    'A1-Prox-CM':  '#2ca02c',
    'A1-Prox-NAG': '#d62728',
    'A2':          '#9467bd',
}

# Extended palette for parameter sweeps
SWEEP_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
