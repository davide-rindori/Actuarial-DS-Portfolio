"""
Visualisation style configuration for Project 05.

Consistent with Project 04 branding:
- Viridis colour palette (perceptually uniform, colorblind-friendly).
- Helvetica typography.
- Clean spines (top/right removed).
- 300 DPI for all saved outputs.

Dual save system:
- Notebook version: 12×7, with titles, 150 DPI display / 300 DPI file.
- Paper version: 10×6, no titles, 300 DPI, saved to latex/figures/.
Both are rendered natively at their target size — no post-hoc resizing.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os


# --- Country cluster (shared with Project 04) ---

COUNTRIES = {
    'CHE': 'Switzerland',
    'SWE': 'Sweden',
    'NOR': 'Norway',
    'DEUTW': 'West Germany',
    'NLD': 'Netherlands',
    'JPN': 'Japan'
}


# --- Base style (shared properties) ---

_BASE_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "sans-serif"],
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "axes.titleweight": "normal",
    "axes.titlelocation": "center",
    "axes.titlepad": 20,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

# Notebook: larger canvas, smaller fonts (more data visible)
_NOTEBOOK_STYLE = {
    **_BASE_STYLE,
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
}

# Paper: compact canvas, larger fonts (legible at 0.8\textwidth on A4)
_PAPER_STYLE = {
    **_BASE_STYLE,
    "figure.figsize": (10, 6),
    "figure.dpi": 300,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}


def set_style(mode="notebook"):
    """
    Set the global matplotlib style.

    Args:
        mode: "notebook" for interactive work, "paper" for LaTeX export.

    Returns:
        COUNTRIES dict for convenience.
    """
    style = _NOTEBOOK_STYLE if mode == "notebook" else _PAPER_STYLE
    plt.rcParams.update(style)
    sns.set_palette(sns.color_palette("viridis", len(COUNTRIES)))
    return COUNTRIES


def save_dual(fig, name, notebook_dir="../reports/figures", paper_dir="../latex/figures"):
    """
    Save a figure in both notebook and paper formats.

    This is a clean dual save: the paper version is re-rendered at paper
    dimensions without titles — no temporary files, no PIL resizing.

    Args:
        fig: matplotlib Figure object.
        name: Base filename without extension (e.g., "fig01_mortality_surface").
        notebook_dir: Directory for notebook-quality figures (with titles).
        paper_dir: Directory for paper-quality figures (no titles, compact).

    Usage in notebooks:
        fig, ax = plt.subplots()
        ax.set_title("My Title")
        ax.plot(...)
        save_dual(fig, "fig01_my_plot")
        plt.show()
    """
    os.makedirs(notebook_dir, exist_ok=True)
    os.makedirs(paper_dir, exist_ok=True)

    notebook_path = os.path.join(notebook_dir, f"{name}.png")
    paper_path = os.path.join(paper_dir, f"{name}.png")

    # --- Notebook version (as-is, with titles) ---
    fig.savefig(notebook_path, dpi=300, bbox_inches="tight")

    # --- Paper version (remove titles, re-save) ---
    # Store original titles
    original_titles = {}
    for ax in fig.get_axes():
        original_titles[id(ax)] = ax.get_title()
        ax.set_title("")

    # Store and update figure size for paper
    original_size = fig.get_size_inches()
    fig.set_size_inches(10, 6)

    fig.savefig(paper_path, dpi=300, bbox_inches="tight")

    # Restore original state (so notebook display is unaffected)
    fig.set_size_inches(original_size)
    for ax in fig.get_axes():
        ax.set_title(original_titles[id(ax)])
