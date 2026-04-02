import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """
    Sets the global aesthetic standard for the project (Viridis, Helvetica, Centered Titles).
    """
    # Global Parameters
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "sans-serif"],
        "axes.titlesize": 16,
        "axes.titleweight": "normal",
        "axes.titlelocation": "center",
        "axes.titlepad": 20,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.figsize": (12, 7),
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.15,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })

    # Color Cluster Definition
    COUNTRIES = {
        'CHE': 'Switzerland',
        'SWE': 'Sweden',
        'NOR': 'Norway',
        'DEUTW': 'West Germany',
        'NLD': 'Netherlands',
        'JPN': 'Japan'
    }

    # Set Viridis Palette
    sns.set_palette(sns.color_palette("viridis", len(COUNTRIES)))
    
    return COUNTRIES