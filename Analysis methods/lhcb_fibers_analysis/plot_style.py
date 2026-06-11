from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SINGLE_COLUMN_SQUARE = (3.35, 3.35)
SINGLE_COLUMN_WIDE = (3.35, 2.45)
DOUBLE_COLUMN_WIDE = (6.85, 4.35)
DIAGNOSTIC_PANEL = (7.1, 4.9)

SEQUENTIAL_CMAP = "viridis"
CARPET_CMAP = "magma"

COLORS = {
    "black": "#111111",
    "gray": "#5F6673",
    "light_gray": "#E6E8EC",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "teal": "#009E73",
    "purple": "#7E57C2",
}


def set_publication_style(*, base_font_size: float = 8.5) -> None:
    """Apply shared matplotlib settings for publication figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": base_font_size,
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.8,
            "axes.labelsize": base_font_size + 0.5,
            "axes.titlesize": base_font_size + 0.5,
            "axes.titleweight": "normal",
            "axes.grid": False,
            "xtick.labelsize": base_font_size - 0.5,
            "ytick.labelsize": base_font_size - 0.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "legend.frameon": False,
            "legend.fontsize": base_font_size - 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.formatter.use_mathtext": True,
        }
    )


def apply_axes_style(ax: Any, *, grid: bool = False, minor_ticks: bool = True) -> None:
    """Apply shared styling to plot axes."""
    if minor_ticks:
        ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)
    if grid:
        ax.grid(True, which="major", color=COLORS["light_gray"], linewidth=0.45)


def style_colorbar(colorbar: Any) -> None:
    """Apply shared styling to a matplotlib colorbar."""
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.tick_params(which="both", direction="in", width=0.8, length=3.2)


def save_figure(fig: Any, path: str | Path, **kwargs: Any) -> None:
    """Save a matplotlib figure, creating parent directories first."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
