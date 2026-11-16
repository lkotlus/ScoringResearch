"""
Performance-profile figures: two runs' components on one spider chart, shown
with both runs' composite scores.
"""

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from scoring.constants import COMPONENTS

LABELS = {"cov": "Coverage", "und": "Understanding", "pi": "Integrity",
          "T": "Time eff.", "C": "Cost eff."}
SCORE_LABELS = (("sum", "Weighted sum"), ("geo", "Geometric"), ("auc", "AUC"))

# Reference palette, light mode: categorical blue (slot 1) and red (slot 8),
# validated as a pair for colorblind separation and contrast.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"
SERIES = (("A", "#2a78d6", "-"), ("B", "#e34948", "--"))  # name, color, line style


def comparison_figure(a: dict, b: dict, normalized: bool = False) -> Figure:
    """Spider chart of two scored runs (A in blue, B in red) beside their scores.

    normalized=False plots raw components on a shared 0-1 scale, which is what a
    comparison needs. normalized=True plots each run's own profile p~ = p / max(p),
    so shapes can be compared but magnitudes cannot.
    """
    results = (a, b)
    angles = [2 * math.pi * k / len(COMPONENTS) for k in range(len(COMPONENTS))]
    closed_angles = angles + angles[:1]

    fig = plt.figure(figsize=(8.0, 4.4), facecolor=SURFACE)
    ax = fig.add_axes((0.10, 0.13, 0.40, 0.72), polar=True, facecolor=SURFACE)

    # Axes start at the top and run clockwise.
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"])
    ax.tick_params(axis="y", labelcolor=TEXT_SECONDARY, labelsize=7)
    ax.set_rlabel_position(180 / len(COMPONENTS))  # between the first two axes
    ax.set_xticks(angles)
    ax.set_xticklabels(
        [f"{LABELS[c]}\n{a['components'][c]:.2f} / {b['components'][c]:.2f}" for c in COMPONENTS],
        color=TEXT, fontsize=8.5)
    ax.tick_params(axis="x", pad=8)
    for label, angle in zip(ax.get_xticklabels(), angles):
        side = math.sin(angle)
        label.set_horizontalalignment("left" if side > 0.1 else "right" if side < -0.1 else "center")
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)

    for result, (_, color, style) in zip(results, SERIES):
        source = result["profile"] if normalized else result["components"]
        values = [source[c] for c in COMPONENTS]
        closed_values = values + values[:1]
        ax.fill(closed_angles, closed_values, color=color, alpha=0.15, linewidth=0)
        ax.plot(closed_angles, closed_values, color=color, linewidth=2, linestyle=style)
        ax.plot(angles, values, linestyle="none", marker="o", markersize=6,
                color=color, markeredgecolor=SURFACE, markeredgewidth=1.5)

    # Legend: which run is which
    x_name, x_a, x_b = 0.60, 0.86, 0.96
    for k, (result, (name, color, style)) in enumerate(zip(results, SERIES)):
        y = 0.84 - 0.10 * k
        fig.add_artist(Line2D([x_name, x_name + 0.035], [y + 0.012, y + 0.012], color=color,
                              linewidth=2, linestyle=style, transform=fig.transFigure))
        fig.text(x_name + 0.05, y, f"{name}  Run {result.get('id') or '?'}",
                 color=TEXT, fontsize=10, fontweight="bold")
        if result.get("archetype"):
            fig.text(x_name + 0.05, y - 0.045, f"Archetype {result['archetype']}",
                     color=TEXT_SECONDARY, fontsize=8.5)

    # Scores table
    fig.text(x_name, 0.56, "Composite scores", color=TEXT_SECONDARY, fontsize=9)
    fig.text(x_a, 0.56, "A", color=TEXT_SECONDARY, fontsize=9, ha="right")
    fig.text(x_b, 0.56, "B", color=TEXT_SECONDARY, fontsize=9, ha="right")
    for k, (key, label) in enumerate(SCORE_LABELS):
        y = 0.48 - 0.075 * k
        fig.text(x_name, y, label, color=TEXT, fontsize=10)
        fig.text(x_a, y, f"{a['scores'][key]:.3f}", color=TEXT, fontsize=10,
                 ha="right", family="monospace")
        fig.text(x_b, y, f"{b['scores'][key]:.3f}", color=TEXT, fontsize=10,
                 ha="right", family="monospace")

    note = ("Each profile normalized by its own strongest\ncomponent; labels show raw values (A / B)."
            if normalized else "Raw component values; labels show A / B.")
    fig.text(x_name, 0.14, note, color=TEXT_SECONDARY, fontsize=7.5)
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())


def use_headless_backend() -> None:
    matplotlib.use("Agg")
