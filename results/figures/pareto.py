"""Accuracy vs. latency per dataset, with the Pareto frontier: the methods no other method beats on both.

Latencies within LATENCY_TOL of each other count as equal, since per-text timings that close are within
measurement noise (e.g. k-means k=1 vs. k=5 on MiniLM differ by microseconds)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from figures import style
from figures.data import DATASET_NAMES, SHORT_NAMES, Summary, method_of

# Points are labelled with SHORT_NAMES. The embedder needs no label: it sets the speed, so MiniLM
# and EmbeddingGemma 2 points sit in separate columns (and differ in brightness).
LABEL_SIZE = 8.5  # points
LABEL_GAP = 11  # minimum vertical distance between labels in one column, in points
LATENCY_TOL = 0.10  # latencies within 10% of each other are treated as tied


def dominates(q: tuple[float, float, str], p: tuple[float, float, str]) -> bool:
    """q is at least as accurate as p and not meaningfully slower, and strictly better on one of the two."""
    (q_lat, q_acc, _), (p_lat, p_acc, _) = q, p
    return q_acc >= p_acc and q_lat <= p_lat * (1 + LATENCY_TOL) and (q_acc > p_acc or q_lat < p_lat)


def frontier(points: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """(latency, accuracy, key) points that no other point dominates, fastest first."""
    return sorted((p for p in points if not any(dominates(q, p) for q in points if q is not p)), key=lambda p: p[0])


def spread(ys: np.ndarray) -> np.ndarray:
    """Label positions for sorted point heights `ys` (in points): labels closer than LABEL_GAP form a group,
    spaced LABEL_GAP apart and centred on the group's points. Groups that then collide merge; labels with
    room stay level with their point."""
    groups = [[i] for i in range(len(ys))]
    while True:
        pos = [ys[g].mean() + (np.arange(len(g)) - (len(g) - 1) / 2) * LABEL_GAP for g in groups]
        clash = next((i for i in range(1, len(groups)) if pos[i][0] - pos[i - 1][-1] < LABEL_GAP), None)
        if clash is None:
            return np.concatenate(pos)
        groups[clash - 1:clash + 1] = [groups[clash - 1] + groups[clash]]


def label_points(ax, points: list[tuple[float, float, str]], color: str, bg: str) -> None:
    """Short name right of each point. Points in one column are spread apart so labels don't overlap,
    with a thin leader line when a label had to move."""
    to_pt = 72 / ax.figure.dpi
    disp = [(*ax.transData.transform((x, y)), key) for x, y, key in points]
    disp.sort(key=lambda p: p[0])
    columns, current = [], []
    for p in disp:  # points closer than ~15 px horizontally share a column
        if current and p[0] - current[-1][0] > 15:
            columns.append(current)
            current = []
        current.append(p)
    columns.append(current)

    for column in columns:
        column.sort(key=lambda p: p[1])
        ys = np.array([p[1] * to_pt for p in column])  # display y in points
        placed = spread(ys)
        for (px, py, key), y0, y1 in zip(column, ys, placed):
            dy = y1 - y0
            ax.annotate(SHORT_NAMES.get(method_of(key), key), ax.transData.inverted().transform((px, py)),
                        xytext=(9, dy), textcoords="offset points", ha="left", va="center",
                        fontsize=LABEL_SIZE, color=color, zorder=4,
                        bbox=dict(boxstyle="square,pad=0.15", facecolor=bg, edgecolor="none"),  # mask the frontier line
                        arrowprops=dict(arrowstyle="-", color=color, linewidth=0.5, shrinkA=0, shrinkB=4)
                        if abs(dy) > 3 else None)


def pareto(summary: Summary, theme: str, path: Path) -> None:
    t = style.THEMES[theme]
    keys = [k for k, _ in summary.methods if k in summary.latency_ms]
    fig, axes = style.new_figure(theme, ncols=len(summary.datasets), figsize=(13, 5.4))
    all_lat = [summary.latency_ms[k][d] for k in keys for d in summary.latency_ms[k]]
    xlim = (min(all_lat) * 0.7, max(all_lat) * 4)

    for ax, dataset in zip(axes, summary.datasets):
        points = [(summary.latency_ms[k][dataset], summary.accuracy[k][dataset], k)
                  for k in keys if dataset in summary.latency_ms[k]]
        front = frontier(points)

        # Frontier as a step line: from each frontier point, flat until the next (slower, more accurate) one.
        xs = [p[0] for p in front] + [xlim[1]]
        ys = [p[1] for p in front] + [front[-1][1]]
        ax.step(xs, ys, where="post", color=t["fg2"], linewidth=1.2, linestyle="--", zorder=1)

        for lat, acc, key in points:
            ax.scatter(lat, acc, s=60, color=style.color(key, summary, theme), edgecolor=t["bg"], linewidth=0.8, zorder=3)

        accs = [p[1] for p in points]
        pad = (max(accs) - min(accs)) * 0.08
        ax.set_ylim(min(accs) - pad, max(accs) + pad)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.minorticks_off()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        style.style_axes(ax, theme, grid="both")
        ax.set_title(DATASET_NAMES.get(dataset, dataset), color=t["fg"], fontsize=12, loc="left")
        ax.set_xlabel("ms per text (log scale)", color=t["fg2"], fontsize=11)

    axes[0].set_ylabel("accuracy", color=t["fg2"], fontsize=11)
    style.legend(axes[-1], summary, keys, theme)
    legend = axes[-1].get_legend()  # add the frontier to the shared legend
    handles = legend.legend_handles + [Line2D([], [], color=t["fg2"], linestyle="--", linewidth=1.2,
                                              label="Pareto frontier")]
    axes[-1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9.5,
                    labelcolor=t["fg"], handlelength=1.2, labelspacing=0.5)
    fig.tight_layout()  # fix the layout before converting point positions to display coordinates
    for ax, dataset in zip(axes, summary.datasets):
        label_points(ax, [(summary.latency_ms[k][dataset], summary.accuracy[k][dataset], k)
                          for k in keys if dataset in summary.latency_ms[k]], t["fg"], t["bg"])
    style.save(fig, path, theme)
