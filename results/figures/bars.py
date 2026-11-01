"""Grouped bar charts: datasets on the X axis, one bar per method, a classifier's embedder bars side by side."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figures import style
from figures.data import DATASET_NAMES, Summary, method_of


def bar_chart(summary: Summary, values: dict[str, dict[str, float]], theme: str, path: Path, *,
              ylabel: str, fmt, log: bool = False) -> None:
    t = style.THEMES[theme]
    methods = [k for k, _ in summary.methods if k in values]  # e.g. no latency for the majority class
    fig, ax = style.new_figure(theme, figsize=(13, 5))

    # Offsets inside a dataset group: one method's embedder bars touch, methods are separated by a gap.
    gap, offsets, pos, prev = 0.5, [], 0.0, None
    for key in methods:
        if prev is not None and method_of(key) != prev:
            pos += gap
        offsets.append(pos)
        pos, prev = pos + 1, method_of(key)
    offsets = np.array(offsets) - offsets[-1] / 2
    width = 0.86 / (offsets[-1] - offsets[0] + 1)

    x = np.arange(len(summary.datasets))
    for key, off in zip(methods, offsets):
        xs = x + off * width
        ys = [values[key].get(d, np.nan) for d in summary.datasets]
        ax.bar(xs, ys, width * 0.94, color=style.color(key, summary, theme), hatch=style.hatch(key, summary),
               edgecolor=t["bg"], linewidth=0, zorder=2)
        for xi, yi in zip(xs, ys):
            if not np.isnan(yi):
                ax.annotate(fmt(yi), (xi, yi), xytext=(0, 3), textcoords="offset points", rotation=90,
                            ha="center", va="bottom", fontsize=7.5, color=t["fg"])

    if log:
        ax.set_yscale("log")
        ax.set_ylim(1, max(v for m in values.values() for v in m.values()) * 12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.minorticks_off()
    else:
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])

    style.style_axes(ax, theme)
    ax.set_xticks(x, [DATASET_NAMES.get(d, d) for d in summary.datasets], fontsize=12, color=t["fg"])
    ax.tick_params(axis="x", pad=8)
    ax.set_ylabel(ylabel, color=t["fg2"], fontsize=11)
    style.legend(ax, summary, methods, theme)
    style.save(fig, path, theme)


def accuracy(summary: Summary, theme: str, path: Path) -> None:
    bar_chart(summary, summary.accuracy, theme, path, ylabel="accuracy", fmt=lambda v: f"{v:.1%}")


def speed(summary: Summary, theme: str, path: Path) -> None:
    bar_chart(summary, summary.latency_ms, theme, path, ylabel="ms per text (log scale)", fmt=style.fmt_ms, log=True)
