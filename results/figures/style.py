"""Gruvbox look shared by every headline figure: themes, colours, legend and axis styling.

One hue per classifier family (neutral gruvbox accents on the light background, bright ones on dark):
both k-means variants look identical and share one legend entry; only their labels tell them apart. Each
embedder then shifts the hue's brightness, first embedder lightest, so a classifier's bars or points read as
one colour and the embedder as light vs. dark. Claude is orange; the majority class is grey.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from figures.data import EMBEDDER_NAMES, FULL_NAMES, SHORT_NAMES, Summary, embedder_of, legend_label, method_of

FONTS = ["FiraCode Nerd Font", "Fira Code", "Menlo", "DejaVu Sans Mono", "monospace"]

# https://github.com/morhetz/gruvbox, accent -> (bright, neutral)
GRUVBOX = {
    "purple": ("#d3869b", "#b16286"), "aqua": ("#8ec07c", "#689d6a"), "yellow": ("#fabd2f", "#d79921"),
    "blue": ("#83a598", "#458588"), "green": ("#b8bb26", "#98971a"), "red": ("#fb4934", "#cc241d"),
    "orange": ("#fe8019", "#d65d0e"), "gray": ("#a89984", "#928374"),
}
CLASSIFIER_ACCENTS = {"k-means (k=1)": "purple", "k-means (k=5)": "purple", "logistic regression": "yellow",
                      "QDA": "aqua", "linear SVM": "blue", "RBF SVM": "green", "XGBoost": "red"}
SPARE_ACCENTS = ["aqua", "blue", "green", "red", "yellow", "purple"]  # cycled for classifiers not listed above
BRIGHTNESS = (0.35, -0.3)  # per embedder: > 0 mixes toward white, < 0 toward black (more embedders: interpolated)
THEMES = {
    "light": dict(bg="#fbf1c7", fg="#3c3836", fg2="#665c54", grid="#d5c4a1", accent=1, suffix=""),
    "dark": dict(bg="#282828", fg="#ebdbb2", fg2="#bdae93", grid="#3c3836", accent=0, suffix="-dark"),
}


def shade(hex_color: str, amount: float) -> tuple[float, float, float]:
    rgb = np.array(to_rgb(hex_color))
    return tuple(rgb + amount * (1 - rgb) if amount > 0 else rgb * (1 + amount))


def brightness(i: int, n: int) -> float:
    return float(np.linspace(*BRIGHTNESS, n)[i]) if n > 1 else 0.0


def color(key: str, summary: Summary, theme: str) -> tuple[float, float, float]:
    """The method's hue, shifted by its embedder's brightness for "<embedder>:<classifier>" keys."""
    t, method = THEMES[theme], method_of(key)
    base = GRUVBOX[accent(method, summary)][t["accent"]]
    emb = embedder_of(key)
    return shade(base, brightness(summary.embedders.index(emb), len(summary.embedders))) if emb else to_rgb(base)


def accent(method: str, summary: Summary) -> str:
    if method == "claude":
        return "orange"
    if method == "majority":
        return "gray"
    if method in CLASSIFIER_ACCENTS:
        return CLASSIFIER_ACCENTS[method]
    unlisted = [c for c in summary.classifiers if c not in CLASSIFIER_ACCENTS]
    return SPARE_ACCENTS[unlisted.index(method) % len(SPARE_ACCENTS)]


def new_figure(theme: str, ncols: int = 1, **kwargs):
    t = THEMES[theme]
    plt.rcParams["font.family"] = FONTS
    fig, axes = plt.subplots(1, ncols, facecolor=t["bg"], **kwargs)
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(t["bg"])
    return fig, axes


def style_axes(ax, theme: str, grid: str = "y") -> None:
    t = THEMES[theme]
    ax.tick_params(colors=t["fg2"], labelsize=10, length=0)
    ax.grid(axis=grid, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])


def legend(ax, summary: Summary, keys: list[str], theme: str) -> None:
    """One swatch per colour shown (methods that share a colour share the swatch, e.g. "K-Means (km1, km5)"),
    then one grey swatch per embedder showing its brightness, right of `ax`."""
    t = THEMES[theme]
    claude = dict(summary.methods).get("claude", "Claude")
    by_accent: dict[str, list[str]] = {}
    for m in dict.fromkeys(method_of(k) for k in keys):
        by_accent.setdefault(accent(m, summary), []).append(m)
    handles = [Patch(color=color(ms[0], summary, theme), label=legend_label(ms[0], claude) if len(ms) == 1
                     else shared_label(ms)) for ms in by_accent.values()]
    handles.append(Patch(color="none", label=""))
    gray = GRUVBOX["gray"][t["accent"]]
    n = len(summary.embedders)
    handles += [Patch(color=shade(gray, brightness(i, n)), label=f"{EMBEDDER_NAMES.get(e, e)} embeddings")
                for i, e in enumerate(summary.embedders)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=9.5,
              labelcolor=t["fg"], handlelength=1.2, labelspacing=0.5)


def shared_label(methods: list[str]) -> str:
    """["k-means (k=1)", "k-means (k=5)"] -> "K-Means (km1, km5)": the full names' common part, then each short name."""
    fulls = [FULL_NAMES.get(m, m) for m in methods]
    common = os.path.commonprefix(fulls).rstrip(", =k").strip() or " / ".join(fulls)
    return f"{common} ({', '.join(SHORT_NAMES.get(m, m) for m in methods)})"


def save(fig, path: Path, theme: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=THEMES[theme]["bg"])
    plt.close(fig)


def fmt_ms(v: float) -> str:
    return f"{v:,.0f} ms" if v >= 100 else f"{v:.1f} ms"
