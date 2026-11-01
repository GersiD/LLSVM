"""Plot experiment results.

Reads results/<stem>.csv and results/<stem>.json written by experiment.py and
writes a dot plot of macro-F1 and accuracy per classifier to results/plots/<stem>.png.

    uv run results/plot.py                    # every results file
    uv run results/plot.py emotion_minilm     # just these stems
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent
PLOTS = RESULTS / "plots"

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# Categorical slots 1-4, in fixed order (validated colorblind-safe for adjacent pairs).
FAMILY_COLORS = {"k-means": "#2a78d6", "logistic regression": "#1baf7a", "discriminant analysis": "#8b5cd6",
                 "SVM": "#eb6834", "gradient boosting": "#eda100"}


def plot(results: pd.DataFrame, title: str, subtitle: str, path: Path) -> None:
    """Dot plot of macro-F1 and accuracy per classifier, colored by family."""
    results = results[results["family"].isin(FAMILY_COLORS)]
    names = list(results.index)
    families = list(results["family"])
    # One row per classifier, top to bottom, with an extra gap between families.
    y, pos = [], 0.0
    for i, fam in enumerate(families):
        if i and fam != families[i - 1]:
            pos += 0.6
        y.append(-pos)
        pos += 1
    y = np.array(y)
    colors = [FAMILY_COLORS[f] for f in families]

    plt.rcParams["font.family"] = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(names) + 2.4), sharey=True, facecolor=SURFACE)
    for ax, metric, label in zip(axes, ["macro_f1", "accuracy"], ["Macro F1", "Accuracy"]):
        values = results[metric].to_numpy()
        ax.set_facecolor(SURFACE)
        ax.scatter(values, y, s=80, c=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, v in zip(y, values):
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(9, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)

        lo, hi = values.min(), values.max()
        pad = (hi - lo) * 0.08
        ax.set_xlim(lo - pad, min(1.0, hi + 3 * pad) if hi + 3 * pad > 1.0 else hi + 3 * pad)
        ax.set_title(label, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0)
        ax.tick_params(axis="y", length=0)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)

    axes[0].set_yticks(y, names, fontsize=10, color=INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c, label=f)
               for f, c in FAMILY_COLORS.items() if f in families]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=10,
               labelcolor=INK_2, bbox_to_anchor=(0.99, 0.985))
    fig.text(0.01, 0.975, title, ha="left", va="top", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=4)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_stem(stem: str) -> Path:
    results = pd.read_csv(RESULTS / f"{stem}.csv", index_col="classifier")
    meta = json.loads((RESULTS / f"{stem}.json").read_text())

    best = results[results["family"].isin(FAMILY_COLORS)]["macro_f1"].idxmax()
    model_id = meta["embedder_config"].get("model_id", meta["embedder"]).split("/")[-1]
    baseline = results.loc["majority class"]
    path = PLOTS / f"{stem}.png"
    plot(
        results,
        title=f"{best[0].upper()}{best[1:]} leads on the {meta['dataset_title']}",
        subtitle=(f"{model_id} embeddings · {meta['n_train']:,} train / {meta['n_test']:,} test · "
                  f"majority-class baseline: macro F1 {baseline['macro_f1']:.3f}, "
                  f"accuracy {baseline['accuracy']:.3f}"),
        path=path,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stems", nargs="*", help="results to plot, e.g. emotion_minilm (default: all)")
    args = parser.parse_args()

    stems = args.stems or sorted(p.stem for p in RESULTS.glob("*.json"))
    PLOTS.mkdir(exist_ok=True)
    for stem in stems:
        print(f"wrote {plot_stem(stem).relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
