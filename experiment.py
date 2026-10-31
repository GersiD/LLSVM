"""Compare classifiers on a text classification dataset.

For each seed, makes a stratified train/test split, trains every classifier in
`make_classifiers` on the same embeddings, and scores it on the test set.
Writes per-split scores to results/<dataset>_<embedder>.csv and a summary plot
(mean ± std over splits) to plots/<dataset>_<embedder>.png.

    uv run experiment.py                                   # sms_spam, minilm, seeds 0-4
    uv run experiment.py --dataset sms_spam --embedder minilm --test-size 0.2 --seeds 0 1 2

To add a dataset, write a loader returning a Dataset and register it in DATASETS.
To add a classifier, add it to make_classifiers.
"""

import argparse
import csv
import sys
import time
from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # flat layout: modules live in src/

from huggingface_text_embedder import HuggingFaceTextEmbedder  # noqa: E402
from kmeans import KMeansClassifier  # noqa: E402
from logistic_regression import LogisticRegressionClassifier  # noqa: E402
from svm import KernelSVM, LinearSVM  # noqa: E402
from text_classifier import TextClassifier  # noqa: E402
from text_embedder import TextEmbedder  # noqa: E402
from xgboost_classifier import XGBoostClassifier  # noqa: E402

# --- Datasets -----------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    name: str  # used in file names
    title: str  # used in plots
    texts: list[str]
    labels: list[Hashable]


def load_sms_spam() -> Dataset:
    path = ROOT / "datasets/sms_spam/data/sms_spam.parquet"
    if not path.exists():
        sys.exit(f"{path} not found; run: uv run datasets/sms_spam/download.py")
    # 403 exact duplicate messages (none with conflicting labels) would otherwise
    # leak between train and test splits.
    df = pd.read_parquet(path).drop_duplicates("sms")
    labels = df["label"].map({0: "ham", 1: "spam"}).tolist()
    return Dataset("sms_spam", "SMS Spam Collection", df["sms"].tolist(), labels)


DATASETS: dict[str, Callable[[], Dataset]] = {
    "sms_spam": load_sms_spam,
}

# --- Embedders and classifiers ------------------------------------------------

EMBEDDERS: dict[str, Callable[[], TextEmbedder]] = {
    "minilm": lambda: HuggingFaceTextEmbedder("sentence-transformers/all-MiniLM-L6-v2", progress=True),
}


def make_classifiers(embedder: TextEmbedder) -> dict[str, TextClassifier]:
    """The classifiers to compare, by display name, in plot order."""
    return {
        "k-means (k=1)": KMeansClassifier(embedder, k=1),
        "k-means (k=5)": KMeansClassifier(embedder, k=5),
        "logistic regression": LogisticRegressionClassifier(embedder),
        "linear SVM": LinearSVM(embedder),
        "RBF SVM": KernelSVM(embedder, kernel="rbf"),
        "XGBoost": XGBoostClassifier(embedder),
    }


FAMILIES = {
    KMeansClassifier: "k-means",
    LogisticRegressionClassifier: "logistic regression",
    LinearSVM: "SVM",
    KernelSVM: "SVM",
    XGBoostClassifier: "gradient boosting",
}


def family(clf: TextClassifier) -> str:
    return FAMILIES[type(clf)]


class CachedEmbedder(TextEmbedder):
    """Memoizes another embedder in memory, so all classifiers and splits share one embedding pass."""

    def __init__(self, inner: TextEmbedder):
        self.inner = inner
        self._cache: dict[str, np.ndarray] = {}

    def config(self) -> dict:
        return self.inner.config()

    @property
    def model_hash(self) -> str:
        return self.inner.model_hash

    @property
    def size(self) -> int:
        return self.inner.size

    def _embed(self, texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            self._cache.update(zip(missing, self.inner.embed(missing)))
        return np.stack([self._cache[t] for t in texts])


# --- Experiment ---------------------------------------------------------------


def run(dataset: Dataset, embedder: TextEmbedder, test_size: float, seeds: list[int]) -> list[dict]:
    """One row per (seed, classifier), plus a majority-class baseline row per seed."""
    print(f"{dataset.name}: {len(dataset.texts)} texts, classes {dict(Counter(dataset.labels))}")
    embedder = CachedEmbedder(embedder)
    embedder.embed(dataset.texts)

    rows = []
    n_classifiers = len(make_classifiers(embedder))
    bar = tqdm(total=len(seeds) * n_classifiers, desc="training", unit="fit")
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.texts, dataset.labels, test_size=test_size, stratify=dataset.labels, random_state=seed
        )
        majority = [Counter(y_train).most_common(1)[0][0]] * len(y_test)
        rows.append({
            "seed": seed, "classifier": "majority class", "family": "baseline",
            "macro_f1": f1_score(y_test, majority, average="macro"),
            "accuracy": accuracy_score(y_test, majority), "fit_seconds": 0.0, "config_hash": "",
        })
        for name, clf in make_classifiers(embedder).items():
            bar.set_postfix_str(f"seed {seed} · {name}")
            start = time.perf_counter()
            clf.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            pred = clf.predict(X_test)
            rows.append({
                "seed": seed, "classifier": name, "family": family(clf),
                "macro_f1": f1_score(y_test, pred, average="macro"),
                "accuracy": accuracy_score(y_test, pred),
                "fit_seconds": fit_seconds, "config_hash": clf.config_hash,
            })
            bar.update()
    bar.close()
    return rows


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["classifier"]))
    summary = df.groupby("classifier", sort=False).agg(
        family=("family", "first"),
        macro_f1=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        accuracy=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        fit_seconds=("fit_seconds", "mean"),
    )
    return summary.loc[order].fillna({"macro_f1_std": 0.0, "accuracy_std": 0.0})


# --- Plot ---------------------------------------------------------------------

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# Categorical slots 1-4, in fixed order (validated colorblind-safe for adjacent pairs).
FAMILY_COLORS = {"k-means": "#2a78d6", "logistic regression": "#1baf7a", "SVM": "#eb6834", "gradient boosting": "#eda100"}


def plot(summary: pd.DataFrame, title: str, subtitle: str, path: Path) -> None:
    """Dot plot of mean macro-F1 and accuracy per classifier, ±1 std whiskers, colored by family."""
    summary = summary[summary["family"] != "baseline"]
    names = list(summary.index)
    families = list(summary["family"])
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
        mean, std = summary[metric].to_numpy(), summary[f"{metric}_std"].to_numpy()
        ax.set_facecolor(SURFACE)
        for yi, m, s, c in zip(y, mean, std, colors):
            ax.plot([m - s, m + s], [yi, yi], color=c, linewidth=2, solid_capstyle="round", alpha=0.45, zorder=2)
            ax.scatter(m, yi, s=80, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{m:.3f}", (m + s, yi), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)

        lo, hi = (mean - std).min(), (mean + std).max()
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


# --- Main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="sms_spam")
    parser.add_argument("--embedder", choices=EMBEDDERS, default="minilm")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]()
    embedder = EMBEDDERS[args.embedder]()
    rows = run(dataset, embedder, args.test_size, args.seeds)
    summary = summarize(rows)
    print(summary.round(4).to_string())

    stem = f"{dataset.name}_{args.embedder}"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    with (ROOT / "results" / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    best = summary[summary["family"] != "baseline"]["macro_f1"].idxmax()
    model_id = embedder.config().get("model_id", args.embedder).split("/")[-1]
    baseline = summary.loc["majority class"]
    plot(
        summary,
        title=f"{best[0].upper()}{best[1:]} leads on the {dataset.title}",
        subtitle=(f"{model_id} embeddings · mean ± 1 std over {len(args.seeds)} stratified "
                  f"{1 - args.test_size:.0%}/{args.test_size:.0%} splits · majority-class baseline: "
                  f"macro F1 {baseline['macro_f1']:.3f}, accuracy {baseline['accuracy']:.3f}"),
        path=ROOT / "plots" / f"{stem}.png",
    )
    print(f"wrote results/{stem}.csv and plots/{stem}.png")


if __name__ == "__main__":
    main()
