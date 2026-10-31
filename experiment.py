"""Compare classifiers on a text classification dataset.

Trains every classifier in `make_classifiers` on the dataset's train split (all
sharing one embedding pass) and scores it on the test split. Writes scores,
train time, batch predict time and single-text latency (mean, p90, p95, p99) for
each classifier and for the embedder to results/<dataset>_<embedder>.csv, and a
plot to plots/<dataset>_<embedder>.png.

    uv run experiment.py                                   # sms_spam, minilm
    uv run experiment.py --dataset emotion --embedder minilm

Each dataset lives in datasets/<name>/, whose download.py writes data/train.parquet
and data/test.parquet with `text` and `label` columns. To add a dataset, add that
folder and register its plot title in DATASETS. To add a classifier, add it to
make_classifiers.
"""

import argparse
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

DATASETS: dict[str, str] = {  # folder name under datasets/ -> title used in plots
    "sms_spam": "SMS Spam Collection",
    "emotion": "Emotion dataset",
}


@dataclass(frozen=True)
class Split:
    texts: list[str]
    labels: list[Hashable]


def load_split(dataset: str, split: str) -> Split:
    path = ROOT / "datasets" / dataset / "data" / f"{split}.parquet"
    if not path.exists():
        sys.exit(f"{path} not found; run: uv run datasets/{dataset}/download.py")
    df = pd.read_parquet(path)
    return Split(df["text"].tolist(), df["label"].tolist())


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
    """Memoizes another embedder in memory, so all classifiers share one embedding pass."""

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

LATENCY_SAMPLES = 1000  # test texts timed one at a time for latency percentiles
COLUMNS = [
    "family", "macro_f1", "accuracy", "train_seconds", "predict_seconds",
    "latency_ms_mean", "latency_ms_p90", "latency_ms_p95", "latency_ms_p99", "config_hash",
]


def timed(fn: Callable, *args):
    """(fn(*args), seconds taken)."""
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def latency_ms(predict_one: Callable[[str], object], texts: list[str]) -> dict:
    """Single-text latency stats in milliseconds, after one warm-up call."""
    predict_one(texts[0])
    ms = np.array([timed(predict_one, t)[1] * 1000 for t in texts])
    return {
        "latency_ms_mean": ms.mean(),
        **{f"latency_ms_p{q}": np.percentile(ms, q) for q in (90, 95, 99)},
    }


def run(train: Split, test: Split, embedder: TextEmbedder, latency_samples: int) -> pd.DataFrame:
    """One row per classifier, plus a majority-class baseline and an embedding row, indexed by name.

    train_seconds and the predict/latency columns cover the classifier only (embeddings are
    cached); the embedding row holds the embedder's own batch and single-text costs.
    """
    rng = np.random.default_rng(0)
    sample = [test.texts[i] for i in rng.choice(len(test.texts), min(latency_samples, len(test.texts)), replace=False)]

    _, embed_seconds = timed(embedder.embed, test.texts)
    rows = [{
        "classifier": "embedding", "family": "embedder", "predict_seconds": embed_seconds,
        **latency_ms(lambda t: embedder.embed([t]), sample), "config_hash": embedder.model_hash,
    }]
    embedder = CachedEmbedder(embedder)
    embedder.embed(train.texts + test.texts)

    majority = Counter(train.labels).most_common(1)[0][0]
    baseline = [majority] * len(test.labels)
    rows.append({
        "classifier": "majority class", "family": "baseline",
        "macro_f1": f1_score(test.labels, baseline, average="macro"),
        "accuracy": accuracy_score(test.labels, baseline),
    })

    bar = tqdm(make_classifiers(embedder).items(), desc="classifiers", unit="clf")
    for name, clf in bar:
        bar.set_postfix_str(f"{name}: train")
        _, train_seconds = timed(clf.fit, train.texts, train.labels)
        bar.set_postfix_str(f"{name}: predict")
        pred, predict_seconds = timed(clf.predict, test.texts)
        bar.set_postfix_str(f"{name}: latency")
        rows.append({
            "classifier": name, "family": family(clf),
            "macro_f1": f1_score(test.labels, pred, average="macro"),
            "accuracy": accuracy_score(test.labels, pred),
            "train_seconds": train_seconds, "predict_seconds": predict_seconds,
            **latency_ms(lambda t: clf.predict([t]), sample),
            "config_hash": clf.config_hash,
        })
    return pd.DataFrame(rows).set_index("classifier")[COLUMNS]


# --- Plot ---------------------------------------------------------------------

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
# Categorical slots 1-4, in fixed order (validated colorblind-safe for adjacent pairs).
FAMILY_COLORS = {"k-means": "#2a78d6", "logistic regression": "#1baf7a", "SVM": "#eb6834", "gradient boosting": "#eda100"}


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


# --- Main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="sms_spam")
    parser.add_argument("--embedder", choices=EMBEDDERS, default="minilm")
    args = parser.parse_args()

    train, test = load_split(args.dataset, "train"), load_split(args.dataset, "test")
    print(f"{args.dataset}: {len(train.texts):,} train / {len(test.texts):,} test, "
          f"train classes {dict(Counter(train.labels))}")
    embedder = EMBEDDERS[args.embedder]()
    results = run(train, test, embedder, LATENCY_SAMPLES)
    print(results.drop(columns="config_hash").round(4).to_string())

    stem = f"{args.dataset}_{args.embedder}"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "plots").mkdir(exist_ok=True)
    results.to_csv(ROOT / "results" / f"{stem}.csv", lineterminator="\n")

    best = results[results["family"].isin(FAMILY_COLORS)]["macro_f1"].idxmax()
    model_id = embedder.config().get("model_id", args.embedder).split("/")[-1]
    baseline = results.loc["majority class"]
    plot(
        results,
        title=f"{best[0].upper()}{best[1:]} leads on the {DATASETS[args.dataset]}",
        subtitle=(f"{model_id} embeddings · {len(train.texts):,} train / {len(test.texts):,} test · "
                  f"majority-class baseline: macro F1 {baseline['macro_f1']:.3f}, "
                  f"accuracy {baseline['accuracy']:.3f}"),
        path=ROOT / "plots" / f"{stem}.png",
    )
    print(f"wrote results/{stem}.csv and plots/{stem}.png")


if __name__ == "__main__":
    main()
