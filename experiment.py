"""Compare classifiers on a text classification dataset.

For every embedder in EMBEDDERS, trains every classifier in `make_classifiers` on
the dataset's train split (all sharing one embedding pass) and scores it on the
test split. Writes scores, train time, batch predict time and single-text latency
(mean, p90, p95, p99) for each classifier and for the embedder to
results/<dataset>_<embedder>.csv, and
the run's metadata (dataset, embedder, split sizes) to results/<dataset>_<embedder>.json.
Plotting is separate: see results/plot.py.

    uv run experiment.py                     # sms_spam, all embedders
    uv run experiment.py --dataset emotion

Each dataset lives in datasets/<name>/, whose download.py writes data/train.parquet
and data/test.parquet with `text` and `label` columns. To add a dataset, add that
folder and register its plot title in DATASETS. To add a classifier, add it to
make_classifiers.
"""

import argparse
import gc
import json
import sys
import time
from collections import Counter
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from llsvm.huggingface_text_embedder import HuggingFaceTextEmbedder
from llsvm.kmeans import KMeansClassifier
from llsvm.logistic_regression import LogisticRegressionClassifier
from llsvm.qda import QDAClassifier
from llsvm.svm import KernelSVM, LinearSVM
from llsvm.text_classifier import TextClassifier
from llsvm.text_embedder import TextEmbedder
from llsvm.xgboost_classifier import XGBoostClassifier

ROOT = Path(__file__).resolve().parent

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
    "gemma2": lambda: HuggingFaceTextEmbedder("google/embeddinggemma-2", progress=True),
    "minilm": lambda: HuggingFaceTextEmbedder("sentence-transformers/all-MiniLM-L6-v2", progress=True),
}


def make_classifiers(embedder: TextEmbedder) -> dict[str, TextClassifier]:
    """The classifiers to compare, by display name, in plot order."""
    return {
        "k-means (k=1)": KMeansClassifier(embedder, k=1),
        "k-means (k=5)": KMeansClassifier(embedder, k=5),
        "logistic regression": LogisticRegressionClassifier(embedder),
        "QDA": QDAClassifier(embedder),
        "linear SVM": LinearSVM(embedder),
        "RBF SVM": KernelSVM(embedder, kernel="rbf"),
        "XGBoost": XGBoostClassifier(embedder),
    }


FAMILIES = {
    KMeansClassifier: "k-means",
    LogisticRegressionClassifier: "logistic regression",
    QDAClassifier: "discriminant analysis",
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

    def add(self, texts: list[str], vecs: np.ndarray) -> None:
        """Store embeddings already computed by the inner embedder."""
        self._cache.update(zip(texts, vecs))

    def _embed(self, texts: list[str]) -> np.ndarray:
        missing =list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            self._cache.update(zip(missing, self.inner.embed(missing)))
        return np.stack([self._cache[t] for t in texts])


# --- Experiment ---------------------------------------------------------------

LATENCY_SAMPLES = 100  # test texts timed one at a time for latency percentiles
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

    test_vecs, embed_seconds = timed(embedder.embed, test.texts)
    rows = [{
        "classifier": "embedding", "family": "embedder", "predict_seconds": embed_seconds,
        **latency_ms(lambda t: embedder.embed([t]), sample), "config_hash": embedder.model_hash,
    }]
    embedder = CachedEmbedder(embedder)
    embedder.add(test.texts, test_vecs)  # reuse the timed pass rather than embedding test again
    embedder.embed(train.texts)

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


# --- Main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="sms_spam")
    args = parser.parse_args()

    train, test = load_split(args.dataset, "train"), load_split(args.dataset, "test")
    print(f"{args.dataset}: {len(train.texts):,} train / {len(test.texts):,} test, "
          f"train classes {dict(Counter(train.labels))}")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    for embedder_name, make_embedder in EMBEDDERS.items():
        print(f"\n=== {embedder_name} ===")
        embedder = make_embedder()
        results = run(train, test, embedder, LATENCY_SAMPLES)
        print(results.drop(columns="config_hash").round(4).to_string())

        stem = f"{args.dataset}_{embedder_name}"
        results.to_csv(out / f"{stem}.csv", lineterminator="\n")
        metadata = {
            "dataset": args.dataset,
            "dataset_title": DATASETS[args.dataset],
            "embedder": embedder_name,
            "embedder_config": embedder.config(),
            "n_train": len(train.texts),
            "n_test": len(test.texts),
            "latency_samples": min(LATENCY_SAMPLES, len(test.texts)),
        }
        (out / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"wrote results/{stem}.csv and results/{stem}.json")

        del embedder, results
        gc.collect()  # free the previous model before loading the next


if __name__ == "__main__":
    main()
