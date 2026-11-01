"""Load experiment results into one table of methods x datasets.

A method key is either "<embedder>:<classifier>" (a classifier on that embedder's embeddings),
"claude" or "majority". Datasets and embedders are discovered from
results/<dataset>_<embedder>.{csv,json} (experiment.py); Claude comes from
results/claude/<dataset>_claude-*_predictions.csv (experiment_claude.py), answered texts only.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[1]

NOT_CLASSIFIERS = {"embedding", "majority class"}  # other rows of results/<dataset>_<embedder>.csv
EMBEDDER_ORDER = ["minilm", "gemma2"]  # unknown embedders follow, sorted
EMBEDDER_NAMES = {"minilm": "MiniLM", "gemma2": "EmbeddingGemma 2"}
DATASET_NAMES = {"sms_spam": "SMS Spam", "emotion": "Emotion"}
# Method names: full ones for legends, short ones for point labels. Legends show "Full Name (SHORT)".
FULL_NAMES = {"k-means (k=1)": "K-Means, k=1", "k-means (k=5)": "K-Means, k=5", "logistic regression": "Logistic Regression",
              "QDA": "Quadratic Discriminant Analysis", "linear SVM": "Linear SVM", "RBF SVM": "RBF SVM", "XGBoost": "XGBoost", "majority": "Majority Class"}
SHORT_NAMES = {"k-means (k=1)": "km1", "k-means (k=5)": "km5", "logistic regression": "LR", "QDA": "QDA",
               "linear SVM": "SVM", "RBF SVM": "RBF", "XGBoost": "XGB", "claude": "Claude"}


def method_of(key: str) -> str:
    """"minilm:linear SVM" -> "linear SVM"; "claude" -> "claude"."""
    return key.split(":", 1)[-1]


def legend_label(method: str, claude_label: str = "Claude") -> str:
    """"logistic regression" -> "Logistic Regression (LR)". Claude keeps its name: its point label is "Claude"."""
    if method == "claude":
        return claude_label
    full, short = FULL_NAMES.get(method, method), SHORT_NAMES.get(method)
    return f"{full} ({short})" if short else full


def embedder_of(key: str) -> str | None:
    """"minilm:linear SVM" -> "minilm"; "claude" -> None."""
    return key.split(":", 1)[0] if ":" in key else None


@dataclass
class Summary:
    datasets: list[str]
    embedders: list[str]
    classifiers: list[str]  # in the order experiment.py writes them; a new classifier shows up on its own
    methods: list[tuple[str, str]]  # (key, label), classifier pairs side by side, then Claude, then majority
    accuracy: dict[str, dict[str, float]]  # key -> dataset -> accuracy on the test split
    latency_ms: dict[str, dict[str, float]]  # key -> dataset -> ms per text (no entry for the majority class)


def _claude_name(stem: str) -> str:
    m = re.search(r"claude-(\w+?)-(\d+)-(\d+)", stem)
    return f"Claude {m[1].title()} {m[2]}.{m[3]}" if m else "Claude"


def load() -> Summary:
    stems: dict[str, dict[str, str]] = {}  # dataset -> embedder -> results stem
    for path in sorted(RESULTS.glob("*.json")):
        meta = json.loads(path.read_text())
        if "embedder" in meta and "dataset" in meta:
            stems.setdefault(meta["dataset"], {})[meta["embedder"]] = path.stem
    found = {e for d in stems.values() for e in d}
    embedders = [e for e in EMBEDDER_ORDER if e in found] + sorted(found - set(EMBEDDER_ORDER))
    datasets = sorted(stems, key=lambda d: DATASET_NAMES.get(d, d))

    acc: dict[str, dict[str, float]] = {}
    ms: dict[str, dict[str, float]] = {}
    classifiers: list[str] = []
    for dataset, by_emb in stems.items():
        for emb, stem in by_emb.items():
            df = pd.read_csv(RESULTS / f"{stem}.csv", index_col="classifier")
            embed_ms = df.loc["embedding", "latency_ms_mean"]
            rows = [c for c in df.index if c not in NOT_CLASSIFIERS]
            classifiers += [c for c in rows if c not in classifiers]
            for clf in rows:
                acc.setdefault(f"{emb}:{clf}", {})[dataset] = float(df.loc[clf, "accuracy"])
                ms.setdefault(f"{emb}:{clf}", {})[dataset] = float(embed_ms + df.loc[clf, "latency_ms_mean"])
            acc.setdefault("majority", {})[dataset] = float(df.loc["majority class", "accuracy"])

    claude = "Claude"
    for dataset in datasets:
        files = sorted((RESULTS / "claude").glob(f"{dataset}_claude-*_predictions.csv"))
        if not files:
            continue
        preds = pd.read_csv(files[0])
        answered = preds[preds["pred"].notna()]
        claude = _claude_name(files[0].name)
        acc.setdefault("claude", {})[dataset] = float((answered["pred"] == answered["label"]).mean())
        ms.setdefault("claude", {})[dataset] = float(answered["latency_ms"].mean())  # mean, like the local pipelines

    methods = [(f"{e}:{c}", f"{EMBEDDER_NAMES.get(e, e)} + {c}") for c in classifiers for e in embedders
               if f"{e}:{c}" in acc]
    if "claude" in acc:
        methods.append(("claude", f"{claude} (zero-shot)"))
    methods.append(("majority", "majority class"))
    return Summary(datasets, embedders, classifiers, methods, acc, ms)
