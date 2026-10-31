"""Download the Emotion dataset from Hugging Face and write data/train.parquet and data/test.parquet."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

HF_ID = "dair-ai/emotion"
OUT_DIR = Path(__file__).resolve().parent / "data"
PARTS = {"train": ["train"], "test": ["validation", "test"]}


def main() -> None:
    splits = load_dataset(HF_ID, "split")
    names = dict(enumerate(splits["train"].features["label"].names))

    OUT_DIR.mkdir(exist_ok=True)
    for name, sources in PARTS.items():
        part = pd.concat([splits[s].to_pandas() for s in sources], ignore_index=True)
        part["label"] = part["label"].map(names)
        part[["text", "label"]].to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(part)} rows {part['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
