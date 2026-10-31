"""Download the SMS Spam Collection from Hugging Face into ./data/."""

from pathlib import Path

from datasets import load_dataset

HF_ID = "ucirvine/sms_spam"
OUT_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    ds = load_dataset(HF_ID, split="train")  # the only split published
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "sms_spam.parquet"
    ds.to_parquet(out)

    n_spam = sum(ds["label"])
    print(f"Saved {len(ds)} rows to {out}")
    print(f"ham: {len(ds) - n_spam}, spam: {n_spam}")


if __name__ == "__main__":
    main()
