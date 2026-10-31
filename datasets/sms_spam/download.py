"""Download the SMS Spam Collection from Hugging Face and write data/train.parquet and data/test.parquet."""

from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

HF_ID = "ucirvine/sms_spam"
OUT_DIR = Path(__file__).resolve().parent / "data"
TEST_SIZE = 0.2
SEED = 0


def main() -> None:
    df = load_dataset(HF_ID, split="train").to_pandas()  # the only split published
    df = df.drop_duplicates("sms").rename(columns={"sms": "text"})
    df["label"] = df["label"].map({0: "ham", 1: "spam"})
    train, test = train_test_split(df[["text", "label"]], test_size=TEST_SIZE, stratify=df["label"], random_state=SEED)

    OUT_DIR.mkdir(exist_ok=True)
    for name, part in (("train", train), ("test", test)):
        part.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(part)} rows {part['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
