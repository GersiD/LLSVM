# SMS Spam Collection

5,574 English SMS messages labeled ham or spam (about 13% spam).

- Source: [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) on Hugging Face
- Original: [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
  (Almeida & Hidalgo, 2011)

## Download

```sh
uv run datasets/sms_spam/download.py
```

This writes `data/train.parquet` and `data/test.parquet` next to this README
(git-ignored).

## Format

| column  | type | meaning          |
|---------|------|------------------|
| `text`  | str  | message text     |
| `label` | str  | `ham` or `spam`  |

The source publishes no split, so `download.py` makes one:

- It drops the 403 exact duplicate messages (none have conflicting labels), so
  copies of a message can't land in both train and test. 5,171 messages remain
  (653 spam).
- It makes a stratified 80/20 split with seed 0: 4,136 train, 1,035 test.
