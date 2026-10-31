# SMS Spam Collection

5,574 English SMS messages labeled ham or spam (about 13% spam).

- Source: [`ucirvine/sms_spam`](https://huggingface.co/datasets/ucirvine/sms_spam) on Hugging Face
- Original: [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
  (Almeida & Hidalgo, 2011)

## Download

```sh
uv run datasets/sms_spam/download.py
```

This writes `data/sms_spam.parquet` next to this README (git-ignored).

## Format

| column  | type | meaning              |
|---------|------|----------------------|
| `sms`   | str  | message text         |
| `label` | int  | 0 = ham, 1 = spam    |

There is only a `train` split; make your own stratified train/test split.

The raw file has 403 exact duplicate messages (none with conflicting labels).
`experiment.py` drops them before splitting, leaving 5,171 messages (653 spam),
so copies of a message can't land in both train and test.
