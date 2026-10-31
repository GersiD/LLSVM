# Emotion

20,000 English tweets labeled with one of six emotions.

- Source: [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) on Hugging Face (`split` config)
- Original: Saravia et al., "CARER: Contextualized Affect Representations for Emotion Recognition", EMNLP 2018

## Download

```sh
uv run datasets/emotion/download.py
```

This writes `data/train.parquet` and `data/test.parquet` next to this README
(git-ignored).

## Format

| column  | type | meaning                                                      |
|---------|------|--------------------------------------------------------------|
| `text`  | str  | tweet text (lowercased, tokenized by the dataset authors)    |
| `label` | str  | `sadness`, `joy`, `love`, `anger`, `fear` or `surprise`      |

`download.py` keeps the dataset's own split: its 16,000 `train` rows go to
`train.parquet`, and its `validation` and `test` rows (2,000 each) are combined
into `test.parquet` (4,000).

Classes are imbalanced: joy 6,761, sadness 5,797, anger 2,709, fear 2,373,
love 1,641, surprise 719.
