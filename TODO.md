# TODO

- [ ] Move experiment settings (datasets, embedders, classifiers and their hyperparameters, latency sample size) out of `experiment.py` into a config file.
- [ ] Add a local LLM classifier (prompting a local model for the label) as a baseline, to measure the speedup of embedding + SVM over LLM inference.
- [ ] Add CI (e.g. a GitHub Actions workflow) that runs `uv run pytest` on every push and PR, and a badge in the README.
