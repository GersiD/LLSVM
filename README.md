# LLSVM
Classification via LLM encoding then SVM inference for inference speed, and machine processing, for multiple domains.

![Classifier comparison on the SMS Spam Collection](plots/sms_spam_minilm.png)

## macOS

XGBoost and PyTorch each ship an OpenMP runtime (`libomp`), and loading both in one
process segfaults. `.env-macos` points XGBoost at PyTorch's copy. From the repo root, either
pass it per command or set it once for your shell:

    uv run --env-file .env-macos experiment.py
    export UV_ENV_FILE=.env-macos

## TODO

- [ ] Professionalism pass over the whole project: comments, docstrings, naming and consistency.
- [ ] Write the story behind this repo, below the plot.
- [ ] Make experiments more modular: `experiment.py` should only write a results file, and plotting should be a separate script that reads results files.
- [ ] Move experiment settings (datasets, embedders, classifiers and their hyperparameters, latency sample size) out of `experiment.py` into a config file.
- [ ] Add a local LLM classifier (prompting a local model for the label) as a baseline, to measure the speedup of embedding + SVM over LLM inference.
