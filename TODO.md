# TODO

- [ ] Professionalism pass over the whole project: comments, docstrings, naming and consistency.
- [ ] Make experiments more modular: `experiment.py` should only write a results file, and plotting should be a separate script that reads results files.
- [ ] Move experiment settings (datasets, embedders, classifiers and their hyperparameters, latency sample size) out of `experiment.py` into a config file.
- [ ] Add a local LLM classifier (prompting a local model for the label) as a baseline, to measure the speedup of embedding + SVM over LLM inference.
