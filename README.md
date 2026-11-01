# LLSVM

Embed text with a small language model, then classify it with an SVM: as accurate as zero-shot Claude Sonnet 5.5, or more, at 43–320× lower latency per text and no per-call cost.

<!-- TODO: add a three-bullet takeaway or a short "when to use this vs. an LLM API" guide here, above the figure.
     e.g. stable labels + high volume + latency or privacy limits -> embeddings + classical classifier;
     no labeled data, categories that change often, or low volume -> prompt an LLM. -->
<!-- TODO: call the second embedder "EmbeddingGemma 2" (google/embeddinggemma-2) here, in the table and in the figure
     labels (EMBEDDER_NAMES in results/plot_summary.py). "Gemma 2" reads as the Gemma 2 language model. -->

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/plots/summary-dark.png">
  <img alt="Accuracy, time per text and cost of 1M texts: small-LM embeddings plus a classical classifier vs. zero-shot Claude Sonnet 5.5 on the Emotion and SMS Spam datasets" src="results/plots/summary.png">
</picture>

| | SMS Spam (2 classes) | Emotion (6 classes) |
|---|---|---|
| MiniLM + best classifier | 98.8% · **4.9 ms** / text | 74.6% · **6.5 ms** / text |
| Gemma 2 + best classifier | **99.5%** · 30.4 ms / text | **74.7%** · 32.9 ms / text |
| Claude Sonnet 5.5, zero-shot | 98.8% · 1,578 ms / text | 60.3% · 1,420 ms / text |
| Always guess the most common label | 87.3% | 35.0% |

<sub>Accuracy on the test split. Local times are embedding + classifier, one text at a time, mean over 100 test texts on an Apple M2 Pro laptop. Claude times are the median model time per request reported by Claude Code. "Best classifier" is the most accurate of six, picked on the test split.</sub>

<!-- TODO: mention batch throughput. Embedding the whole test set in one batch takes about 0.3-0.4 ms per text
     (predict_seconds / n_test on the "embedding" row of results/*_minilm.csv), about 10x faster than one text at a time. -->
<!-- TODO: add a 5-line predict example (load embedder + trained classifier, classify one text, print label and ms),
     or a small demo, so a visitor can try it. -->

Through the API at list prices, the same zero-shot Claude Sonnet 5.5 call would cost at least about $200 per 1M texts (about $10 with Claude Haiku 5.5). The local pipeline has no per-call cost.

<!-- Estimate: input tokens >= (system prompt + mean text length) / 4 characters per token (about 75-80 tokens), output >= 5
     tokens (the JSON label), at $2 / $10 per 1M tokens for Sonnet 5.5 and $0.10 / $0.50 for Haiku 5.5. It leaves out the
     structured-output schema and any thinking tokens, so it is a floor. Claude Code's own estimate (~$3,000 per 1M texts)
     is about 10x higher because Claude Code adds ~1,100 tokens of its own prompt to every request. -->
<!-- TODO: measure instead of estimate: run `uv run --env-file .env-secrets experiment_claude.py --backend api --limit 200`
     on each dataset (real money, a few cents) to get API cost and latency. Claude Code's prompt also adds latency,
     so a direct API call may be faster than the 1.4-1.6 s measured here, which would shrink the speedup. -->
<!-- TODO: add a few-shot and/or Claude Haiku 5.5 row, so the LLM baseline isn't only zero-shot Sonnet at low effort. -->

## Why this repo exists

**The idea.** Large language models (LLMs) have changed what we can do with text. Yet most classifiers running in production are still support vector machines (SVMs), logistic regressions and gradient-boosted trees. They are cheap to run, their failure modes are well understood, and decades of research cover how they generalize and how to calibrate them. LLSVM (LLMs + SVMs) keeps those models and changes only their input. A small pretrained language model turns each text into an embedding, and a classical classifier makes the decision from that vector.

**Why it should be fast.** The language model only reads the text and never generates any, so a compact encoder like MiniLM is enough. The classifier on top predicts with a dot product or a small tree ensemble. That suits fast inference, on-device computing, and anywhere else a full LLM call is too slow, too expensive or too far from the data.

**What we test.** Each experiment embeds a dataset once and trains several classifiers on the same embeddings: a majority-class baseline, nearest-centroid (k-means), logistic regression, linear and RBF SVMs, and XGBoost. Each one is scored on accuracy, macro-F1 (F1 averaged equally over classes, so rare classes count), training time and per-text latency. As the LLM baseline, Claude Sonnet 5.5 labels every test text zero-shot, one request per text.

**What we found.** On SMS Spam, the MiniLM pipeline ties Claude at 98.8% accuracy, and Gemma 2 with an RBF SVM edges past it at 99.5%. That dataset is close to solved, though (guessing "not spam" every time already scores 87%), so gaps this small are probably noise. Emotion is the harder test, and there the classical pipeline wins clearly: 74.6% for MiniLM against 60.3% for Claude, whose macro-F1 is 0.55. Even plain logistic regression on MiniLM embeddings (69.0%) beats it. Almost all of the local inference time goes to the encoder, at about 5 ms per text for MiniLM and 30 ms for Gemma 2. The classifier adds 0.01–0.3 ms, apart from the RBF SVM (up to 2.6 ms on Emotion).

<!-- TODO: check whether the Emotion labels come from hashtags (distant supervision in Saravia et al., EMNLP 2018).
     If so, say that zero-shot Claude is partly penalized for the labeling convention, not only for understanding less. -->

**Limitations.** The classifiers learn from each dataset's labeled training split, while Claude sees only the label names, so this compares a trained pipeline with an untrained one. The LLM baseline is one model with a short zero-shot prompt and low effort. Few-shot examples or more effort might close the Emotion gap. Claude's latency includes the network round trip and Claude Code's overhead. Other gaps: two embedders, one train/test split with no confidence intervals, the best classifier picked on the test split, default hyperparameters, and no measurements on a phone or other edge device.

<!-- TODO: learning curve. Train each classifier on 10, 25, 50, 100, 250, 500, 1k, ... labeled examples per dataset and
     plot accuracy against training-set size, with zero-shot Claude as a horizontal line. The point where the curve
     crosses Claude is "how many labels you need to beat an LLM", which answers the trained vs. zero-shot objection
     head on. Runs locally, no API cost. -->
<!-- TODO: once TODO.md's model-selection item is done, replace "picked on the test split" and "no confidence intervals"
     above with the new protocol. -->

## Running it

    uv run experiment.py                     # SMS Spam, every embedder
    uv run experiment.py --dataset emotion
    uv run experiment_claude.py --dataset emotion   # Claude baseline (asks before sending requests)

Each `experiment.py` run goes through every embedder in `EMBEDDERS` and writes `results/<dataset>_<embedder>.csv` and `results/<dataset>_<embedder>.json` for each. `experiment_claude.py` writes to `results/claude/`. Then:

    uv run results/plot_summary.py   # headline figure: results/plots/summary.png
    uv run results/plot.py           # per-run plots: results/plots/<dataset>_<embedder>.png

Per-run plots compare all six classifiers, e.g. [SMS Spam with MiniLM](results/plots/sms_spam_minilm.png).

### macOS

XGBoost and PyTorch each ship an OpenMP runtime (`libomp`), and loading both in one
process segfaults. `.env-macos` points XGBoost at PyTorch's copy. From the repo root, either
pass it per command or set it once for your shell:

    uv run --env-file .env-macos experiment.py
    export UV_ENV_FILE=.env-macos

<!-- TODO: author section: your name, links (site, LinkedIn, Google Scholar), one line on how this connects to your
     research, and a "what I'd extend next" line (e.g. on-device benchmark, more datasets, fine-tuned small transformer). -->
