# Tests

Run from the repo root:

```sh
uv run pytest                  # everything
uv run pytest -m "not slow"    # skip tests that download or load real models
uv run pytest -k fake          # only the contract tests for one embedder (by id)
uv run pytest tests/test_text_embedder_base.py   # one file
uv run pytest -v               # list each test as it runs
```

`uv run` installs the dev dependencies (pytest) and the `llsvm` package from
`src/llsvm/` (in editable mode) on first use.

## Files

- `test_text_embedder_contract.py`: model-agnostic checks that every
  `TextEmbedder` must pass (output shape and dtype, determinism, batch/order
  consistency, stable `model_hash`, input validation).
- `test_text_embedder_base.py`: tests for the `TextEmbedder` base class itself.
- `test_huggingface_text_embedder.py`: `HuggingFaceTextEmbedder`-specific tests
  (slow; loads a real model).
- `test_text_classifier_contract.py`: model-agnostic checks that every
  `TextClassifier` must pass (learns separable data, label mapping, unfitted
  errors, input validation, stable `config_hash`).
- `test_text_classifier_base.py`: tests for the `TextClassifier` base class itself.
- `test_svm.py`: a few targeted `LinearSVM` / `KernelSVM` tests (hyperparameters
  reach `config_hash`, RBF fits XOR where linear can't, balanced class weights).
- `test_kmeans.py`: a few targeted `KMeansClassifier` tests (k=1 gives class
  means, small classes, k=2 fits XOR where k=1 can't).
- `test_logistic_regression.py`, `test_xgboost_classifier.py`: hyperparameters
  reach `config_hash`; XGBoost fits XOR.
- `test_qda.py`: shrinkage reaches `config_hash`; QDA separates two classes
  that differ only in spread, where logistic regression can't.
- `fakes.py`: `FakeEmbedder`, a deterministic model-free embedder. Texts
  starting with `c<k>` (e.g. `"c2 hello"`) land in linearly separable cluster k.
- `conftest.py`: the `EMBEDDER_FACTORIES` and `CLASSIFIER_FACTORIES` registries.

## Adding a new embedder

Add a zero-argument factory to `EMBEDDER_FACTORIES` in `conftest.py` and the
whole contract suite runs against it. Mark ones that load real models as slow:

```python
pytest.param(lambda: HuggingFaceTextEmbedder(...), id="hf-minilm", marks=pytest.mark.slow),
```

## Adding a new classifier

Add a factory that takes a `TextEmbedder` to `CLASSIFIER_FACTORIES` in
`conftest.py`; usually that's just the class:

```python
pytest.param(LinearSVM, id="linear-svm"),
```
