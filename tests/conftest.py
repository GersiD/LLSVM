import pytest

from fakes import FakeEmbedder
from llsvm.huggingface_text_embedder import HuggingFaceTextEmbedder
from llsvm.kmeans import KMeansClassifier
from llsvm.logistic_regression import LogisticRegressionClassifier
from llsvm.qda import QDAClassifier
from llsvm.svm import KernelSVM, LinearSVM
from llsvm.xgboost_classifier import XGBoostClassifier
from llsvm.text_embedder import TextEmbedder

# Register every TextEmbedder implementation here as a zero-arg factory so it
# runs the contract tests in test_text_embedder_contract.py. Use
# pytest.param(..., marks=pytest.mark.slow) for ones that load real models.
EMBEDDER_FACTORIES = [
    pytest.param(lambda: FakeEmbedder(16), id="fake"),
    pytest.param(
        lambda: HuggingFaceTextEmbedder("sentence-transformers/all-MiniLM-L6-v2"),
        id="hf-minilm",
        marks=pytest.mark.slow,
    ),
]

# Register every TextClassifier implementation here as a factory taking a
# TextEmbedder, so it runs the contract tests in test_text_classifier_contract.py.
CLASSIFIER_FACTORIES = [
    pytest.param(LinearSVM, id="linear-svm"),
    pytest.param(KernelSVM, id="rbf-svm"),
    pytest.param(lambda emb: KMeansClassifier(emb, k=2), id="kmeans-k2"),
    pytest.param(LogisticRegressionClassifier, id="logreg"),
    pytest.param(QDAClassifier, id="qda"),
    pytest.param(XGBoostClassifier, id="xgboost"),
]


@pytest.fixture(scope="session", params=EMBEDDER_FACTORIES)
def embedder_factory(request):
    return request.param


@pytest.fixture(scope="session")
def embedder(embedder_factory) -> TextEmbedder:
    return embedder_factory()


@pytest.fixture(params=CLASSIFIER_FACTORIES)
def classifier_factory(request):
    return request.param
