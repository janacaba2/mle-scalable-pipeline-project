import pytest
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(str(Path(__file__).resolve().parent.parent))
from .ml.data import process_data
from . import config


@pytest.fixture
def data():  # pylint: disable=redefined-outer-name
    """Load data"""
    BASE_DIR =  Path(__file__).resolve().parent.parent
    data = pd.read_csv(BASE_DIR / config.DATA_FOLDER / config.DATA_FILE)
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    return [train, test]


@pytest.fixture
def cat_features():  # pylint: disable=redefined-outer-name
    """Obtain categorical feature columns"""
    return config.CAT_FEATURES


@pytest.fixture
def processed_train_data(data, cat_features):  # pylint: disable=redefined-outer-name
    """Obtain train data as fixture"""
    train, _ = data

    return process_data(
        train, categorical_features=cat_features, label="salary", training=True
        )


@pytest.fixture
def processed_test_data(data, cat_features, processed_train_data):  # pylint: disable=redefined-outer-name
    """Obtain test data as fixture"""
    _, test = data
    _, _, encoder, lb = processed_train_data

    return process_data(
        test, categorical_features=cat_features,
        label="salary", training=False,
        encoder=encoder, lb=lb
    )


def test_process_data(processed_train_data, processed_test_data):
    """Test preprocessing data"""
    X_train, _, encoder, lb = processed_train_data

    X_test, _, encoder2, lb2 = processed_test_data

    # check train vs test data shape
    assert X_train.shape[0] > X_test.shape[0], "Training data shape is not larger then validation data"

    # check that encoder and lb are the same object
    assert encoder is encoder2, "One-Hot-Encoders do not point to the same object"
    assert lb is lb2, "Label Binarizers do not point to the same object"


@pytest.fixture
def model(processed_train_data):  # pylint: disable=redefined-outer-name
    """Train a ML model"""
    from .ml.model import train_model
    X_train, y_train, _, _ = processed_train_data

    tmp_model = train_model(X_train, y_train)
    return tmp_model


def test_train_model(model):
    """Test trained model on prediction attribute"""
    assert hasattr(model, "predict"), "Model should have a predict method"


def test_inference(processed_test_data, model):
    """Test inference of ML Model"""
    from .ml.model import inference

    X_test, y_test, _, _ = processed_test_data
    pred = inference(model, X_test)

    assert pred.shape[0] == X_test.shape[0], "Prediction and test input should have the same shape"

    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.5, f"Model accuracy should be > 0.5, got {accuracy}."
