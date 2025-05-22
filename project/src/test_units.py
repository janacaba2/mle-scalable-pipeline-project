import pytest
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .ml.data import process_data
from . import config


@pytest.fixture
def data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(BASE_DIR, config.DATA_FOLDER, config.DATA_FILE))
    train, test = train_test_split(data, test_size=0.20, random_state=42)
    return [train, test]


@pytest.fixture
def cat_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        ]


@pytest.fixture
def processed_train_data(data, cat_features):

    train, test = data

    return process_data(
        train, categorical_features=cat_features, label="salary", training=True
        )


@pytest.fixture
def processed_test_data(data, cat_features, processed_train_data):

    train, test = data
    X_train, y_train, encoder, lb = processed_train_data

    return process_data(
        test, categorical_features=cat_features,
        label="salary", training=False,
        encoder=encoder, lb=lb
    )


def test_process_data(processed_train_data, processed_test_data):

    X_train, y_train, encoder, lb = processed_train_data

    X_test, y_test, encoder2, lb2 = processed_test_data

    # check train vs test data shape
    assert X_train.shape[0] > X_test.shape[0], "Training data shape is not larger then validation data"

    # check that encoder and lb are the same object
    assert encoder is encoder2, "One-Hot-Encoders do not point to the same object"
    assert lb is lb2, "Label Binarizers do not point to the same object"


@pytest.fixture
def model(processed_train_data):
    from .ml.model import train_model
    X_train, y_train, encoder, lb = processed_train_data

    model = train_model(X_train, y_train)
    return model


def test_train_model(model):
    assert hasattr(model, "predict"), "Model should have a predict method"


def test_inference(processed_test_data, model):
    from .ml.model import inference

    X_test, y_test, encoder2, lb2 = processed_test_data
    pred = inference(model, X_test)

    assert pred.shape[0] == X_test.shape[0], "Prediction and test input should have the same shape"

    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.5, f"Model accuracy should be > 0.5, got {accuracy}."
