"""
Define all functions for train/infer ML model.
"""
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import joblib


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = RandomForestClassifier()

    model.fit(X_train, y_train)
    
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pred = model.predict(X)

    return pred


def save_model(model, filename):
    """ Save model to filename using joblib library.

    Inputs
    ------
    model : ???
        Trained machine learning model or other component (e.g., encoders).
    filename : str
        Filename to save the model to.
    """
    joblib.dump(model, filename)
    print(f"Saved {model.__class__.__name__} to {filename}")


def load_model(filename):
    """ Load model from filename using joblib library.

    Inputs
    ------
    filename : str
        Filename to load the model or other component from.
    Returns
    -------
    model : ???
        Trained machine learning model or other component (e.g., encoders).
    """
    model = joblib.load(filename)
    print(f"Loaded {model.__class__.__name__} from {filename}")

    return model
