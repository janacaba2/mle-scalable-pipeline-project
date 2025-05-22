"""
Script to train machine learning model.
"""
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, save_model
import config


# Load in the data.
current_dir = Path(__file__).parent
data_dir = current_dir.parent / config.DATA_FOLDER / config.DATA_FILE
data = pd.read_csv(data_dir)
# Data train-test split.
train, test = train_test_split(data, test_size=0.20) # noqa: F821
label_col = config.OUTPUT_COL
cat_features = config.CAT_FEATURES
# Process the train data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label_col, training=True
) # noqa: F821
print(f"Training Data Shape: {X_train.shape}")
# Process the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label=label_col, training=False,
    encoder=encoder, lb=lb
)
print(f"Test Data Shape: {X_test.shape}")
# Save one-hot-encoder and label binarizer
lbfile = current_dir.parent / config.MODEL_FOLDER / config.FILENAME_LB
save_model(lb, lbfile)
ohefile = current_dir.parent / config.MODEL_FOLDER / config.FILENAME_OHE
save_model(encoder, ohefile)
# Train and save a model.
model = train_model(X_train, y_train)
modelfile = current_dir.parent / config.MODEL_FOLDER / config.FILENAME_MLMODEL
save_model(model, modelfile)


# Print Model Performance Scores to File
def obtain_slice_scores(data, y, pred, colname, catname):
    """Use original data to identify indices of a data slice
    and use compute metrics on slices of the data

    Inputs
    ------
    data : pd.DataFrame
        Original data before preprocessing.
    y : np.array
        Preprocessed output data.
    pred : np.array
        Predicted outputs from model.
    colname : str
        Column name to slice the data on.
    catname : str
        Unique category name within the column.

    Returns
    -------
    compute_model_metrics : list
        Returns the output of compute_model_metrics (list with 3 scores).

    """
    indices = np.where(data[colname] == catname)
    return compute_model_metrics(y[indices], pred[indices])


with open(config.SCORE_OUTPUT_FILE, "w") as f:
    # Overall test performance
    f.write("*******************Overall Test Performance******************\n")
    pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, pred)
    f.write(f"Precision: {precision}, Recall: {recall}, FBeta: {fbeta}\n\n")

    # Sliced model performance
    for col in cat_features:
        f.write(f"*******************{col} Slicing***********************\n")
        for cat_name in test[col].unique():
            precision, recall, fbeta = obtain_slice_scores(test, y_test,
                                                           pred, col, cat_name)
            f.write(f"** {cat_name} Scores:\n")
            f.write(f"""Precision: {precision}, Recall: {recall},
                    FBeta: {fbeta}\n""")
        f.write('\n')
