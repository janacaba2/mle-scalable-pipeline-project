"""
Definition of all the repeatedly used config information
"""

MODEL_FOLDER = "model"
DATA_FOLDER = "data"
DATA_FILE = "clean_census.csv"
FILENAME_OHE = "onehotencoder.pkl"
FILENAME_LB = "labelbinarizer.pkl"
FILENAME_MLMODEL = "randomforestclassifier.pkl"

OUTPUT_COL = "salary"

SCORE_OUTPUT_FILE = "slice_output.txt"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
