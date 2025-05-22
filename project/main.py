"""
###Code for FastAPI###
"""
import os
from fastapi import FastAPI
import joblib
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import my python scripts
from .src import config
from .src.ml.data import process_data
from .src.ml.model import inference
# import pydantic input data schema
from .dataschema import ModelData

# Instantiate the app.
app = FastAPI()

# Load the trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, config.MODEL_FOLDER, config.FILENAME_MLMODEL))
lb = joblib.load(os.path.join(BASE_DIR, config.MODEL_FOLDER, config.FILENAME_LB))
encoder = joblib.load(os.path.join(BASE_DIR, config.MODEL_FOLDER, config.FILENAME_OHE))

# Load Data Info
cat_features = config.CAT_FEATURES


# Define a GET on the specified endpoint.
@app.get("/")
async def say_hello() -> dict[str, str]:
    return {"greeting": "Welcome to the World of Machine Learning!"}


@app.post("/inference")
async def inference_result(input_data: ModelData):
    data = input_data.to_dataframe()

    X, _, _, _ = process_data(data, categorical_features=cat_features,
                              training=False, lb=lb, encoder=encoder)

    pred = inference(model, X)

    result = lb.inverse_transform(pred)

    return pd.Series(result, name=config.OUTPUT_COL).to_dict()
