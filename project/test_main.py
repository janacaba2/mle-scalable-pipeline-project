from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Welcome to the World of Machine Learning!"}


def test_infer_negative_prediction():
    data_json = {
        "age": 25,
        "workclass": "State-gov",
        "fnlgt": 203488,
        "education": "Bachelors",
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 100,
        "hours-per-week": 35,
        "native-country": "United-States",
    }
    r = client.post("/inference", json=data_json)
    assert r.status_code == 200
    assert r.json() == {"0": "<=50K"}

def test_infer_positive_prediction():
    data_json = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 203488,
        "education": "Masters",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    r = client.post("/inference", json=data_json)
    assert r.status_code == 200
    assert r.json() == {"0": ">50K"}


def test_infer_malformed():
    r = client.post("/inference", json={"bad": "json"})
    print(r.status_code)
    assert r.status_code == 422

def test_infer_outrange_category():
    data_json = {
        "age": 50,
        "workclass": "Love",
        "fnlgt": 203488,
        "education": "Masters",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    r = client.post("/inference", json=data_json)
    assert r.status_code == 422

def test_infer_outrange_numeric():
    data_json = {
        "age": 150,
        "workclass": "Private",
        "fnlgt": 203488,
        "education": "Masters",
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States",
    }
    r = client.post("/inference", json=data_json)
    assert r.status_code == 422
