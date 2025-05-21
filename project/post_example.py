import requests

url = "https://mle-scalable-pipeline-project.onrender.com/inference"

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


response = requests.post(url, json=data_json)


status_code = response.status_code
data_res = response.json()

print(status_code)
print(data_res)