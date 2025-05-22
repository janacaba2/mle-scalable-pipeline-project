import pytest
import pandas as pd
from pathlib import Path
from . import config


@pytest.fixture
def data():
    BASE_DIR =  Path(__file__).resolve().parent.parent
    df = pd.read_csv(BASE_DIR / config.DATA_FOLDER / config.DATA_FILE)
    return df


@pytest.fixture
def high_salary_slice(data):
    return data[data['salary']=='>50K']


@pytest.fixture
def low_salary_slice(data):
    return data[data['salary']=='<=50K']


def test_data_shape(data):
    assert data.shape == data.dropna().shape, "Dropping null changes shape."


def test_age_range_high(high_salary_slice):
    avg_value = high_salary_slice["age"].mean()
    assert (34 < avg_value < 54), "Avg. age for high salary group not in expected range."


def test_age_range_low(low_salary_slice):
    avg_value = low_salary_slice["age"].mean()
    assert (22 < avg_value < 50), "Avg. age for low salary group not in expected range."


def test_education_range_high(high_salary_slice):
    avg_value = high_salary_slice['education-num'].mean()
    assert (9 < avg_value < 13), "Avg. education grade for high salary group not in expected range."


def test_education_range_low(low_salary_slice):
    avg_value = low_salary_slice['education-num'].mean()
    assert (7 < avg_value < 11), "Avg. eduction grade for low salary group not in expected range."


def test_hpw_range_high(high_salary_slice):
    avg_value = high_salary_slice['hours-per-week'].mean()
    assert (34 < avg_value < 56), "Avg. hours-per-week for high salary group not in expected range."


def test_hpw_range_low(low_salary_slice):
    avg_value = low_salary_slice['hours-per-week'].mean()
    assert (26 < avg_value < 50), "Avg. hours-per-week for low salary group not in expected range."
