"""
Define input data scheme for app POST.
Use pydantic to control input data correctness.
"""
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pandas as pd

EDUCATION_MAP = {
    'Preschool': 1,
    '1st-4th': 2,
    '5th-6th': 3,
    '7th-8th': 4,
    '9th': 5,
    '10th': 6,
    '11th': 7,
    '12th': 8,
    'HS-grad': 9,
    'Some-college': 10,
    'Assoc-voc': 11,
    'Assoc-acdm': 12,
    'Bachelors': 13,
    'Masters': 14,
    'Prof-school': 15,
    'Doctorate': 16
    }


# Define Data Schema
class ModelData(BaseModel):
    age: Annotated[int, Field(examples=[30], description="Age between 16 and 100", ge=16, le=100)]
    workclass: Annotated[Literal["Private", "Self-emp-not-inc", "Local-gov", "?",
                                 "State-gov", "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked"],
                                 Field(examples=['Private'])]
    fnlgt: Annotated[int, Field(examples=[203488])]
    education: Annotated[Literal['HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Assoc-voc',
                                 '11th', 'Assoc-acdm', '10th', '7th-8th', 'Prof-school',
                                 '9th', '12th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool'],
                                 Field(examples=["Some-college"])]
    #education_num: Annotated[int, Field(examples=[10], alias='education-num')]
    marital_status: Annotated[Literal["Married-civ-spouse", "Never-married", "Divorced",
                                      "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
                                      Field(examples=['Divorced'], alias='marital-status')]
    occupation: Annotated[Literal['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
                                  'Prof-specialty', 'Other-service', 'Sales', 'Craft-repair',
                                  'Transport-moving', 'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
                                  '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'],
                                  Field(examples=['Sales'])]
    relationship: Annotated[Literal['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
                            Field(examples=['Husband'])]
    race: Annotated[Literal['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
                    Field(examples=['Black'])]
    sex: Annotated[Literal['Male', 'Female'], Field(examples=['Female'])]
    capital_gain: Annotated[int, Field(examples=[0], alias='capital-gain')]
    capital_loss: Annotated[int, Field(examples=[0], alias='capital-loss')]
    hours_per_week: Annotated[int, Field(examples=[45], alias='hours-per-week')]
    native_country: Annotated[str, Field(examples=['Germany'], alias='native-country')]


    model_config = {
        "validate_by_name": True,
        "populate_by_name": True
    }

    @computed_field(alias="education-num", return_type=int)
    @property
    def education_num(self) -> int:
        return EDUCATION_MAP[self.education]

    def to_dataframe(self) -> pd.DataFrame:
        data = self.model_dump(by_alias=True)
        df = pd.DataFrame([data])
        column_order = ['age', 'workclass', 'fnlgt', 'education', 'education-num',
       'marital-status', 'occupation', 'relationship', 'race', 'sex',
       'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        return df[column_order]
