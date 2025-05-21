# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

The model is a Random Forest Classifier from scikit-learn that has been trained with the default settings.

## Intended Use
The Random Forest Classifier predicts whether a person makes over 50K a year based on selected input features.

## Data

The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). Extraction was done by Barry Becker from the 1994 Census database. 
Among 14 available features, 8 features have been identified as categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country) and an One-Hot-Encoder was used to further preprocess them.

The used data set has 32561 rows, and a 80-20 split was used to break this into a train and test set. 
A label binarizer was used to preprocess the output labels.

## Metrics

The model has the following performance:
- Precision: 0.7335849056603774, 
- Recall: 0.6187141947803947, 
- F1-Beta: 0.6712707182320442,

## Ethical Considerations

The model discriminates on race, gender and origin country. In additional to probable unfairness, using such features could be unethical, if not illegal in some production settings.

## Caveats and Recommendations

The data is from the 1994 census database which is no longer representative of the current US society. Given the ethical considerations mentioned above, this model should not be used in production.

Other models and additional features should be considered. Also, bias could also be researched with the library Aequitas for example.

This model was built solely for the purpose of understanding a Machine Learning CI/CD workflow in python using DVC for 