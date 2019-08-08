from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from eda.stats import features_missing_too_many

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data = pd.concat([train.iloc[:, :-1], test], axis=0)

# Dropping features with too many missing values
data.drop(features_missing_too_many, axis=1, inplace=True)

# Imputing missing numerical features
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_features = ['LotFrontage', 'MasVnrArea', 'GarageArea']
mean_imputer.fit(data[imputed_features])
data[imputed_features] = mean_imputer.transform(data[imputed_features])

# Filling missing categoric features
features = ['Fence', 'FireplaceQu']
for feature in features:
    data[feature].fillna('Missing', inplace=True)

# Filling the rest of the features with the mode
data.fillna(data.mode().iloc[0, :], inplace=True)





