import pandas as pd
from resources import *

# print("Skewness: %f" % train['SalePrice'].skew())
# print("Kurtosis: %f" % train['SalePrice'].kurt())

# Figuring out missing values proportion
missing_count = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
# print(missing_count, percent)

# Computing how representative are the missing values
missing_data = pd.concat([missing_count, percent], axis=1, keys=['Total', 'Percent'])
features_missing_too_many = missing_data[missing_data['Percent'] > 0.85].index
# print(features_missing_too_many)