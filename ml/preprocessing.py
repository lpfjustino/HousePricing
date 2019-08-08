from sklearn.impute import SimpleImputer
from eda.stats import features_missing_too_many
from resources import *

def preprocess():
    drop_missing()
    fill_missing_numeric()
    fill_missing_cathegoric()
    fill_remaining()


def fill_remaining():
    # Filling the rest of the features with the mode
    data.fillna(data.mode().iloc[0, :], inplace=True)


def fill_missing_cathegoric():
    # Filling missing categoric features
    features = ['Fence', 'FireplaceQu']
    for feature in features:
        data[feature].fillna('Missing', inplace=True)


def fill_missing_numeric():
    # Imputing missing numerical features
    mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputed_features = ['LotFrontage', 'MasVnrArea', 'GarageArea']
    mean_imputer.fit(data[imputed_features])
    data[imputed_features] = mean_imputer.transform(data[imputed_features])


def drop_missing():
    # Dropping features with too many missing values
    data.drop(features_missing_too_many, axis=1, inplace=True)






