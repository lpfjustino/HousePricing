from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from eda.stats import features_missing_too_many
from resources import *
from scipy.stats import skew

def preprocess():
    drop_ids()
    drop_missing()
    fill_missing_numeric()
    fill_missing_cathegoric()
    fill_remaining()
    stringify_meaningless_as_numeric()

    dummies = get_dummies()
    transformed_data = log_transform_skewed_data()
    final_data = pd.concat([dummies, transformed_data], axis=1)
    scaler = MinMaxScaler(copy=False)
    scaler.fit(final_data)
    scaler.transform(final_data)

    return final_data


def log_transform_skewed_data():
    categoric_variables = get_categoric_variables(data)
    data_drop = data.drop(columns=categoric_variables, axis=1)
    for col in data_drop.columns:
        if skew(data_drop[col]) > 0.75:
            data_drop[col] = np.log1p(data_drop[col])
    return data_drop


def get_dummies():
    dummies_df = pd.DataFrame()
    categoric_variables = get_categoric_variables(data)
    for name in categoric_variables.columns:
        dummies = pd.get_dummies(data[name], drop_first=False)
        dummies = dummies.add_prefix("{}_".format(name))
        dummies_df = pd.concat([dummies_df, dummies], axis=1)
    return dummies_df


def stringify_meaningless_as_numeric():
    str_vars = ['MSSubClass', 'YrSold', 'MoSold']
    for var in str_vars:
        data[var] = data[var].apply(str)


def drop_ids():
    train.drop("Id", axis=1, inplace=True)


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






