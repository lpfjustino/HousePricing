import pandas as pd
import numpy as np

def get_categoric_variables(data):
    categoric_variables = data.select_dtypes(include='object')
    return categoric_variables


def get_numeric_variables(data):
    numeric_variables = data.select_dtypes(include=['int64', 'float64'])
    return numeric_variables


def drop_outliers(data):
    outliers = [30, 88, 462, 631, 1322]
    data.drop(data.index[outliers], inplace=True)

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
drop_outliers(train)
data = pd.concat([train.iloc[:, :-1], test], axis=0)