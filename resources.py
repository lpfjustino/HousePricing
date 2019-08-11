import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
data = pd.concat([train.iloc[:, :-1], test], axis=0)

def get_categoric_variables(data):
    categoric_variables = data.select_dtypes(include='object')
    return categoric_variables


def get_numeric_variables(data):
    numeric_variables = data.select_dtypes(include=['int64', 'float64'])
    return numeric_variables

