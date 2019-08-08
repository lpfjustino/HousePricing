import pandas as pd
import numpy as np

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data = pd.concat([train.iloc[:, :-1], test], axis=0)