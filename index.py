import pandas as pd
import numpy as np
from ml.model import *

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data = pd.concat([train.iloc[:, :-1], test], axis=0)

v1_prediction = model_v1()
output = pd.concat([test['Id'], v1_prediction], axis=1)
output.to_csv(r'C:\Users\lpfjustino\Desktop\output_v1.csv', index=None, header=True)