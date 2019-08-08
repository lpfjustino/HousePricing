from ml.model import model_v1
from resources import *

v1_prediction = model_v1()
output = pd.concat([test['Id'], v1_prediction], axis=1)
output.to_csv(r'C:\Users\lpfjustino\Desktop\output_v1.csv', index=None, header=True)