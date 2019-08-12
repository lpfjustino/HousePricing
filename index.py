from ml.model import model_v1, model_v2, model_v2_1, model_v2_2, model_v2_3
from resources import *

# v1_prediction = model_v1()
# output = pd.concat([test['Id'], v1_prediction], axis=1)
# output.to_csv(r'C:\Users\lpfjustino\Desktop\output_v1.csv', index=None, header=True)

# v2_prediction = model_v2()
# output = pd.concat([test['Id'], v2_prediction], axis=1)
# output.to_csv(r'C:\Users\lpfjustino\Desktop\output_v2.csv', index=None, header=True)

# v2_1_prediction = model_v2_1()
# output = pd.concat([test['Id'], v2_1_prediction], axis=1)
# output.to_csv(r'C:\Users\lpfjustino\Desktop\output_v2_1.csv', index=None, header=True)

# v2_2_prediction = model_v2_2()
# output = pd.concat([test['Id'], v2_2_prediction], axis=1)
# output.to_csv(r'C:\Users\lpfjustino\Desktop\output_v2_2.csv', index=None, header=True)

v2_3_prediction = model_v2_3()
output = pd.concat([test['Id'], v2_3_prediction], axis=1)
output.to_csv(r'C:\Users\lpfjustino\Desktop\output_v2_3.csv', index=None, header=True)
