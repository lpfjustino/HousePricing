import pandas as pd

df_train = pd.read_csv('input/train.csv')

print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())