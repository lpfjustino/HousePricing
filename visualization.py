import pandas as pd
import seaborn as sns

df_train = pd.read_csv('../input/train.csv')

# Plotting the histogram
sns.distplot(df_train['SalePrice']);