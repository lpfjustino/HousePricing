import pandas as pd
from sklearn.svm import SVR

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

features = ['OverallQual']
X = train[features]
y = train['SalePrice']
clf = SVR(C=1000)
clf.fit(X, y)

Xt = test[features]
print(clf.predict(Xt)[:5])
print(y[:5])
