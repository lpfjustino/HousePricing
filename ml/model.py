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
predicted = pd.DataFrame(clf.predict(Xt), columns=['SalePrice'])

output = pd.concat([test['Id'], predicted], axis=1)
output.to_csv(r'C:\Users\lpfjustino\Desktop\output.csv', index=None, header=True)
