from sklearn.svm import SVR
from resources import *

def model_v1():
    features = ['OverallQual']
    X = train[features]
    y = train['SalePrice']
    clf = SVR(C=1000)
    clf.fit(X, y)
    Xt = test[features]
    predicted = pd.DataFrame(clf.predict(Xt), columns=['SalePrice'])

    return predicted
