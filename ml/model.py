from sklearn.svm import SVR
from resources import *
from ml.preprocessing import preprocess

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def model_v1():
    features = ['OverallQual']
    X = train[features]
    y = train['SalePrice']
    reg = SVR(C=1000, gamma='auto')
    reg.fit(X, y)
    Xt = test[features]
    predicted = pd.DataFrame(reg.predict(Xt), columns=['SalePrice'])

    return predicted

def model_v2():
    data = preprocess()
    train_n_rows = train.shape[0]
    train_df = data.iloc[:train_n_rows, 1:]
    test_data = data.iloc[train_n_rows:, 1:]

    # Reattaching the target variable
    train_data = pd.concat([train_df, train.iloc[:, -1]], axis=1)

    X = train_data.iloc[:, :-1]
    y = train_data.iloc[:, -1]

    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.01, 0.1, 0.5, 2, 10], 'degree': [2, 3]}
    svr = SVR(gamma='auto', max_iter=100000)
    reg = GridSearchCV(svr, parameters, cv=5, verbose=True)
    reg.fit(X, y)
    predicted = pd.DataFrame(reg.predict(test_data), columns=['SalePrice'])
    print(reg.best_estimator_)

    return predicted
