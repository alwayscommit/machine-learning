import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("crop_dataset/Potato.csv")

X = df[['Temperature', 'Rainfall']]
y = df['Produce']

y = y.values.reshape(-1, 1)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

X1 = X[:, 0]
X2 = X[:, 1]
X3 = X1 * X2
X = np.column_stack((X1, X2, X3))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

svr_poly = SVR(kernel="poly", C=0.001, degree=10)
svr_poly.fit(Xtrain, ytrain)
yPred = svr_poly.predict(Xtest)

print("Root mean squared error: %.2f" % -cross_val_score(svr_poly, Xtest, ytest, cv=5,
                                                         scoring='neg_root_mean_squared_error').mean())
print("r2: %.2f" % cross_val_score(svr_poly, ytest, yPred, cv=5, scoring='r2').mean())

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(Xtrain, ytrain)
dummy_pred = dummy_regr.predict(Xtest)

print("Dummy - Root mean squared error: %.2f" % sqrt(mean_squared_error(ytest, dummy_pred)))
print("Dummy - r2 square: %.2f" % r2_score(ytest, dummy_pred))

