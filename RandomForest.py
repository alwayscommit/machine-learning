import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.dummy import DummyRegressor

df = pd.read_csv("crop_dataset/Wheat.csv")
X_original = df[['Temperature', 'Rainfall']]
y = df['Produce']

X1 = X_original.iloc[:, 0]
X2 = X_original.iloc[:, 1]
X3 = X1 * X2
X = np.column_stack((X1, X2, X3))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

regr = RandomForestRegressor()
regr.fit(Xtrain, ytrain)
yPred = regr.predict(Xtest)
print("Mean squared error: %.2f" % -cross_val_score(regr, Xtest, yPred, cv=5, scoring='neg_mean_squared_error').mean())
print("Root mean squared error: %.2f" % -cross_val_score(regr, Xtest, yPred, cv=5,
                                                         scoring='neg_root_mean_squared_error').mean())
print("r2: %.2f" % cross_val_score(regr, Xtest, yPred, cv=5, scoring='r2').mean())

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(Xtrain, ytrain.values.ravel())
dummy_regr.predict(Xtest)

print("Dummy - Mean squared error: %.2f" % -cross_val_score(dummy_regr, Xtest, yPred, cv=5, scoring='neg_mean_squared_error').mean())
print("Dummy - Root mean squared error: %.2f" % -cross_val_score(dummy_regr, Xtest, yPred, cv=5,
                                                         scoring='neg_root_mean_squared_error').mean())
print("Dummy - r2: %.2f" % cross_val_score(dummy_regr, Xtest, yPred, cv=5, scoring='r2').mean())
