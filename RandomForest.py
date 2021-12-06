import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

crop='Cotton(lint)'
df_test = pd.read_csv("test_"+crop+".csv")
df_train = pd.read_csv("train_"+crop+".csv")

X_test = df_test[['Temperature', 'Rainfall']]
y_test = df_test['Produce']

X_train = df_train[['Temperature', 'Rainfall']]
y_train = df_train['Produce']


X1 = X_train.iloc[:, 0]
X2 = X_train.iloc[:, 1]
X3=X1*X2
X_train = np.column_stack((X1, X2,X3))

X1_test = X_test.iloc[:, 0]
X2_test= X_test.iloc[:, 1]
X3_test=X1_test*X2_test
X_test = np.column_stack((X1_test, X2_test,X3_test))


regr = RandomForestRegressor()
regr.fit(X_train, y_train)
yPred = regr.predict(X_test)
print("Mean squared error: %.2f" % -cross_val_score(regr, X_test, yPred, cv=5, scoring='neg_mean_squared_error').mean())
print("Root mean squared error: %.2f" % -cross_val_score(regr, X_test, yPred, cv=5, scoring='neg_root_mean_squared_error').mean())
print("r2: %.2f" % cross_val_score(regr, X_test, yPred, cv=5, scoring='r2').mean())