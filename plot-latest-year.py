import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

df_train = pd.read_csv("train_" + "Banana.csv")
df_test = pd.read_csv("test_" + "Banana.csv")
label = "Banana Crop"
df_train = df_train.dropna()
df_test = df_test.dropna()

# change the crop label and dataframe to be used
X_train_original = df_train[['Temperature', 'Rainfall']]
y_train = df_train[['Produce']]

X_test_original = df_test[['Temperature', 'Rainfall']]
y_test = df_test[['Produce']]

#scaling doesn't improve the score much, commented for now.
# scale = MinMaxScaler().fit(X_train_original)
# X_train_scaled = pd.DataFrame(scale.transform(X_train_original))
# X_test_scaled = pd.DataFrame(scale.transform(X_test_original))

X1 = X_train_original.iloc[:, 0]
X2 = X_train_original.iloc[:, 1]
# add new feature based on rainfall and temp together
X3 = X1*X2
X_train = np.column_stack((X1, X2, X3))

X1_test = X_test_original.iloc[:, 0]
X2_test = X_test_original.iloc[:, 1]
# add new feature based on rainfall and temp together
X3_test = X1_test*X2_test
X_test = np.column_stack((X1_test, X2_test, X3_test))

# model = linear_model.LinearRegression()
model = RandomForestRegressor()
# model = DecisionTreeRegressor(max_depth=5)

#ravel() is used for for random forest
model.fit(X_train, y_train.values.ravel())
# model.fit(X_train, y_train)
yPred = model.predict(X_test)

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

plt.scatter(X_train[:, 1], y_train, color='blue', marker='o', label=label)
plt.scatter(X_test[:, 1], yPred, color='yellow', marker='o', label="Predicted")
plt.xlabel("Rainfall")
plt.ylabel("Produce")
plt.legend(loc='center left', bbox_to_anchor=(-0.5, -0.3))
plt.show()

plt.scatter(X_train[:, 0], y_train, color='blue', marker='o', label=label)
plt.scatter(X_test[:, 0], yPred, color='yellow', marker='o', label="predicted")
plt.xlabel("Temperature")
plt.ylabel("Produce")
plt.legend(loc='center left', bbox_to_anchor=(-0.5, -0.3))
plt.show()

#scores
print(model.score(X_test, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, yPred))
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, yPred)))
print("r2 square: %.2f" % r2_score(y_test, yPred))

#dummy
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train.values.ravel())
dummy_regr.predict(X_test)
print(dummy_regr.score(X_test, y_test))
