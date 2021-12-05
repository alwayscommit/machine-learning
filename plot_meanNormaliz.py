import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("Turmeric.csv")
label = "Turmeric Crop"
df = df.dropna()

# change the crop label and dataframe to be used
X_original = df[['Temperature', 'Rainfall']]
y = df[['Produce']]

# X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_original))

X1 = X_original.iloc[:, 0]
X2 = X_original.iloc[:, 1]
X = np.column_stack((X1, X2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

means = X_train.mean(axis=0)
ranges = X_train.max(axis=0) - X_train.min(axis=0)

X_train_scaled = (X_train - means) / ranges
X_test_scaled = (X_test - means) / ranges

# np.round(X_train.describe(), 1)
# np.round(X_train_scaled.describe(), 1)

model = RandomForestRegressor()
# ravel() is used for for random forest
model.fit(X_train_scaled, y_train.values.ravel())
# model.fit(X_train, y_train)
yPred = model.predict(X_test_scaled)

# yPred_individual = linear.predict([[]])

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

plt.scatter(X_train_scaled[:, 1], y_train, color='blue', marker='o', label=label)
plt.scatter(X_test_scaled[:, 1], yPred, color='yellow', marker='o', label="Predicted")
plt.xlabel("Rainfall")
plt.ylabel("Produce")
plt.legend(loc='center left', bbox_to_anchor=(-0.5, -0.3))
plt.show()

plt.scatter(X_train_scaled[:, 0], y_train, color='blue', marker='o', label=label)
plt.scatter(X_test_scaled[:, 0], yPred, color='yellow', marker='o', label="predicted")
plt.xlabel("Temperature")
plt.ylabel("Produce")
plt.legend(loc='center left', bbox_to_anchor=(-0.5, -0.3))
plt.show()

print(model.score(X_test_scaled, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, yPred))
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, yPred)))
print("r2 square: %.2f" % r2_score(y_test, yPred))

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train_scaled, y_train.values.ravel())
dummy_regr.predict(X_test_scaled)
print(dummy_regr.score(X_test_scaled, y_test))
