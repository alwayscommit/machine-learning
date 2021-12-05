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

# df_rice = pd.read_csv("Rice.csv")
# df_rice = df_rice.dropna()
# df_potato = pd.read_csv("Potato.csv")
# df_potato = df_potato.dropna()
# df_banana = pd.read_csv("Banana.csv")
# df_banana = df_banana.dropna()

df = pd.read_csv("Turmeric.csv")
label = "Turmeric Crop"
df = df.dropna()

# X_rice = df_rice[['Temperature', 'Rainfall']]
# X1_rice = df_rice.iloc[:, 3]
# X2_rice = df_rice.iloc[:, 4]
# y_rice = df_rice['Produce']

# X_potato = df_potato[['Temperature', 'Rainfall']]
# X1_potato = df_potato.iloc[:, 3]
# X2_potato = df_potato.iloc[:, 4]
# y_potato = df_potato['Produce']

# X_banana = df_banana[['Temperature', 'Rainfall']]
# X1_banana = df_banana.iloc[:, 3]
# X2_banana = df_banana.iloc[:, 4]
# y_banana = df_banana['Produce']

# change the crop label and dataframe to be used
X_original = df[['Temperature', 'Rainfall']]
y = df[['Produce']]

X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_original))

X1 = X_scaled.iloc[:, 0]
X2 = X_scaled.iloc[:, 1]
# add new feature based on rainfall and temp together
X3 = X1*X2
X = np.column_stack((X1, X2,X3))

# model = linear_model.LinearRegression()
model = RandomForestRegressor()
# model = DecisionTreeRegressor(max_depth=5)
# linear.fit(X_rice, y_rice)
# yPred_rice = linear.predict(X_rice)
# linear.fit(X_banana, y_banana)
# yPred_banana = linear.predict(X_banana)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

#ravel() is used for for random forest
model.fit(X_train, y_train.values.ravel())
# model.fit(X_train, y_train)
yPred = model.predict(X_test)


# yPred_individual = linear.predict([[]])

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# plt.scatter(X2_rice,y_rice, color='red', marker='o', label="Rice crop")
# plt.scatter(X2_rice,yPred_rice, color='yellow', marker='o', label="predicted")
# plt.scatter(X2_banana,y_banana, color='yellow', marker='o', label="Banana crop")
# plt.scatter(X2_banana,yPred_banana, color='blue', marker='o', label="predicted")
plt.scatter(X_train[:, 1], y_train, color='blue', marker='o', label=label)
plt.scatter(X_test[:, 1], yPred, color='yellow', marker='o', label="Predicted")
plt.xlabel("Rainfall")
plt.ylabel("Produce")
plt.legend(loc='center left', bbox_to_anchor=(-0.5, -0.3))
plt.show()

# plt.scatter(X1_rice,y_rice, color='red', marker='o', label="Rice crop")
# plt.scatter(X1_rice,yPred_rice, color='yellow', marker='o', label="predicted")
# plt.scatter(X1_banana,y_banana, color='yellow', marker='o', label="Banana crop")
# plt.scatter(X1_banana,yPred_banana, color='blue', marker='o', label="predicted")
plt.scatter(X_train[:, 0], y_train, color='blue', marker='o', label=label)
plt.scatter(X_test[:, 0], yPred, color='yellow', marker='o', label="predicted")
plt.xlabel("Temperature")
plt.ylabel("Produce")
plt.legend(loc='center left', bbox_to_anchor=(-0.5, -0.3))
plt.show()

print(model.score(X_test, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, yPred))
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, yPred)))
print("r2 square: %.2f" % r2_score(y_test, yPred))

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train.values.ravel())
dummy_regr.predict(X_test)
print(dummy_regr.score(X_test, y_test))
