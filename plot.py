import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_rice = pd.read_csv("Rice.csv")
df_rice=df_rice.dropna()
df_potato = pd.read_csv("Potato.csv")
df_potato=df_potato.dropna()
df_banana = pd.read_csv("Banana.csv")
df_banana=df_banana.dropna()

X_rice = df_rice[['Temperature', 'Rainfall']]
X1_rice = df_rice.iloc[:, 3]
X2_rice = df_rice.iloc[:, 4]
y_rice = df_rice['Produce']

X = np.column_stack((X1_rice, X2_rice))

X_potato = df_potato[['Temperature', 'Rainfall']]
X1_potato = df_potato.iloc[:, 3]
X2_potato = df_potato.iloc[:, 4]
y_potato = df_potato['Produce']

X_banana = df_banana[['Temperature', 'Rainfall']]
X1_banana = df_banana.iloc[:, 3]
X2_banana = df_banana.iloc[:, 4]
y_banana = df_banana['Produce']

from sklearn import linear_model

linear = linear_model.LinearRegression()
# linear.fit(X_rice, y_rice)
# yPred_rice = linear.predict(X_rice)
# linear.fit(X_banana, y_banana)
# yPred_banana = linear.predict(X_banana)
linear.fit(X_potato, y_potato)
yPred_potato = linear.predict(X_potato)

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

# plt.scatter(X2_rice,y_rice, color='red', marker='o', label="Rice crop")
# plt.scatter(X2_rice,yPred_rice, color='yellow', marker='o', label="predicted")
# plt.scatter(X2_banana,y_banana, color='yellow', marker='o', label="Banana crop")
# plt.scatter(X2_banana,yPred_banana, color='blue', marker='o', label="predicted")
plt.scatter(X2_potato,y_potato, color='blue', marker='o', label="Potato crop")
plt.scatter(X2_potato,yPred_potato, color='yellow', marker='o', label="Predicted")
plt.xlabel("Rainfall")
plt.ylabel("produce")
plt.legend(loc='center left', bbox_to_anchor=(-0.5, -0.3))
plt.show()

# plt.scatter(X1_rice,y_rice, color='red', marker='o', label="Rice crop")
# plt.scatter(X1_rice,yPred_rice, color='yellow', marker='o', label="predicted")
# plt.scatter(X1_banana,y_banana, color='yellow', marker='o', label="Banana crop")
# plt.scatter(X1_banana,yPred_banana, color='blue', marker='o', label="predicted")
plt.scatter(X1_potato,y_potato, color='blue', marker='o', label="Potato crop")
plt.scatter(X1_potato,yPred_potato, color='yellow', marker='o', label="predicted")
plt.xlabel("Temperature")
plt.ylabel("produce")
plt.legend(loc='center left', bbox_to_anchor=(-0.5, -0.3))
plt.show()

from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f" % mean_squared_error(y_potato, yPred_potato))
print("r2 square: %.2f" % r2_score(y_potato, yPred_potato))

