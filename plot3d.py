import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("Rice.csv")
label = "Rice Crop"
df = df.dropna()

X_original = df[['Temperature', 'Rainfall']]
y = df[['Produce']]

X1 = X_original.iloc[:, 0]
X2 = X_original.iloc[:, 1]
X = np.column_stack((X1, X2))

model = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model.fit(X_train, y_train.values.ravel())
yPred = model.predict(X_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(label)
ax.scatter(X1, X2, y, color='black', label="Features")
ax.set_xlabel("Temperature")
ax.set_ylabel("Rainfall")
ax.set_zlabel("Produce")
predicted_val = ax.plot_trisurf(X_test[:, 0], X_test[:, 1], yPred, color='red', label="Predictions")
predicted_val._facecolors2d = predicted_val._facecolor3d
predicted_val._edgecolors2d = predicted_val._edgecolor3d
plt.legend(loc="best")
plt.show()
