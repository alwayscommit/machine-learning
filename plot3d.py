import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

df = pd.read_csv("Banana.csv")
label = "Banana Crop"
df = df.dropna()

X_original = df[['Temperature', 'Rainfall']]
y = df[['Produce']]

X1 = X_original.iloc[:, 0]
X2 = X_original.iloc[:, 1]

# Original features with RandomForestRegressor
# X = np.column_stack((X1, X2))
# model = RandomForestRegressor()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# model.fit(X_train, y_train.values.ravel())
# yPred = model.predict(X_test)

# Polynomial features with RandomForestRegressor
# X3 = X1 * X2
# X = np.column_stack((X1, X2, X3))
# model = RandomForestRegressor()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# model.fit(X_train, y_train.values.ravel())
# yPred = model.predict(X_test)

# Original features with knn
# X = np.column_stack((X1, X2))
# yKnn = y.astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, yKnn, test_size=0.2)
# kmodel = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train.values.ravel())
# yPred = kmodel.predict(X_test)

# Polynomial features with knn
# X3 = X1 * X2
# X = np.column_stack((X1, X2, X3))
# yKnn = y.astype(int)
# X_train, X_test, y_train, y_test = train_test_split(X, yKnn, test_size=0.2)
# kmodel = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train.values.ravel())
# yPred = kmodel.predict(X_test)

# Original features with Lasso (Bad)
# X = np.column_stack((X1, X2))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# model = Lasso(alpha=1 / (2 * 50))  # C=50
# model.fit(X_train, y_train)
# yPred = model.predict(X_test)

# Original features with Lasso (Still Bad)
# X3 = X1 * X2
# X = np.column_stack((X1, X2, X3))
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# model = Lasso(alpha=1 / (2 * 50))  # C=50
# model.fit(X_train, y_train)
# yPred = model.predict(X_test)

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
