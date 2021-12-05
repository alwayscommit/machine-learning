import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

# df = pd.read_csv("Papaya.csv")
df = pd.read_csv("Potato.csv")

X_original = df[['Temperature', 'Rainfall']]
y = df['Produce']

X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_original))

X1 = X_scaled.iloc[:, 0]
X2 = X_scaled.iloc[:, 1]
X3 = X1 * X2
X = np.column_stack((X1, X2, X3))

# if used you would need to pass this value to the model instead of X
# Xpoly = PolynomialFeatures(3).fit_transform(X)

regr = RandomForestRegressor()
regr.fit(X, y)
yPred = regr.predict(X)

score = cross_validate(
    regr, X, y, scoring=["r2", "neg_mean_absolute_error"], n_jobs=-1, verbose=0
)

print(regr.score(X, y))
print("Mean squared error: %.2f" % -np.mean(score["test_neg_mean_absolute_error"]))
print("mse std : %.2f" % np.std(score["test_neg_mean_absolute_error"]))
print("r2 square: %.2f" % np.mean(score["test_r2"]))
print("r2 std : %.2f" % np.std(score["test_r2"]))

print("r2 square using r2_score: %.2f" % r2_score(y, yPred))
