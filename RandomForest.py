import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.dummy import DummyRegressor

df = pd.read_csv("crop_dataset/Wheat.csv")
X_original = df[['Temperature', 'Rainfall']]
y = df['Produce']

scale = MinMaxScaler().fit(X_original)
X_train_scaled = pd.DataFrame(scale.transform(X_original))

X1 = X_train_scaled.iloc[:, 0]
X2 = X_train_scaled.iloc[:, 1]
X3 = X1 * X2
X = np.column_stack((X1, X2, X3))

regr = RandomForestRegressor()
regr.fit(X, y)

featureNames = []
featureImportance = []
# Remove fields that we can't use as features
df.drop(['Year', 'District', 'Produce'], inplace=True, axis=1)
# get the importance of the different columns in the datafield
for f, n in zip(df, regr.feature_importances_):
    featureNames.append(f)
    featureImportance.append(n)

# plot the importance of the features
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.barh(featureNames, featureImportance)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(crop)
plt.show()

print("Mean squared error: %.2f" % -cross_val_score(regr, X, y, cv=5, scoring='neg_mean_squared_error').mean())
print("Root mean squared error: %.2f" % -cross_val_score(regr, X, y, cv=5,
                                                         scoring='neg_root_mean_squared_error').mean())
print("r2: %.2f" % cross_val_score(regr, X, y, cv=5, scoring='r2').mean())
