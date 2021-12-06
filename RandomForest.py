import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

crop = 'Rice'
df_test = pd.read_csv("crop_datasets/test_" + crop + ".csv")
df_train = pd.read_csv("crop_datasets/train_" + crop + ".csv")

X_test = df_test[['Temperature', 'Rainfall']]
y_test = df_test['Produce']

X_train = df_train[['Temperature', 'Rainfall']]
y_train = df_train['Produce']

X1 = X_train.iloc[:, 0]
X2 = X_train.iloc[:, 1]
X3 = X1 * X2
X_train = np.column_stack((X1, X2, X3))

X1_test = X_test.iloc[:, 0]
X2_test = X_test.iloc[:, 1]
X3_test = X1_test * X2_test
X_test = np.column_stack((X1_test, X2_test, X3_test))

regr = RandomForestRegressor()
regr.fit(X_train, y_train)
yPred = regr.predict(X_test)

featureNames=[]
featureImportance=[]
# get the importance of the different columns in the datafield
df_train.drop(['Year','District','Produce'],inplace = True, axis = 1)
print(df_train)
for f, n in zip(df_train, regr.feature_importances_):
    print(f)
    print(n)
    featureNames.append(f)
    featureImportance.append(n)


plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.barh(featureNames, featureImportance)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(crop)
plt.show()

print("Mean squared error: %.2f" % -cross_val_score(regr, X_test, yPred, cv=2, scoring='neg_mean_squared_error').mean())
print("Root mean squared error: %.2f" % -cross_val_score(regr, X_test, yPred, cv=2,
                                                         scoring='neg_root_mean_squared_error').mean())
print("r2: %.2f" % cross_val_score(regr, X_test, yPred, cv=2, scoring='r2').mean())
