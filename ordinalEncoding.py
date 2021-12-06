from math import sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

crop = 'Cotton(lint)'
df = pd.read_csv("crop_dataset/" + crop + ".csv")

ordinal_encoder = OrdinalEncoder()
# encoded_seasons_df_train = pd.DataFrame(ordinal_encoder.fit_transform(df[['Season']]), columns=['SeasonEncoded'])
encoded_seasons_df_train = pd.DataFrame(ordinal_encoder.fit_transform(df[['Season', 'State']]),
                                        columns=['SeasonEncoded', 'StateEncoding'])
# join the encoded Seasons dataframe to the original training dataframe
encoded_train_df = df.join(encoded_seasons_df_train)

X_original = encoded_train_df[['Temperature','Rainfall', 'SeasonEncoded', 'StateEncoding']]
# X_original = encoded_train_df[['Temperature', 'SeasonEncoded', 'StateEncoding']]
# X_original = encoded_train_df[['Temperature', 'StateEncoding']]
y = encoded_train_df['Produce']

# scaling doesn't improve the score much, commented for now.
scale = MinMaxScaler().fit(X_original)
X_train_scaled = pd.DataFrame(scale.transform(X_original))

X1 = X_train_scaled.iloc[:, 0]
X2 = X_train_scaled.iloc[:, 1]
# X3 = X1 * X2
X4 = X_train_scaled.iloc[:, 2]
X5 = X_train_scaled.iloc[:, 3]
# X = np.column_stack((X1, X2, X3, X4, X5))
# X = np.column_stack((X1, X2, X4))
X = np.column_stack((X1, X2, X4,X5))
# X = np.column_stack((X1, X2))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# getting the best n values

kf = KFold(n_splits=5)
rmse_n = []
r2_n = []
est_range = [10, 50, 100, 500, 1000]
for i in est_range:
    temp = []
    temp2 = []
    for train, test in kf.split(Xtrain):
        model = RandomForestRegressor(n_estimators=i, random_state=0, n_jobs=-1)
        model.fit(Xtrain[train], y[train])
        yPred = model.predict(Xtrain[test])
        temp.append(cross_val_score(model, Xtrain, ytrain, cv=5, scoring='neg_root_mean_squared_error'))
        temp2.append(cross_val_score(model, Xtrain, ytrain, cv=5, scoring='r2'))
    rmse_n.append(np.array(temp).mean())
    r2_n.append(np.array(temp).mean())

rmse_n = [x * -1 for x in rmse_n]
nMin = min(rmse_n)
minIndex = rmse_n.index(nMin)
nVal = est_range[minIndex]  # this is needed only if we are passing the min
# plot rmse against N values
plt.plot(est_range, rmse_n, color='green')
plt.title(crop + ' RMSE VS. N Value')
plt.xlabel('N')
plt.ylabel('RMSE')
plt.show()

plt.plot(est_range, r2_n, color='green')
plt.title(crop + ' R-Square VS. N Value')
plt.xlabel('N')
plt.ylabel('R-Square')
plt.show()
#
print(nVal)
regr = RandomForestRegressor(n_estimators=nVal, random_state=0, n_jobs=-1)
regr.fit(Xtrain, ytrain)
yPred = regr.predict(Xtest)

print(regr.score(Xtest, ytest))
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(ytest, yPred)))
print("r2 square: %.2f" % r2_score(ytest, yPred))

print("Cross val - Mean squared error: %.2f" % -cross_val_score(regr, X, y, cv=5,
                                                                scoring='neg_mean_squared_error').mean())
print("Cross val - Root mean squared error: %.2f" % -cross_val_score(regr, X, y, cv=5,
                                                                     scoring='neg_root_mean_squared_error').mean())
print("Cross val - r2: %.2f" % cross_val_score(regr, X, y, cv=5, scoring='r2').mean())

# dummy
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(Xtrain, ytrain)
dummy_pred = dummy_regr.predict(Xtest)
print(dummy_regr.score(Xtest, ytest))

print("Dummy - Root mean squared error: %.2f" % sqrt(mean_squared_error(ytest, dummy_pred)))
print("Dummy - r2 square: %.2f" % r2_score(ytest, dummy_pred))

print("Dummy -Cross val -  Mean squared error: %.2f" % -cross_val_score(dummy_regr, X, y, cv=5,
                                                                        scoring='neg_mean_squared_error').mean())
print("Dummy - Cross val -  Root mean squared error: %.2f" % -cross_val_score(dummy_regr, X, y, cv=5,
                                                                              scoring='neg_root_mean_squared_error').mean())
print("Dummy - Cross val - r2: %.2f" % cross_val_score(dummy_regr, X, y, cv=5, scoring='r2').mean())
