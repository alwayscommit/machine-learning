from math import sqrt

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

crop='Wheat'
df_test = pd.read_csv("crop_datasets/test_"+crop+".csv")
df_train = pd.read_csv("crop_datasets/train_"+crop+".csv")

ordinal_encoder = OrdinalEncoder()
# encoded_seasons_df_train = pd.DataFrame(ordinal_encoder.fit_transform(df_train[['State']]), columns=['StateEnc'])
# encoded_seasons_df_train = pd.DataFrame(ordinal_encoder.fit_transform(df_train[['Season', 'State']]), columns=['SeasonEncoded','StateEnc'])
encoded_seasons_df_train = pd.DataFrame(ordinal_encoder.fit_transform(df_train[['Season']]), columns=['SeasonEncoded'])
# join the encoded Seasons dataframe to the original training dataframe
encoded_train_df = df_train.join(encoded_seasons_df_train)

# encoded_seasons_df_test = pd.DataFrame(ordinal_encoder.fit_transform(df_test[['State']]), columns=['StateEnc'])
# encoded_seasons_df_test = pd.DataFrame(ordinal_encoder.fit_transform(df_test[['Season', 'State']]), columns=['SeasonEncoded','StateEnc'])
encoded_seasons_df_test = pd.DataFrame(ordinal_encoder.fit_transform(df_test[['Season']]), columns=['SeasonEncoded'])
encoded_test_df = df_test.join(encoded_seasons_df_train)

X_original = encoded_train_df[['Temperature', 'Rainfall', 'SeasonEncoded']]
# X_original = encoded_train_df[['Temperature', 'Rainfall', 'StateEnc']]
# X_original = encoded_train_df[['Temperature', 'Rainfall', 'SeasonEncoded','StateEnc']]
y = encoded_train_df['Produce']

X_original_test = encoded_test_df[['Temperature', 'Rainfall', 'SeasonEncoded']]
# X_original_test = encoded_test_df[['Temperature', 'Rainfall','StateEnc']]
# X_original_test = encoded_test_df[['Temperature', 'Rainfall', 'SeasonEncoded','StateEnc']]
y_test = encoded_test_df['Produce']

#scaling doesn't improve the score much, commented for now.
scale = MinMaxScaler().fit(X_original)
X_train_scaled = pd.DataFrame(scale.transform(X_original))
X_test_scaled = pd.DataFrame(scale.transform(X_original_test))

X1 = X_train_scaled.iloc[:, 0]
X2 = X_train_scaled.iloc[:, 1]
X3 = X1 * X2
X4 = X_train_scaled.iloc[:, 2]
# X5 = X_train_scaled.iloc[:, 3]
# X_train = np.column_stack((X1, X2, X3, X4,X5))
X_train = np.column_stack((X1, X2, X3, X4))

X1_test = X_test_scaled.iloc[:, 0]
X2_test = X_test_scaled.iloc[:, 1]
X3_test = X1_test * X2_test
X4_test = X_test_scaled.iloc[:, 2]
# X5_test = X_test_scaled.iloc[:, 3]
# X_test = np.column_stack((X1_test, X2_test, X3_test, X4_test, X5_test))
X_test = np.column_stack((X1_test, X2_test, X3_test, X4_test))

regr = RandomForestRegressor()
regr.fit(X_train, y)
yPred = regr.predict(X_test)

featureNames=[]
featureImportance=[]
# Remove fields that we can't use as features
df_train.drop(['Year','District','Produce'],inplace = True, axis = 1)
# get the importance of the different columns in the datafield
for f, n in zip(df_train, regr.feature_importances_):
    featureNames.append(f)
    featureImportance.append(n)

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.barh(featureNames, featureImportance)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title(crop)
plt.show()

print(regr.score(X_test, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, yPred))
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, yPred)))
print("r2 square: %.2f" % r2_score(y_test, yPred))
