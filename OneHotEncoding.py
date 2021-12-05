from math import sqrt

import numpy as np
import pandas as pd
from skimage.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

df_test = pd.read_csv("test_Banana.csv")
df_train = pd.read_csv("train_Banana.csv")

oneHotEncoder = OneHotEncoder(handle_unknown='ignore')
# renaming the encoded columns "Season"+ their values(feature names)
encoded_seasons_df_train = pd.DataFrame(oneHotEncoder.fit_transform(df_train[['Season']]).toarray(), columns=oneHotEncoder.get_feature_names(['Season']))
# join the encoded Seasons dataframe to the original training dataframe
encoded_train_df = df_train.join(encoded_seasons_df_train)

# renaming the encoded columns "Season"+ their values(feature names)
enc_df_test = pd.DataFrame(oneHotEncoder.fit_transform(df_test[['Season']]).toarray(), columns=oneHotEncoder.get_feature_names(['Season']))
encoded_test_df = df_test.join(encoded_seasons_df_train)

# in case you want to see the result
# encoded_train_df.to_csv("train_Potato_season" + ".csv", index=False)
# encoded_test_df.to_csv("test_Potato_season" + ".csv", index=False)

X_original = encoded_train_df[['Temperature', 'Rainfall', 'Season_Kharif', 'Season_Rabi', 'Season_Whole Year']]
y = encoded_train_df['Produce']

X_original_test = encoded_test_df[['Temperature', 'Rainfall', 'Season_Kharif', 'Season_Rabi', 'Season_Whole Year']]
y_test = encoded_test_df['Produce']

#scaling doesn't improve the score much, commented for now.
# scale = MinMaxScaler().fit(X_original)
# X_train_scaled = pd.DataFrame(scale.transform(X_original))
# X_test_scaled = pd.DataFrame(scale.transform(X_original_test))

X1 = X_original.iloc[:, 0]
X2 = X_original.iloc[:, 1]
X4 = X_original.iloc[:, 2]
X5 = X_original.iloc[:, 3]
X6 = X_original.iloc[:, 4]
X3 = X1 * X2
X_train = np.column_stack((X1, X2, X3, X4, X5, X6))

X1_test = X_original_test.iloc[:, 0]
X2_test = X_original_test.iloc[:, 1]
X4_test = X_original_test.iloc[:, 2]
X5_test = X_original_test.iloc[:, 3]
X6_test = X_original_test.iloc[:, 4]
X3_test = X1_test * X2_test
X_test = np.column_stack((X1_test, X2_test, X3_test, X4_test, X5_test, X6_test))

regr = RandomForestRegressor()
regr.fit(X_train, y)
yPred = regr.predict(X_test)

print(regr.score(X_test, y_test))
print("Mean squared error: %.2f" % mean_squared_error(y_test, yPred))
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, yPred)))
print("r2 square: %.2f" % r2_score(y_test, yPred))
