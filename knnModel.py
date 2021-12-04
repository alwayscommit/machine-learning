import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
df_rice = pd.read_csv("Rice.csv")


# change the crop label and dataframe to be used
label = "Rice Crop"
X_original = df_rice[['Temperature', 'Rainfall']]
yF = df_rice[['Produce']]
df_rice.reset_index(drop=True, inplace=True)
print(X_original)
# X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_original))

X1 = X_original.iloc[:, 0]
X2 = X_original.iloc[:, 1]
y = yF.astype(int)
X = np.column_stack((X1, X2))

# split the data into train test
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

# KVal-NN model
kmodel = KNeighborsClassifier(n_neighbors=20).fit(Xtrain, ytrain.values.ravel())
k_pred = kmodel.predict(Xtest)
k_predX = kmodel.predict(X)
print("kNN")
print(confusion_matrix(ytest, k_pred))


print("Mean squared error: %.2f" % mean_squared_error(ytest, k_pred))
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(ytest, k_pred)))
print("r2 square: %.2f" % r2_score(ytest, k_pred))
