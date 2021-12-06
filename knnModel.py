import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import statistics
import matplotlib.pyplot as plt

df = pd.read_csv("./crop_dataset/Banana.csv")

# change the crop label and dataframe to be used
label = "Banana Crop"
X_original = df[['Temperature', 'Rainfall']]
y = df[['Produce']]

# X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_original))

X1 = X_original.iloc[:, 0]
X2 = X_original.iloc[:, 1]
X = np.column_stack((X1, X2))
y = y.astype(int)

# split the data into train test
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

mean_list = []
stdev_list = []

# 0.01 to 100 because it was observed that the feature penalties stabilize after 100
k_neighbours_list = np.arange(1, 25, 5)
for k in k_neighbours_list:
    k_fold = KFold(n_splits=5)
    mean_squared_error_list = []
    for train_index, test_index in k_fold.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train.values.ravel())
        y_pred = knn_model.predict(X_test)

        mean_squared_error_list.append(mean_squared_error(y_test, y_pred))
    mean_list.append(statistics.mean(mean_squared_error_list))
    stdev_list.append(statistics.stdev(mean_squared_error_list))

#Ridge MSE with K-fold 5
fig = plt.figure()
ax = fig.add_subplot(111)
#x has C values, y has mean
ax.errorbar(k_neighbours_list, mean_list, color='blue', label='Mean')
#x has C values, y has standard deviation
ax.errorbar(k_neighbours_list, stdev_list, color='red', label='Standard Deviation')
ax.set_ylabel("mean & standard deviation")
ax.set_xlabel("k")
ax.set_title("5 Folds with KNN")
ax.legend(loc='upper right')

# print("kNN")
# print(confusion_matrix(ytest, k_pred))

# print("Mean squared error: %.2f" % mean_squared_error(ytest, k_pred))
# print("Root mean squared error: %.2f" % sqrt(mean_squared_error(ytest, k_pred)))
# print("r2 square: %.2f" % r2_score(ytest, k_pred))

plt.show()