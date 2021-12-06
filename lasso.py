import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.linear_model import Lasso

label = ["Cotton(lint)", "Masoor", "Rice", "Wheat", "Sunflower"]
df_train = pd.read_csv("crop_datasets/train_" + label[0] + ".csv")
df_test = pd.read_csv("crop_datasets/test_" + label[0] + ".csv")
df_train = df_train.dropna()
df_test = df_test.dropna()

X_train_original = df_train[['Temperature', 'Rainfall']]
y = df_train[['Produce']]

X_test_original = df_test[['Temperature', 'Rainfall']]
y_test = df_test[['Produce']]

X_train_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_train_original))
X1_train = X_train_scaled.iloc[:, 0]
X2_train = X_train_scaled.iloc[:, 1]
X = np.column_stack((X1_train, X2_train))

X_test_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_test_original))
X1_test = X_test_scaled.iloc[:, 0]
X2_test = X_test_scaled.iloc[:, 1]
X_test = np.column_stack((X1_test, X2_test))

std_error = []
std_errorCV = []
rme = []
rmeCV = []
Ci_range = [0.1, 1, 5, 10, 15, 50, 75, 100]
for Ci in Ci_range:
    temp = []
    tempCV = []
    r2 = []
    r2CV = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = Lasso(alpha=1 / (2 * Ci))
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])

        temp.append(sqrt(mean_squared_error(y[test], ypred)))
        # tempCV.append(
        #     -cross_val_score(model, X[test], ypred, cv=5, scoring='neg_root_mean_squared_error'))

        r2.append(r2_score(y[test], ypred))
        # r2CV.append(cross_val_score(model, X[test], ypred, cv=5, scoring='r2'))

    print("C : ", Ci)

    print("Root mean squared error: ", np.array(temp).mean())
    print("r2 square: ", np.array(r2).mean())
    print("Cross val scores:-")
    rme.append(np.array(temp).mean())
    std_error.append(np.array(rme).std())

    # print("Root mean squared error: ", np.array(tempCV).mean())
    # print("r2: ", np.array(r2CV).mean())
    # rmeCV.append(np.array(tempCV).mean())
    # std_errorCV.append(np.array(rmeCV).std())

plt.errorbar(Ci_range, rme, yerr=std_error)
plt.title("Normal Root Mean Square Error")
plt.xlabel('Ci')
plt.ylabel("RMSE")
# plt.xlim(0, 250)
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.show()

# plt.errorbar(Ci_range, rmeCV, yerr=std_errorCV)
# plt.title("Cross Validation Neg Root Mean Square Error")
# plt.xlabel('Ci')
# plt.ylabel("RMSE")
# # plt.xlim(0, 250)
# plt.rc('font', size=18)
# plt.rcParams['figure.constrained_layout.use'] = True
# plt.show()

# Compare Against Dummy Classifier
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X, y)
dummy_pred = dummy_regr.predict(X_test)

model = Lasso(alpha=1 / (2 * 100))  # best c value
model.fit(X, y)
ypred = model.predict(X_test)

print("Dummy - Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, dummy_pred)))
print("Dummy - r2 square: %.2f" % r2_score(y_test, dummy_pred))

# print("Dummy - Cross val -  Root mean squared error: %.2f"
#       % -cross_val_score(model, X_test, dummy_pred, cv=5,
#                          scoring='neg_root_mean_squared_error').mean())
# print("Dummy - Cross val - r2: %.2f" % cross_val_score(model, X_test, dummy_pred, cv=5, scoring='r2').mean())


# Plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title(label[0])
ax.scatter(X[:, 0], X[:, 1], y, color='black', label="Features")
ax.set_xlabel("Temperature")
ax.set_ylabel("Rainfall")
ax.set_zlabel("Produce")
predicted_val = ax.plot_trisurf(X_test[:, 0], X_test[:, 1], ypred, color='red', label="Predictions")
predicted_val._facecolors2d = predicted_val._facecolor3d
predicted_val._edgecolors2d = predicted_val._edgecolor3d
plt.legend(loc="best")
plt.show()
