import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

# Change the crop label and dataframe to be used
label = ["Cotton(lint)", "Masoor", "Rice", "Wheat", "Sunflower"]
df = pd.read_csv("crop_dataset/" + label[4] + ".csv")
df = df.dropna()

x = df[['Temperature', 'Rainfall']]
X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(x))
X1 = X_scaled.iloc[:, 0]
X2 = X_scaled.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df['Produce']

std_error = []
std_errorCV = []
rme = []
se = []
rmeCV = []
seCV = []
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
        # std_error.append(np.array(temp).std())

        # tempCV.append(
        #     -cross_val_score(model, X[test], ypred, cv=5, scoring='neg_root_mean_squared_error').mean())
        tempCV.append(
            -cross_val_score(model, X[test], ypred, cv=5, scoring='neg_root_mean_squared_error'))
        # std_errorCV.append(np.array(tempCV).std())

        r2.append(r2_score(y[test], ypred))
        r2CV.append(cross_val_score(model, X[test], ypred, cv=5, scoring='r2'))

    print("C : ", Ci)

    print("Root mean squared error: ", np.array(temp).mean())
    print("r2 square: ", np.array(r2).mean())
    print("Cross val scores:-")
    rme.append(np.array(temp).mean())
    std_error.append(np.array(rme).std())

    print("Root mean squared error: ", np.array(tempCV).mean())
    print("r2: ", np.array(r2CV).mean())
    rmeCV.append(np.array(tempCV).mean())
    std_errorCV.append(np.array(rmeCV).std())

plt.errorbar(Ci_range, rme, yerr=std_error)
plt.title("Normal Root Mean Square Error")
plt.xlabel('Ci')
plt.ylabel("RMSE")
# plt.xlim(0, 250)
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.show()

plt.errorbar(Ci_range, rmeCV, yerr=std_errorCV)
plt.title("Cross Validation Neg Root Mean Square Error")
plt.xlabel('Ci')
plt.ylabel("RMSE")
# plt.xlim(0, 250)
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
plt.show()

# Compare Against Dummy Classifier
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(Xtrain, ytrain)
dummy_pred = dummy_regr.predict(Xtest)

model = Lasso(alpha=1 / (2 * 100))  # best c value
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("Dummy - Root mean squared error: %.2f" % sqrt(mean_squared_error(ytest, dummy_pred)))
print("Dummy - r2 square: %.2f" % r2_score(ytest, dummy_pred))

print("Dummy - Cross val -  Root mean squared error: %.2f"
      % -cross_val_score(model, Xtest, dummy_pred, cv=5,
                         scoring='neg_root_mean_squared_error').mean())
print("Dummy - Cross val - r2: %.2f" % cross_val_score(model, Xtest, dummy_pred, cv=5, scoring='r2').mean())
