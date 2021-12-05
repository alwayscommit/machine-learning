import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.linear_model import Lasso

df_rice = pd.read_csv("Rice.csv")
df_rice = df_rice.dropna()
df_potato = pd.read_csv("Potato.csv")
df_potato = df_potato.dropna()
df_banana = pd.read_csv("Banana.csv")
df_banana = df_banana.dropna()

df = pd.read_csv("Turmeric.csv")
df = df.dropna()

# X_rice = df_rice[['Temperature', 'Rainfall']]
# X1_rice = df_rice.iloc[:, 3]
# X2_rice = df_rice.iloc[:, 4]
# y_rice = df_rice['Produce']

# X_potato = df_potato[['Temperature', 'Rainfall']]
# X1_potato = df_potato.iloc[:, 3]
# X2_potato = df_potato.iloc[:, 4]
# y_potato = df_potato['Produce']

# X_banana = df_banana[['Temperature', 'Rainfall']]
# X1_banana = df_banana.iloc[:, 3]
# X2_banana = df_banana.iloc[:, 4]
# y_banana = df_banana['Produce']

# change the crop label and dataframe to be used
label = "Turmeric Crop"
X_original = df[['Temperature', 'Rainfall']]
y = df[['Produce']]

X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X_original))

X1 = X_scaled.iloc[:, 0]
X2 = X_scaled.iloc[:, 1]
X = np.column_stack((X1, X2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

mean_error = []
std_error = []
Ci_range = [0.1, 1, 5, 10, 15, 50, 75, 100]
for Ci in Ci_range:
    temp = []
    model = Lasso(alpha=1 / (2 * Ci))
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    mean_error.append(mean_squared_error(y_test, ypred))
    std_error.append(np.array(mean_error).std())
    print("Mean squared error: %.2f" % mean_squared_error(y_test, ypred))
    print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, ypred)))
    print("r2 square: %.2f" % r2_score(y_test, ypred))

plt.errorbar(Ci_range, mean_error, yerr=std_error)
plt.xlabel('Ci')
plt.ylabel("Mean square error")
# plt.xlim(0, 250)
plt.show()

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
