import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

crop = 'Sunflower'
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

input_size = 3
output_size = 1
model = tf.keras.Sequential([tf.keras.layers.Dense(output_size)])
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(optimizer=Adam(lr=1e-3, decay=1e-3 / 200), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=500, verbose=1)
yPred = model.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, yPred))
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(y_test, yPred)))
print("r2 square: %.2f" % r2_score(y_test, yPred))
