import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
import tensorflow as tf
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("crop_dataset/Sunflower.csv")

X = df[['Temperature', 'Rainfall']]
y = df['Produce']

y = y.values.reshape(-1, 1)

X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

X1 = X[:, 0]
X2 = X[:, 1]
X3 = X1 * X2
X = np.column_stack((X1, X2, X3))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

input_size = 3
output_size = 1
model = tf.keras.Sequential([tf.keras.layers.Dense(output_size)])
model.add(Dense(128, activation="relu", input_dim=3))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(optimizer=Adam(lr=1e-3, decay=1e-3 / 200), loss='mean_squared_error')
history = model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=500, batch_size=32, verbose=2)

history_dict = history.history

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.figure()
plt.plot(loss_values, 'b', label="training loss")
plt.plot(val_loss_values, 'r', label="validation loss")
plt.legend(loc="upper right")
plt.xlabel("epochs")
plt.ylabel("error")
plt.title("Sunflower")
plt.show()

yPred = model.predict(Xtest)
print("Root mean squared error: %.2f" % sqrt(mean_squared_error(ytest, yPred)))
print("r2 square: %.2f" % r2_score(ytest, yPred))

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(Xtrain, ytrain)
dummy_pred = dummy_regr.predict(Xtest)
print("Dummy - Root mean squared error: %.2f" % sqrt(mean_squared_error(ytest, dummy_pred)))
print("Dummy - r2 square: %.2f" % r2_score(ytest, dummy_pred))



