import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from feature_engine.outliers import Winsorizer

# All the scores mentioned below are for banana crop for now.

df_train = pd.read_csv("train_" + "Banana.csv")
df_test = pd.read_csv("test_" + "Banana.csv")
df_train = df_train.dropna()
df_test = df_test.dropna()
label = "Banana Crop"

X_train_original = df_train[['Temperature', 'Rainfall']]
y_train = df_train[['Produce']]

def diagnostic_plots(df, variable):
    # function takes a dataframe (df) and
    # the variable of interest as arguments

    # define figure size
    plt.figure(figsize=(16, 4))

    # histogram
    plt.subplot(1, 3, 1)
    sns.histplot(df[variable], bins=30)
    plt.title('Histogram')

    # Q-Q plot
    plt.subplot(1, 3, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.ylabel('Variable quantiles')

    # boxplot
    plt.subplot(1, 3, 3)
    sns.boxplot(y=df[variable])
    plt.title('Boxplot')


# diagnostic_plots(X_train_original, 'Temperature')
diagnostic_plots(X_train_original, 'Rainfall')

windsoriser = Winsorizer(capping_method='iqr', # choose iqr for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails
                          fold=1.5,
                          variables=['Temperature', 'Rainfall'])

windsoriser.fit(X_train_original)
X_train_original = windsoriser.transform(X_train_original)

# diagnostic_plots(X_train_original, 'Temperature')
diagnostic_plots(X_train_original, 'Rainfall')
plt.show()



