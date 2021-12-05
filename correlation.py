import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# pass the name of the crop
crop = "Orange"
df = pd.read_csv(crop +'.csv')
# drop the year feature since it is not relevant to our prediction model
df.drop(['Year'], axis=1, inplace=True)

# get the correlation using pearson algorithm
corrDF = df.corr(method='pearson')

# plot the correlation matrix
sns.heatmap(corrDF, cmap="YlGnBu")
plt.title(crop)
plt.show()