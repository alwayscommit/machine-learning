import pandas as pd

#This file is responsible for cleaning up the rainfall dataset
rainfall_df = pd.read_csv('rainfall2.csv', index_col=False)

#DATE formatting to extract out YEAR and MONTH as separate columns required to be merged with crop dataset
rainfall_df['DATE'] = pd.to_datetime(rainfall_df['DATE'], errors='coerce')
rainfall_df["YEAR"] = rainfall_df['DATE'].dt.year
rainfall_df["MONTH"] = rainfall_df['DATE'].dt.month

#Reduces the size of the dataset by removing columns that aren't required.
rainfall_df.drop(columns=['DATE', 'VARIABLE_NOTES', 'FREQUENCY'], inplace=True)

#Removes all rows that have Departures of rainfall (%) as they're not required
rainfall_df = rainfall_df[~rainfall_df.VARIABLE_NAME.str.contains("%")]

#Save to a new file
rainfall_df.to_csv("rainfall-processed.csv", index=False, header=True)

print("Rainfall data cleaned.")