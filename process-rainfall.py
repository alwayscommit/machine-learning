import pandas as pd

rainfall_df = pd.read_csv('rainfall2.csv', index_col=False)

rainfall_df['DATE'] = pd.to_datetime(rainfall_df['DATE'], errors='coerce')
rainfall_df["YEAR"] = rainfall_df['DATE'].dt.year
rainfall_df["MONTH"] = rainfall_df['DATE'].dt.month

rainfall_df.drop(columns=['DATE', 'VARIABLE_NOTES', 'FREQUENCY'], inplace=True)

rainfall_df = rainfall_df[~rainfall_df.VARIABLE_NAME.str.contains("%")]

rainfall_df.to_csv("rainfall-processed.csv", index=False, header=True)

print("Agar-Malwa".replace("-", " ").lower().__contains__("AGAR MALWA".lower()))
