import pandas as pd

crop = 'Rice'

dataset = pd.DataFrame(columns=['State', 'Year', 'Season', 'Produce', 'Temperature', 'Rainfall'])

df = pd.read_csv('dataset.csv')
temperature_df = pd.read_csv('temperature.csv', skiprows=1)
rainfall_df = pd.read_csv('rainfall-processed.csv')

# get unique states and years
state_list = sorted(df['State_Name'].unique())
year_list = sorted(df['Crop_Year'].unique())

df['State_Name'] = df['State_Name'].str.strip()
df['Season'] = df['Season'].str.strip()
df['Crop'] = df['Crop'].str.strip()

season_list = ["Kharif", "Rabi", "Whole Year", "Summer", "Winter"]

print("Crop Years: %", sorted(df['Crop_Year'].unique()))
print("Rainfall Years: %", sorted(rainfall_df['YEAR'].unique()))

print("Rainfall districts: ", sorted(rainfall_df["DISTRICTS_NAME"].unique()))
print("Crop districts: ", sorted(df["District_Name"].unique()))

print("Rainfall districts Length: ", len(rainfall_df["DISTRICTS_NAME"].unique()))
print("Crop districts Length: ", len(df["District_Name"].unique()))

year_list = ["2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015"]

# print(state_list)
# print(sorted(rainfall_df["SUBDIVISION"].unique()))

# temperature_df = temperature_df[
# temperature_df.columns.drop(list(temperature_df.filter(regex='Administrative unit not ava*')))]

# for state in state_list:
#     state_subset = df[df["State_Name"] == state]
#     # district_list = state_subset["District_Name"].unique()
#     # print(len(district_list))
#     for year in year_list:
#         year_subset = state_subset[state_subset["Crop_Year"] == year]
#         for season in season_list:
#             season_subset = year_subset[year_subset["Season"] == season]
#             if not season_subset.empty:
#                 crop_subset = season_subset[season_subset["Crop"] == crop]
#                 if not crop_subset.empty:
#                     total_production = crop_subset["Production"].sum()
#                     total_area = crop_subset["Area"].sum()
#                     produce = (total_production / total_area)
#
#                     rainfall_state_df = rainfall_df[rainfall_df["SUBDIVISION"].str.contains(state)]
#                     rainfall_year_df = rainfall_state_df[rainfall_state_df["YEAR"] == year]
#
#                     if season == "Kharif" or season == "Summer":
#                         rainfall_annual_df = rainfall_year_df["JJAS"]
#                         if not rainfall_annual_df.empty:
#                             rainfall = rainfall_annual_df.iloc[0]
#                             combined_season = "Kharif"
#                         else:
#                             continue
#
#                     if season == "Whole Year":
#                         rainfall_annual_df = rainfall_year_df["ANNUAL"]
#                         if not rainfall_annual_df.empty:
#                             rainfall = rainfall_annual_df.iloc[0]
#                             combined_season = "Whole Year"
#                         else:
#                             continue
#
#                     if season == "Rabi" or season == "Winter":
#                         rainfall = rainfall_year_df[["OND", "JF", "MAR"]].sum().sum()
#                         combined_season = "Rabi"
#
#                     temp_year_df = temperature_df[temperature_df["Year"] == year]
#                     temp_state_df = temp_year_df[state]
#
#                     if not rainfall_year_df.empty and not temp_state_df.empty and (not pd.isna(produce)):
#                         dataset = dataset.append(
#                             {'State': state, 'Year': year, 'Season': combined_season, 'Produce': produce,
#                              'Temperature': temp_state_df.iloc[0],
#                              'Rainfall': rainfall},
#                             ignore_index=True)
#
# dataset.to_csv(crop+".csv", index=False)
# print("Saving crop file as " + crop + ".csv")
