import pandas as pd

crop = 'Rice'

dataset = pd.DataFrame(columns=['State', 'Year', 'Produce', 'Temperature', 'Rainfall'])

df = pd.read_csv('dataset.csv')
temperature_df = pd.read_csv('temperature.csv', skiprows=1)
rainfall_df = pd.read_csv('rainfall.csv')

# get unique states and years
state_list = sorted(df['State_Name'].unique())
year_list = sorted(df['Crop_Year'].unique())

df['State_Name'] = df['State_Name'].str.strip()

# print(state_list)
# print(sorted(rainfall_df["SUBDIVISION"].unique()))

# temperature_df = temperature_df[
#     temperature_df.columns.drop(list(temperature_df.filter(regex='Administrative unit not ava*')))]

for state in state_list:
    # remove "Andaman and Nicobar Islands" and replace with state
    state_subset = df[df["State_Name"] == state]
    for year in year_list:
        year_subset = state_subset[state_subset["Crop_Year"] == year]
        if not year_subset.empty:
            crop_subset = year_subset[year_subset["Crop"] == crop]
            if not crop_subset.empty:
                produce = sum(crop_subset["Production"]) / sum(crop_subset["Area"])

                rainfall_state_df = rainfall_df[rainfall_df["SUBDIVISION"].str.contains(state)]
                rainfall_year_df = rainfall_state_df[rainfall_state_df["YEAR"] == year]
                rainfall_annual_df = rainfall_year_df["ANNUAL"]

                temp_year_df = temperature_df[temperature_df["Year"] == year]
                temp_state_df = temp_year_df[state]

                if not rainfall_year_df.empty and not temp_state_df.empty and not pd.isna(produce):
                    dataset = dataset.append(
                        {'State': state, 'Year': year, 'Produce': produce,
                         'Temperature': temp_state_df.iloc[0],
                         'Rainfall': rainfall_annual_df.iloc[0]},
                        ignore_index=True)

dataset.to_csv(crop + ".csv", index=False)
print("Saving crop file as " + crop + ".csv")
