import pandas as pd

crop = 'Banana'

dataset = pd.DataFrame(columns=['State', 'Year', 'Produce', 'Temperature', 'Rainfall'])

df = pd.read_csv('dataset.csv')
temperature_df = pd.read_csv('temperature.csv', skiprows=1)
rainfall_df = pd.read_csv('rainfall.csv')

# get unique states and years
state_list = sorted(df['State_Name'].unique())
year_list = sorted(df['Crop_Year'].unique())

print(state_list)

temperature_df = temperature_df[temperature_df.columns.drop(list(temperature_df.filter(regex='Administrative unit not ava*')))]
print(sorted(temperature_df.columns.unique()))

for state in state_list:
    # remove "Andaman and Nicobar Islands" and replace with state
    state_subset = df[df["State_Name"] == "Andaman and Nicobar Islands"]
    for year in year_list:
        year_subset = state_subset[state_subset["Crop_Year"] == year]
        if not year_subset.empty:
            crop_subset = year_subset[year_subset["Crop"] == crop]
            produce = sum(crop_subset["Production"]) / sum(crop_subset["Area"])
            dataset = dataset.append(
                {'State': state, 'Year': year, 'Produce': produce, 'Temperature': 10, 'Rainfall': 10},
                ignore_index=True)
            # remove these breaks later
            break
    # remove these breaks later
    break

print("Final Dataset :: ")
print(str(dataset))
