import pandas as pd
import matplotlib.pyplot as plot

file_name = "./owid-covid-data.csv"

data_file = pd.read_csv(file_name)

print(data_file.columns)
column_list = []
for column in data_file.columns:
    column_list.append(column)

country = pd.DataFrame(columns=column_list)
country_name = "Hong Kong"
for index, row in data_file.iterrows():
    location = row["location"]
    if location == country_name:
        country.loc[len(country.index)] = row

target_file = "./CSV/"+country_name+".csv"
country.to_csv(target_file, index=False, sep=',')
