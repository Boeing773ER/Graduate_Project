import pandas as pd
import matplotlib.pyplot as plot

country_name = "Japan"

def get_country_data(country_name):
    file_name = "./owid-covid-data.csv"
    data_file = pd.read_csv(file_name)
    # print(data_file.columns)
    column_list = list()
    for column in data_file.columns:
        column_list.append(column)
    country = pd.DataFrame(columns=column_list)
    for index, row in data_file.iterrows():
        location = row["location"]
        if location == country_name:
            country.loc[len(country.index)] = row
    target_file = "./country_data/"+country_name+".csv"
    country.to_csv(target_file, index=False, sep=',')

get_country_data(country_name)
