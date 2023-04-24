# ----- convert date format -----
import numpy
import pandas as pd
import matplotlib.pyplot as plt

confirmed_file_path = "./country_data/JHU_Confirmed_Global.csv"
death_file_path = "./country_data/JHU_Death_Global.csv"
recovered_file_path = "./country_data/JHU_Recovered_Global.csv"

confirmed_data = pd.read_csv(confirmed_file_path)
death_data = pd.read_csv(death_file_path)
recovered_data = pd.read_csv(recovered_file_path)

temp_date_list = list()
for column in confirmed_data.columns:
    temp_date_list.append(column)
temp_date_list = temp_date_list[4:]

date_list = []
for date in temp_date_list:
    temp_list = date.split('/')
    year = "20" + temp_list[2]
    month = '{:0>2d}'.format(int(temp_list[0]))
    day = '{:0>2d}'.format(int(temp_list[1]))
    temp_str = year + '-' + month + '-' + day
    date_list.append(temp_str)

country_dataframe = pd.DataFrame(columns=["country", "region", "date", "now_cases", "total_cases", "total_deaths",
                                          "total_recovered"])

region = "Japan"
province = ""

# 取出指定日期范围数据
temp_confirmed_data = confirmed_data.loc[confirmed_data["Country/Region"] == region, :]
temp_death_data = death_data.loc[death_data["Country/Region"] == region, :]
temp_recovered_data = recovered_data.loc[recovered_data["Country/Region"] == region, :]
for date, date_str in zip(temp_date_list, date_list):
    data = []
    data.append(temp_confirmed_data[date].tolist()[0])
    data.append(temp_death_data[date].tolist()[0])
    data.append(temp_recovered_data[date].tolist()[0])
    country_dataframe.loc[len(country_dataframe.index)] = [region, province, date_str, data[0]-data[1]-data[2]]+data
    # print(country_dataframe)

country_dataframe.to_csv("./country_data/JHU_"+region+".csv")


plt.plot(country_dataframe.date,  # x轴数据
         country_dataframe.now_cases,  # y轴数据
         linestyle='-',  # 折线类型
         linewidth=2,  # 折线宽度
         color='steelblue',  # 折线颜色
         marker='o',  # 点的形状
         markersize=2,  # 点的大小
         markeredgecolor='black',  # 点的边框色
         markerfacecolor='brown')  # 点的填充色
plt.show()