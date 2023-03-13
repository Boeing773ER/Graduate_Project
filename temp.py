import pandas as pd
import matplotlib.pyplot as plt

file_path = "./CN_COVID_data/domestic_recent_provinces.csv"
data_file = pd.read_csv(file_path)
# 取出指定日期范围数据
sub_data = data_file.loc[data_file.province == "上海", :]
# sub_data = sub_data.loc[data_file.city == "浦东"]
# sub_data = sub_data.loc[data_file.date <= 5, :]
sub_data = sub_data.loc[data_file.year == 2020]
# sub_data = data_file

plt.style.use("ggplot")
# 设置中文编码和符号的正常显示
plt.rcParams["font.sans-serif"] = "KaiTi"
plt.rcParams["axes.unicode_minus"] = False
# 设置图框的大小
fig = plt.figure(figsize=(30, 18))

plt.plot(sub_data.date,  # x轴数据
         sub_data.confirm_add,  # y轴数据
         linestyle='-',  # 折线类型
         linewidth=2,  # 折线宽度
         color='steelblue',  # 折线颜色
         marker='o',  # 点的形状
         markersize=2,  # 点的大小
         markeredgecolor='black',  # 点的边框色
         markerfacecolor='brown')  # 点的填充色

# days = 30
# T = [i for i in range(0, days)]  # 时间
# # result = discrete_stochastic_model.calc(T)
#
# plt.plot(T, , color='b', label='传染者')
plt.show()
