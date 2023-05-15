import pandas as pd
import matplotlib.pyplot as plt

file_path = "../CN_COVID_data/domestic_data.csv"
data_file = pd.read_csv(file_path)
# 取出指定日期范围数据
sub_data = data_file.loc[data_file.province == "陕西", :]
# sub_data = sub_data.loc[data_file.city == "浦东"]
# sub_data = sub_data.loc[data_file.date <= 5, :]
sub_data = sub_data.loc[data_file.date > "2022-08-11", :]
sub_data = sub_data.loc["2022-09-20" > data_file.date, :]
# sub_data.to_csv("./CN_COVID_data/beijing_data.csv")
# sub_data = data_file


plt.style.use("ggplot")
# 设置中文编码和符号的正常显示
plt.rcParams["font.sans-serif"] = "KaiTi"
plt.rcParams["axes.unicode_minus"] = False
# 设置图框的大小
# fig = plt.figure(figsize=(30, 18))

plt.plot(sub_data.date, sub_data.now_confirm, 'r', label='Confirmed')
plt.plot(sub_data.date, sub_data.now_asy, 'b', label='Asymptom')
# plt.plot(sub_data.date, sub_data.heal, 'g', label='Removed')


# plt.plot(sub_data.date,  # x轴数据
#          sub_data.now_confirm,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='steelblue',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色
# plt.plot(sub_data.date,  # x轴数据
#          sub_data.now_asy,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='g',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色
# plt.plot(sub_data.date,  # x轴数据
#          sub_data.heal,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='r',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色

# days = 30
# T = [i for i in range(0, days)]  # 时间
# result = discrete_stochastic_model.calc(T)

# plt.plot(T, , color='b', label='传染者')
plt.show()
