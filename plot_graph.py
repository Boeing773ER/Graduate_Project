import pandas as pd
import matplotlib.pyplot as plt
from SEIR_model import seir_graph
import discrete_stochastic_model

file_path = "./country_data/Hong Kong.csv"
data_file = pd.read_csv(file_path)

plt.style.use("ggplot")
# 设置中文编码和符号的正常显示
plt.rcParams["font.sans-serif"] = "KaiTi"
plt.rcParams["axes.unicode_minus"] = False
# 取出指定日期范围数据
sub_data = data_file.loc[data_file.date >= '2021-01-01', :]
# sub_data = data_file

# 设置图框的大小
fig = plt.figure(figsize=(30, 18))
# new cases
# plt.plot(sub_data.date,  # x轴数据
#          sub_data.new_cases,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='steelblue',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色
# total cases
# plt.plot(sub_data.date,  # x轴数据
#          sub_data.total_cases,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='r',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色


# plt.plot(sub_data.date,  # x轴数据
# sub_data.reproduction_rate,  # y轴数据
# linestyle='-',  # 折线类型
# linewidth=2,  # 折线宽度
# color='r',  # 折线颜色
# marker='o',  # 点的形状
# markersize=2,  # 点的大小
# markeredgecolor='black',  # 点的边框色
# markerfacecolor='brown')  # 点的填充色


# draw SEIR line on the graph
# seir_graph()

# 添加标题和坐标轴标签
plt.title('香港新冠感染总数图')
plt.xlabel('日期')
plt.ylabel('感染总人数')

# 剔除图框上边界和右边界的刻度
plt.tick_params(top='off', right='off')

# 为了避免x轴日期刻度标签的重叠，设置x轴刻度自动展现，并且45度倾斜
fig.autofmt_xdate(rotation=45)

# 显示图形
plt.show()
