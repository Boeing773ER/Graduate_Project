"""maxgen = 500"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file_path = "../img/formal/SEAIR/obj_trace_2-SEAIR-上海-1000-4456-13_14_58.csv"
file_path = "../img/formal/SEIR/obj_trace_1-SEIR_V-上海-1000-10225-13_14_37.csv"
data_file = pd.read_csv(file_path)
# 取出指定日期范围数据
#
# print(type(data_file))

start_date = 200

C1 = data_file.iloc[start_date:, 0]
C2 = data_file.iloc[start_date:, 1]

plt.style.use("ggplot")
# 设置中文编码和符号的正常显示
plt.rcParams["font.sans-serif"] = "KaiTi"
plt.rcParams["axes.unicode_minus"] = False
# 设置图框的大小

gen = 999 - start_date
t = np.linspace(1, gen, gen)

plt.plot(t, C2, 'b', label='C1')

# plt.plot(t,  # x轴数据
#          C1,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='steelblue',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色
# plt.plot(t,  # x轴数据
#          C2,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='g',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色

plt.show()
