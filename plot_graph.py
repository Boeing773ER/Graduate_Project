import pandas as pd
import matplotlib.pyplot as plt
from SEIR_model import calc_SEIR

file_path = "./country_data/Hong Kong.csv"
data_file = pd.read_csv(file_path)

plt.style.use("ggplot")
# 设置中文编码和符号的正常显示
plt.rcParams["font.sans-serif"] = "KaiTi"
plt.rcParams["axes.unicode_minus"] = False
# 取出8月份至9月28日的数据
sub_data = data_file.loc[data_file.date >= '2022-02-01', :]
# sub_data = data_file


N = 7154600  # 人口总数
E = [0]  # 潜伏携带者
I = [1]  # 传染者
S = [N - I[0]]  # 易感者
R = [0]  # 康复者
r = 5.7  # 传染者接触人数
b = 0.03  # 传染者传染概率
a = 0.1  # 潜伏者患病概率
r2 = 10  # 潜伏者接触人数
b2 = 0.03  # 潜伏者传染概率
y = 0.1  # 康复概率
days = 335
T = [i for i in range(0, days)]
Y = calc_SEIR(N, T, S, E, I, R, r, b, a, r2, b2, y)

# 设置图框的大小
fig = plt.figure(figsize=(30, 18))
# 绘图
plt.plot(sub_data.date,  # x轴数据
         sub_data.new_cases,  # y轴数据
         linestyle='-',  # 折线类型
         linewidth=2,  # 折线宽度
         color='steelblue',  # 折线颜色
         marker='o',  # 点的形状
         markersize=2,  # 点的大小
         markeredgecolor='black',  # 点的边框色
         markerfacecolor='brown')  # 点的填充色
# 绘图
plt.plot(sub_data.date,  # x轴数据
         sub_data.total_cases,  # y轴数据
         linestyle='-',  # 折线类型
         linewidth=2,  # 折线宽度
         color='r',  # 折线颜色
         marker='o',  # 点的形状
         markersize=2,  # 点的大小
         markeredgecolor='black',  # 点的边框色
         markerfacecolor='brown')  # 点的填充色

# plt.plot(T, S, color='r', label='易感者')
# plt.plot(T, E, color='k', label='潜伏者')
plt.plot(T, I, color='b', label='传染者')
# plt.plot(T, R, color='g', label='移除者')


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
