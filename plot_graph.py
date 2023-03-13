import pandas as pd
import matplotlib.pyplot as plt
from Classic_SEIR_model import seir_graph
import discrete_stochastic_model
import SEAIR_model


def get_file(file_path):
    data_file = pd.read_csv(file_path)
    # 取出指定日期范围数据
    sub_data = data_file.loc[data_file.date >= '2022/10/01', :]
    # sub_data = data_file
    return sub_data


def config_graph():
    plt.style.use("ggplot")
    # 设置中文编码和符号的正常显示
    plt.rcParams["font.sans-serif"] = "KaiTi"
    plt.rcParams["axes.unicode_minus"] = False
    # 设置图框的大小
    fig = plt.figure(figsize=(30, 18))
    return fig


file_path = "./CN_COVID_data/shanxi_data.csv"
sub_data = get_file(file_path)

fig = config_graph()
# new cases
plt.plot(sub_data.date,  # x轴数据
         sub_data.p_conf,  # y轴数据
         linestyle='-',  # 折线类型
         linewidth=2,  # 折线宽度
         color='steelblue',  # 折线颜色
         marker='o',  # 点的形状
         markersize=2,  # 点的大小
         markeredgecolor='black',  # 点的边框色
         markerfacecolor='brown')  # 点的填充色
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

days = 30
# draw SEIR line on the graph
# seir_graph(days)

T = [i for i in range(0, days)]  # 时间
result = discrete_stochastic_model.calc(T)

plt.plot(T, result['I'], color='b', label='传染者')
# plt.plot(T, result['S'], color='r', label='易感者')
# plt.plot(T, result['S_q'], color='y', label='隔离的易感者')
# plt.plot(T, result['E'], color='g', label='暴露者')

print("Infected", result['I'])
print("Exposed", result['E'])
print("Susceptible", result['S'])


# 添加标题和坐标轴标签
plt.title('新冠感染总数图')
plt.xlabel('日期')
plt.ylabel('感染总人数')

# 剔除图框上边界和右边界的刻度
plt.tick_params(top='off', right='off')

# 为了避免x轴日期刻度标签的重叠，设置x轴刻度自动展现，并且45度倾斜
fig.autofmt_xdate(rotation=45)

# 显示图形
plt.show()
