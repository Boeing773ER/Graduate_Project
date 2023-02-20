import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

file_path = "./CN_COVID_data/shanxi_data.csv"
data_file = pd.read_csv(file_path, converters={'date': str, 'p_conf': int, 'n_l_conf': int, 'conf2rec': int,
                                               'p_asy': int, 'n_l_asy': int, 'asy2rec': int, 'l_asy2conf': int})
# data_file = pd.read_csv(file_path)

plt.style.use("ggplot")
# 设置中文编码和符号的正常显示
plt.rcParams["font.sans-serif"] = "KaiTi"
plt.rcParams["axes.unicode_minus"] = False

sub_data = data_file.loc[data_file.date >= '2022/11/01', :]

total_count = 0
total_confirmed_case = []

date = []
total_asy_count = 0
total_asy_case = []


for index, row in data_file.iterrows():
    n_l_conf = row['n_l_conf']
    total_count += n_l_conf
    total_confirmed_case.append(total_count)

    n_l_asy = row['n_l_asy']
    total_asy_count += n_l_asy
    total_asy_case.append(total_asy_count)

    date.append(row['date'])


# # 设置图框的大小
fig = plt.figure(figsize=(10, 6))
# fig = plt.figure()
# plt.title("SEIR-nCoV 传播时间曲线")
# plt.plot(date, total_confirmed_case, color='r', label='总感染人数')
plt.plot(sub_data.date, sub_data.n_l_conf, color='k', label='新感染人数')
# plt.plot(sub_data.date, sub_data.n_l_asy, color='b', label='新无症状')
# plt.plot(date, total_asy_case, color='g', label='总无症状')
plt.grid(False)
plt.legend()
plt.xlabel("时间(天)")
plt.ylabel("人数")
pl.savefig('SEIR-nCoV 传播时间曲线.png', dpi=900)
# # 为了避免x轴日期刻度标签的重叠，设置x轴刻度自动展现，并且45度倾斜
fig.autofmt_xdate(rotation=45)
plt.show()



# # 绘图
# plt.plot(data_file.date,  # x轴数据
#          data_file.p_conf,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='steelblue',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色
# # 添加标题和坐标轴标签
# plt.title('陕西疫情统计')
# plt.xlabel('日期')
# plt.ylabel('新增感染人数')
#
# # 剔除图框上边界和右边界的刻度
# plt.tick_params(top='off', right='off')
#
# # 显示图形
# plt.show()


