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

# print(data_file["date"], type(data_file["date"]))

date = data_file.loc[:, "year":"date"]
date.insert(loc=0, column='year-date', value=0)
array = pd.to_numeric(data_file["year"])
year_list = array.tolist()
array = pd.to_numeric(data_file["date"])
date_list = array.tolist()
year_date_list = []
z = zip(year_list, date_list)
for year, date in z:
    str_date = str(date)
    month = str_date.split('.')[0]
    day = str_date.split('.')[1]
    str_date = '{:0>2d}'.format(int(month)) + '-' + '{:0>2d}'.format(int(day))
    temp_str = str(year) + "-" + str(str_date)
    year_date_list.append(temp_str)
print(data_file.columns)
data_file.drop(columns=['year', 'date'], axis=1, inplace=True)
data_file.insert(loc=0, column="date", value=year_date_list)
data_file.rename(columns={'deadAdd': 'dead_add', 'now_wzz': 'now_asy', 'wzz_add': 'asy_add'}, inplace=True)
print(data_file.columns)
data_file.to_csv("./CN_COVID_data/domestic_data.csv")


# plt.style.use("ggplot")
# # 设置中文编码和符号的正常显示
# plt.rcParams["font.sans-serif"] = "KaiTi"
# plt.rcParams["axes.unicode_minus"] = False
# 设置图框的大小
# fig = plt.figure(figsize=(30, 18))
#
# plt.plot(sub_data.date,  # x轴数据
#          sub_data.confirm_add,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='steelblue',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=2,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown')  # 点的填充色

# days = 30
# T = [i for i in range(0, days)]  # 时间
# # result = discrete_stochastic_model.calc(T)
#
# plt.plot(T, , color='b', label='传染者')
# plt.show()


# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
# plt.rcParams['font.family'] = ['sans-serif']
# plt.rcParams['font.sans-serif'] = ['SimHei']
# # % matplotlib inline  # jupyter notebook图像显示设置
# # % config InlineBackend.figure_format = 'svg'  # jupyter notebook图像矢量图设置
#
#
# # 自定义函数，curve_fit支持自定义函数的形式进行拟合，这里定义的是指数函数的形式
# # 包括自变量x和a，b，c三个参数
# def func(x, a, b, c):
#     return a * np.exp(-b * x) + c
#
#
# # 产生数据
# xdata = np.linspace(0, 4, 50)  # x从0到4取50个点
# y = func(xdata, 2.5, 1.3, 0.5)  # 在x取xdata，a，b，c分别取2.5, 1.3, 0.5条件下，运用自定义函数计算y的值
#
# # 在y上产生一些扰动模拟真实数据
# np.random.seed(1729)
# # 产生均值为0，标准差为1，维度为xdata大小的正态分布随机抽样0.2倍的扰动
# y_noise = 0.2 * np.random.normal(size=xdata.size)
# ydata = y + y_noise
# plt.plot(xdata, ydata, 'b-', label='data')
#
# # 利用“真实”数据进行曲线拟合
# popt, pcov = curve_fit(func, xdata, ydata)  # 拟合方程，参数包括func，xdata，ydata，
# # 有popt和pcov两个个参数，其中popt参数为a，b，c，pcov为拟合参数的协方差
#
# # plot出拟合曲线，其中的y使用拟合方程和xdata求出
# plt.plot(xdata, func(xdata, *popt), 'r-',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
# #     如果参数本身有范围，则可以设置参数的范围，如 0 <= a <= 3,
# #     0 <= b <= 1 and 0 <= c <= 0.5:
# popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))  # bounds为限定a，b，c参数的范围
#
# plt.plot(xdata, func(xdata, *popt), 'g--',
#          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()
