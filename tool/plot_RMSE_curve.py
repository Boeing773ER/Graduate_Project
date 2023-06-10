"""maxgen = 500"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# file_path = "../img/formal/SEAIR/obj_trace_2-SEAIR-上海-1000-4456-13_14_58.csv"
file_path = "D:\COVID_Prediction\Code\img\\formal\爬坡组\SEIR\obj_trace_1-SEIR_V-上海-1000-2164-20_17_15.csv"
# file_path = "../img/formal/SEAIR/obj_trace_3-SEAIR-上海-1000-3715-13_15_5.csv"
data_file = pd.read_csv(file_path)
# 取出指定日期范围数据
#
# print(type(data_file))

start_gen = 200

C1 = data_file.iloc[start_gen:, 0]
C2 = data_file.iloc[start_gen:, 1]

plt.style.use("ggplot")
# 设置中文编码和符号的正常显示
plt.rcParams["font.sans-serif"] = "KaiTi"
plt.rcParams["axes.unicode_minus"] = False
# 设置图框的大小

gen = 1000
t = np.linspace(start_gen, gen, gen - start_gen-1)

plt.plot(t, C2, 'b', label='C1')
plt.title("SEIAR model RMSE")
plt.xlabel("Gen")
plt.ylabel("RMSE")
plt.show()
