import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import geatpy as ea
import time
from geatpy import crtpc
from geatpy import bs2ri

"""
ρ rho       被隔离的易感者的比例 0.1 ∼ 0.95 Yes
ϕ phi       传染性个体通过接触传播的概率 10^−6 ∼ 10^−3 Yes
β beta      无症状感染者相对于感染者的传播系数 0.1 ∼ 0.9 Yes
ε epsilon   暴露的个体相对于感染性个体的传播系数 0.1 ∼ 0.9 Yes
α alpha     暴露的个人发展为感染性状态的比率 0.1 ∼ 1 Yes
η eta       暴露者发展到有症状的感染状态的比例0.2 ∼ 0.95 Yes
θ theta     有症状的传染病人的隔离率 0.1 ∼ 0.95 Yes
µ mu        无症状感染者的检测率 0 ∼ 1 Yes
γI gamma_I  (未经检疫和有明显症状的)传染病人的清除率0 ∼ 1 Yes
γA gamma_A  未经检疫的无症状携带者和未发现的轻度携带者的清除率 0 ∼ 1 Yes
γIq(t)      有明显症状的被隔离传染者的清除率（= z1 + z2 tanh( t-a/b)） 0 ∼ 1 Yes
γAq gamma_Aq检测到的无症状携带者和检测到的轻度携带者的去除率 0 ∼ 1 Yes
χ chi       检测到的无症状携带者是否被算作确诊病例的二元指标 {0, 1} No

dE/dt = (1 - rho) * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E
dE_q/dt = rho * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E_q
dI/dt = alpha * eta * E - theta * I - gamma_I * I
dI_q/dt = alpha * eta * E_q + theta * I - gamma_Iq(t) * I_q
dA/dt = alpha * (1 - eta) * E - mu * A - gamma_A * A
dA_q/dt = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
dR_1/dt = gamma_Iq(t) * I_q + chi * gamma_Aq * A_q
dR_2/dt = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

gamma_Iq(t) = z_1 + z_2 * tanh((t - a)/b)
"""


def read_file(file_path):
    data_file = pd.read_csv(file_path)
    sub_data = data_file.loc[data_file.province == "湖北", :]
    sub_data = sub_data.loc["2020-02-10" > sub_data.date, :]
    ydata = pd.DataFrame()
    ydata["now_confirm"] = sub_data["now_confirm"]
    ydata["heal"] = sub_data["heal"]
    ydata["now_asy"] = sub_data["now_asy"]
    return ydata


def model(y, t, rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_A, gamma_Aq, eta, mu, chi, N_e, z_1, z_2, a, b):
    E, E_q, I, I_q, A, A_q, R_1, R_2 = y
    # E = y[0]
    # E_q = y[1]
    # I = y[2]
    # I_q = y[3]
    # A = y[4]
    # A_q = y[5]
    # R_1 = y[6]
    # R_2 = y[7]
    # rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_Iq, gamma_A, gamma_Aq, eta, mu, chi, N_e = u

    gamma_Iq = z_1 + z_2 * math.tanh((t - a) / b)

    dE = (1 - rho) * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E
    dE_q = rho * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E_q
    dI = alpha * eta * E - theta * I - gamma_I * I
    dI_q = alpha * eta * E_q + theta * I - gamma_Iq * I_q
    dA = alpha * (1 - eta) * E - mu * A - gamma_A * A
    dA_q = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
    dR_1 = gamma_Iq * I_q + chi * gamma_Aq * A_q
    dR_2 = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

    return [dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2]


file_path = "./CN_COVID_data/domestic_data.csv"
y_data = read_file(file_path)


# E, E_q, I, I_q, A, A_q, R_1, R_2
y0 = [0, 0, 1, 0, 0, 0, 0, 0]
days = 20
t = np.linspace(0, days, days + 1)

rho = 0.85
phi = 3.696e-5
epsilon = 0.5
beta = 0.4
alpha = 0.2
theta = 0.75
gamma_I = 7e-4
gamma_A = 1e-4
gamma_Aq = 0.03
eta = 0.75
mu = 0.2
chi = 0
N_e = 5.8e7
z_1 = 0.045
z_2 = 0.026
a = 64
b = 5
# params = [rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_A, gamma_Aq, eta, mu, chi, N_e, z_1, z_2, a, b]


def mse_loss(x: np.ndarray, y: np.ndarray):
    # x: prediction, y: real
    assert len(x) == len(y)
    # x = np.array(x)
    # y = np.array(y)
    loss = np.sum(np.square(x - y)) / len(x)
    return loss


# 种群染色体矩阵(Chrom)
# 种群表现型矩阵(Phen)
# 种群个体违反约束程度矩阵(CV)
# 种群适应度(FitnV)
def aim(Phen, CV):
    phi = Phen[:, [0]]
    alpha = Phen[:, [1]]

    f = []

    for phi_x, alpha_x in zip(phi, alpha):
        sol = odeint(model, y0, t, args=(rho, phi_x, epsilon, beta, alpha_x, theta, gamma_I, gamma_A, gamma_Aq, eta, mu, chi,
                                         N_e, z_1, z_2, a, b))  # 计算目标函数值
        I_q = sol[:, 3]
        A_q = sol[:, 5]
        R_q = sol[:, 6]

        loss1 = mse_loss(I_q, y_data.now_confirm.to_numpy())
        loss2 = mse_loss(A_q, y_data.now_asy.to_numpy())
        loss3 = mse_loss(R_q, y_data.heal.to_numpy())
        loss = np.mean([loss1, loss2, loss3])
        f.append([loss])
        # print(f)
    f = np.array(f)
    return f, CV  # 返回目标函数值矩阵


file_path = "./CN_COVID_data/domestic_data.csv"
ydata = read_file(file_path)


# 定义种群规模（个体数目）
# phi, alpha

# # 创建“区域描述器”，表明有4个决策变量，范围分别是[-3.1, 4.2], [-2, 2],[0, 1],[3, 5]，
# # FieldDR第三行[0,0,1,1]表示前两个决策变量是连续型的，后两个变量是离散型的
# FieldDR=np.array([[-3.1, -2, 0, 3],
#                   [ 4.2,  2, 1, 5],
#                   [ 0,    0, 1, 1]])
# # 调用crtri函数创建实数值种群
# Chrom=crtpc(Encoding, Nind, FieldDR)
# print(Chrom)

# -------变量设置--------
x1 = [1e-6, 1e-3]
x2 = [0.1, 1]
b1 = [1, 1]
b2 = [1, 1]
ranges = np.vstack([x1, x2]).T
borders = np.vstack([b1, b2]).T
varTypes = np.array([0, 0])

# --------染色体编码设置--------
Encoding = 'BG'  # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
codes = [0, 0]
precisions = [4, 4]
scales = [0, 0]
FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

# ---------遗传算法参数设置---------
NIND = 100
MAXGEN = 200
maxormins = np.array([1])
select_style = 'rws'
rec_style = 'xovdp'
mut_style = 'mutbin'
Lind = np.sum(FieldD[0, :])
pc = 0.7
pm = 1 / Lind
obj_trace = np.zeros((MAXGEN, 2))
var_trace = np.zeros((MAXGEN, int(Lind)))

"""=========================开始遗传算法进化========================"""
start_time = time.time()  # 开始计时
Chrom = ea.crtpc(Encoding, NIND, FieldD)  # 生成种群染色体矩阵
variable = ea.bs2ri(Chrom, FieldD)  # 对初始种群进行解码
CV = np.zeros((NIND, 1))  # 初始化一个CV矩阵（此时因为未确定个体是否满足约束条件，因此初始化元素为0，暂认为所有个体是可行解个体）
ObjV, CV = aim(variable, CV)  # 计算初始种群个体的目标函数值
FitnV = ea.ranking(ObjV, CV, maxormins)  # 根据目标函数大小分配适应度值
best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
# 开始进化
for gen in range(MAXGEN):
    print(time.ctime())
    print("Gen:", gen)
    SelCh = Chrom[ea.selecting(select_style, FitnV, NIND - 1), :]  # 选择
    SelCh = ea.recombin(rec_style, SelCh, pc)  # 重组
    SelCh = ea.mutate(mut_style, Encoding, SelCh, pm)  # 变异
    # 把父代精英个体与子代的染色体进行合并，得到新一代种群
    Chrom = np.vstack([Chrom[best_ind, :], SelCh])
    Phen = ea.bs2ri(Chrom, FieldD)  # 对种群进行解码(二进制转十进制)
    ObjV, CV = aim(Phen, CV)  # 求种群个体的目标函数值
    FitnV = ea.ranking(ObjV, CV, maxormins)  # 根据目标函数大小分配适应度值
    # 记录
    best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
    obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]  # 记录当代种群的目标函数均值
    obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
    var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体
# 进化完成
end_time = time.time()  # 结束计时
ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']])  # 绘制图像

"""============================输出结果============================"""
best_gen = np.argmax(obj_trace[:, [1]])
print('最优解的目标函数值：', obj_trace[best_gen, 1])
variable = ea.bs2ri(var_trace[[best_gen], :], FieldD)  # 解码得到表现型（即对应的决策变量值）
print('最优解的决策变量值为：')
for i in range(variable.shape[1]):
    print('x' + str(i) + '=', variable[0, i])
print('用时：', end_time - start_time, '秒')


# sol = odeint(model, y0, t, args=(rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_A, gamma_Aq, eta, mu, chi,
#                                      N_e, z_1, z_2, a, b))  # 计算目标函数值


# -----------plot graph------------
# # plt.plot(t, sol[:, 0], 'b', label='Exposed')
# # plt.plot(t, sol[:, 1], '--b', label='Exposed_quarantine')
#
# # plt.plot(t, sol[:, 2], 'g', label='Infected')
# plt.plot(t, sol[:, 3], '--g', label='Infected_quarantine')
#
# # plt.plot(t, sol[:, 4], 'r', label='Asy')
# plt.plot(t, sol[:, 5], '--r', label='Asy_quarantine')
#
# plt.plot(t, sol[:, 6], '--y', label='Removed_quarantined')
# # plt.plot(t, sol[:, 7], 'y', label='Removed')


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

# plt.legend(loc='best')
# plt.xlabel('t')
# plt.grid()
# plt.show()
