import _io
import math
# import numpy as np
# from scipy.integrate import odeint
# import matplotlib.pyplot as plt
import geatpy as ea
import time
from functions import *

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


model_name = "SEAIR"
file_path = "../CN_COVID_data/domestic_data.csv"
region = "上海"
start_date = "2022-03-10"
end_date = "2022-04-17"
plot_end_date = "2022-05-01"
"""region = "陕西"
start_date = "2022-08-11"
end_date = "2022-09-20"
plot_end_date = "2022-09-20" """
days = calc_days(start_date, end_date) - 2

# dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2
# y0 = [0, 0, 1, 7, 1, 2, 3383, 0]
# y0 = [0, 0, 500, 548, 2800, 2793, 4472, 0]
y0 = [0, 0, 500, 646, 2800, 370, 4067, 0]
t = np.linspace(0, days, days + 1)

rho = 0.85
phi = 3.696e-5
beta = 0.4
epsilon = 0.5
alpha = 0.2
eta = 0.75
theta = 0.75
mu = 0.2
gamma_I = 7e-4
gamma_A = 1e-4
gamma_Aq = 0.03
chi = 0
N_e = {"上海": 2.489e7, "湖北": 5.830e7, "陕西": 3.95e7}
z_1 = 0.045
z_2 = 0.026
a = 28
b = 5

# rho = 0.7
# phi = 0.001
# beta = 0.06927913080632363
# epsilon = 0.9621558933040346
# alpha = 0.03967527314899591
# eta = 0.27186717939327354
# theta = 0.5458096807666484
# mu = 0.003784410669596533
# gamma_I = 0.759506805835317
# gamma_A = 0.0610999206494537
# gamma_Aq = 0.05
# gamma_Iq = 0.05
# chi = 1
# N_e = 2.489e7
# z_1 = 0.045
# z_2 = 0.026
# a = 28
# b = 5


# gamma_Iq = z_1 + z_2 * math.tanh((10 - a) / b)


# rho = 0.5
# phi = 0.5
# beta = 0.5
# epsilon = 0.5
# alpha = 0.5
# eta = 0.5
# theta = 0.5
# mu = 0.5
# gamma_I = 0.5
# gamma_A = 0.5
# gamma_Aq = 0.5
# # gamma_Iq = 0.05
# chi = 0
# N_e = 2.489e7
# z_1 = 0.045
# z_2 = 0.026
# a = 28
# b = 5

""" ===========变量设置==========="""
params_count = 15
# x1 = [0.1, 0.95]
# x2 = [1e-6, 1e-3]  # 第一个决策变量范围
# x3 = [0.1, 0.9]
# x4 = [0.1, 0.9]
# x5 = [0.1, 1]
# x6 = [0.2, 0.95]
# x7 = [0.1, 0.95]
# x8 = [0, 1]
# x9 = [0, 1]
# x10 = [0, 1]
x1 = [0, 1]
x2 = [0, 1]  # 第一个决策变量范围
x3 = [0, 1]
x4 = [0, 1]
x5 = [0, 1]
x6 = [0, 1]
x7 = [0, 1]
x8 = [0, 1]
x9 = [0, 1]
x10 = [0, 1]
x11 = [0, 1]
x12 = [0, 1]
x13 = [0, 1]
x14 = [0, 100]
x15 = [0, 10]

b1 = [0, 0]  # 第一个决策变量边界，1表示包含范围的边界，0表示不包含
b2 = [0, 0]
b3 = [0, 0]
b4 = [0, 0]
b5 = [0, 0]
b6 = [0, 0]
b7 = [0, 0]
b8 = [0, 0]
b9 = [0, 0]
b10 = [0, 0]
b11 = [0, 0]
b12 = [0, 0]
b13 = [0, 0]
b14 = [0, 0]
b15 = [0, 0]

ranges = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
borders = np.vstack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15]).T  # 生成自变量的边界矩阵
print(ranges, borders)
varTypes = np.array(np.zeros(params_count))  # 决策变量的类型，0表示连续，1表示离散
# varTypes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 决策变量的类型，0表示连续，1表示离散
# ranges = np.vstack([x1, x2, x3, x4, x5, x6]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
# borders = np.vstack([b1, b2, b3, b4, b5, b6]).T  # 生成自变量的边界矩阵
# varTypes = np.array([0, 0, 0, 0, 0, 0])  # 决策变量的类型，0表示连续，1表示离散

""" ===========染色体编码设置==========="""
Encoding = 'BG'  # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
codes = np.zeros(params_count)  # 决策变量的编码方式，0表示决策变量使用二进制编码
precisions = []
for i in range(params_count):
    precisions.append(4)  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位
scales = np.zeros(params_count)  # 0表示采用算术刻度，1表示采用对数刻度
# codes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 决策变量的编码方式，0表示决策变量使用二进制编码
# scales = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0表示采用算术刻度，1表示采用对数刻度
# precisions = [4, 4, 4, 4, 4, 4]  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位


FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

""" ===========遗传算法参数设置==========="""
NIND = 100  # 种群个体数目
MAXGEN = 600  # 最大遗传代数
maxormins = np.array([1])  # 1：目标函数最小化，-1：目标函数最大化
select_style = 'etour'  # 轮盘赌选择
rec_style = 'xovdp'  # 两点交叉
mut_style = 'mutbin'  # 二进制染色体的变异算子
Lind = int(np.sum(FieldD[0, :]))  # 染色体长度
print(Lind)
print(FieldD)
pc = 0.4  # 交叉概率
pm = 1 / Lind  # 变异概率
print(pm)
obj_trace = np.zeros((MAXGEN, 2))
var_trace = np.zeros((MAXGEN, int(Lind)))

y_data = read_file(file_path, region, start_date, end_date)


def model(y, t, rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, z_1, z_2, a, b, chi, N_e):
    E, E_q, I, I_q, A, A_q, R_1, R_2 = y

    gamma_Iq = z_1 + z_2 * math.tanh((t - a) / b)

    dE = (1 - rho) * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E
    dE_q = rho * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E_q
    dI = alpha * eta * E - theta * I - gamma_I * I
    dI_q = alpha * eta * E_q + theta * I - gamma_Iq * I_q
    dA = alpha * (1 - eta) * E - mu * A - gamma_A * A
    dA_q = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
    dR_1 = gamma_Iq * I_q + chi * gamma_Aq * A_q
    dR_2 = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

    return dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2


# 种群染色体矩阵(Chrom)
# 种群表现型矩阵(Phen)
# 种群个体违反约束程度矩阵(CV)
# 种群适应度(FitnV)
def aim(Phen, CV):
    # rho = Phen[:, [0]]
    # phi = Phen[:, [1]]
    # beta = Phen[:, [2]]
    # epsilon = Phen[:, [3]]
    # alpha = Phen[:, [4]]
    # eta = Phen[:, [5]]
    # theta = Phen[:, [6]]
    # mu = Phen[:, [7]]
    # gamma_I = Phen[:, [8]]
    # gamma_A = Phen[:, [9]]
    # gamma_Aq = Phen[:, [10]]
    # z_1 = Phen[:, [11]]
    # z_2 = Phen[:, [12]]
    # a = Phen[:, [13]]
    # b = Phen[:, [14]]
    # rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, chi, N_e, z_1, z_2, a, b

    # TODO: 之前参数设置错误，SEAIR的实验数据有误

    # for rho_x, phi_x, beta_x, epsilon_x, alpha_x, eta_x, theta_x, mu_x, gamma_I_x, gamma_A_x, gamma_Aq_x, z_1_x, z_2_x, \
    #     a_x, b_x in zip(rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, z_1, z_2, a, b):
    #     # 计算目标函数值
    #     sol = odeint(model, y0, t, args=(rho_x[0], phi_x[0], beta_x[0], epsilon_x[0], alpha_x[0], eta_x[0], theta_x[0],
    #                                      mu_x[0], gamma_I_x[0], gamma_A_x[0], gamma_Aq[0], z_1_x[0],
    #                                      z_2_x[0], a_x[0], b_x[0], chi, N_e[region]))
    f = []
    for phen in Phen:
        # 计算目标函数值
        sol = odeint(model, y0, t, args=(*phen, chi, N_e[region]))
        I_q = sol[:, 3]
        A_q = sol[:, 5]
        R_q = sol[:, 6]

        loss1 = loss_eva(rmse_loss, I_q, y_data.now_confirm.to_numpy())
        loss2 = loss_eva(rmse_loss, A_q, y_data.now_asy.to_numpy())
        loss3 = loss_eva(rmse_loss, R_q, y_data.heal.to_numpy())
        loss = np.mean([loss1, loss2, loss3])
        # loss = np.mean([loss1, loss3])
        # loss = loss1
        f.append([loss])
        # print(f)
    f = np.array(f)
    return f, CV  # 返回目标函数值矩阵


def write_param(log_file: _io.TextIOWrapper):
    log_file.writelines(region)
    temp_str = "\ninit setting:\n"
    temp_str += "y0:" + str(y0) + "\n"
    temp_str += "start date:" + str(start_date) + "\n"
    temp_str += "end data:" + str(end_date) + "\n"
    temp_str += "rho: " + str(rho) + "\n"
    temp_str += "phi: " + str(phi) + "\n"
    temp_str += "beta: " + str(beta) + "\n"
    temp_str += "epsilon: " + str(epsilon) + "\n"
    temp_str += "alpha: " + str(alpha) + "\n"
    temp_str += "eta: " + str(eta) + "\n"
    temp_str += "theta: " + str(theta) + "\n"
    temp_str += "mu: " + str(mu) + "\n"
    temp_str += "gamma_I: " + str(gamma_I) + "\n"
    temp_str += "gamma_A: " + str(gamma_A) + "\n"
    temp_str += "gamma_Aq: " + str(gamma_Aq) + "\n"
    temp_str += "chi: " + str(chi) + "\n"
    temp_str += "N_e: " + str(N_e) + "\n"
    temp_str += "z_1: " + str(z_1) + "\n"
    temp_str += "z_2: " + str(z_2) + "\n"
    temp_str += "a: " + str(a) + "\n"
    temp_str += "b: " + str(b) + "\n"
    log_file.writelines(temp_str)


def draw_result(file_path, log_file_name, region, start_date, end_date, variable):
    y_data = read_file(file_path, region, start_date, end_date)
    days = calc_days(start_date, end_date) - 2
    # y0 = [0, 0, 1, 7, 1, 2, 0, 3383]
    t = np.linspace(0, days, days + 1)
    sol = odeint(model, y0, t, args=(variable[0, 0], variable[0, 1], variable[0, 2], variable[0, 3], variable[0, 4],
                                     variable[0, 5], variable[0, 6], variable[0, 7], variable[0, 8], variable[0, 9],
                                     variable[0, 10], variable[0, 11], variable[0, 12],
                                     variable[0, 13], variable[0, 14], chi, N_e[region]))
    plot_graph(log_file_name, sol, model_name, region, t, y_data, [3, 5, 6])


def start_GA(iter_round):
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
        print(time.ctime())
        print("Gen:", gen)
        print(ObjV[best_ind])
    # 进化完成
    end_time = time.time()  # 结束计时

    """============================输出结果============================"""
    best_gen = np.argmin(obj_trace[:, [1]])
    print("最优解代数：", best_gen)

    temp_t = time.localtime()
    day = temp_t.tm_mday
    hour = temp_t.tm_hour
    minute = temp_t.tm_min
    log_file_name = str(iter_round) + '-' + model_name + '-'
    log_file_name += region + '-'
    log_file_name += str(MAXGEN) + '-'
    log_file_name += str(int(obj_trace[best_gen, 1])) + '-'
    log_file_name += str(day) + '_' + str(hour) + '_' + str(minute)

    ea.trcplot(obj_trace, [['Average value of population', 'Population optimal individual value']],
               save_path="../img/track-"+log_file_name+' ')  # 绘制图像
    np.savetxt("../log/obj_trace_"+log_file_name+".csv", obj_trace, delimiter=',')

    with open("../log/" + log_file_name + ".txt", mode='w', encoding="utf-8") as log_file:
        write_param(log_file)
        temp_str = '最优解的目标函数值：' + str(obj_trace[best_gen, 1])
        print(temp_str)
        log_file.writelines(temp_str + "\n")
        variable = ea.bs2ri(var_trace[[best_gen], :], FieldD)  # 解码得到表现型（即对应的决策变量值）
        print('最优解的决策变量值为：')
        log_file.writelines('最优解的决策变量值为：' + "\n")
        for i in range(variable.shape[1]):
            temp_str = 'x' + str(i) + '=' + str(variable[0, i])
            print(temp_str)
            log_file.writelines(temp_str + "\n")
        log_file.writelines("\n")
        print('用时：', end_time - start_time, '秒')

    sol = odeint(model, y0, t, args=(variable[0, 0], variable[0, 1], variable[0, 2], variable[0, 3], variable[0, 4],
                                     variable[0, 5], variable[0, 6], variable[0, 7], variable[0, 8], variable[0, 9],
                                     variable[0, 10], variable[0, 11], variable[0, 12],
                                     variable[0, 13], variable[0, 14], chi, N_e[region]))
    # 保存数据值csv
    np.savetxt("../log/" + log_file_name + ".csv", sol, delimiter=',', header="E, Eq, I, Iq, A, Aq, R1, R2",
               comments="")

    draw_result(file_path, log_file_name, region, start_date, plot_end_date, variable)


# for pc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     for i in range(10):
#         start_GA(i)
for i in range(10):
    start_GA(i)
