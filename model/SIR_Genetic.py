import _io
import io
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import geatpy as ea
import time
from functions import calc_days, read_file, loss_eva, rmse_loss, mse_loss


model_name = "SEIR"
file_path = "./CN_COVID_data/domestic_data.csv"
region = "上海"
start_date = "2022-03-10"
end_date = "2022-04-17"
days = calc_days(start_date, end_date) - 2

round = 1

y0 = [0, 0, 1, 0, 0, 0, 0, 0]
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
# gamma_Iq = 0.05
chi = 0
N_e = {"上海": 2.489e7, "湖北": 5.830e7}
z_1 = 0.045
z_2 = 0.026
a = 28
b = 5


""" ===========变量设置==========="""
params_count = 10
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

b1 = [1, 1]  # 第一个决策变量边界，1表示包含范围的边界，0表示不包含
b2 = [1, 1]
b3 = [1, 1]
b4 = [1, 1]
b5 = [1, 1]
b6 = [1, 1]
b7 = [1, 1]
b8 = [1, 1]
b9 = [1, 1]
b10 = [1, 1]


ranges = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
borders = np.vstack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]).T  # 生成自变量的边界矩阵
# varTypes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 决策变量的类型，0表示连续，1表示离散
varTypes = np.array(np.zeros(params_count))  # 决策变量的类型，0表示连续，1表示离散
# ranges = np.vstack([x1, x2, x3, x4, x5, x6]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
# borders = np.vstack([b1, b2, b3, b4, b5, b6]).T  # 生成自变量的边界矩阵
# varTypes = np.array([0, 0, 0, 0, 0, 0])  # 决策变量的类型，0表示连续，1表示离散

""" ===========染色体编码设置==========="""
Encoding = 'BG'  # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
# codes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 决策变量的编码方式，0表示决策变量使用二进制编码
codes = np.zeros(params_count)  # 决策变量的编码方式，0表示决策变量使用二进制编码
precisions = []
for i in range(params_count):
    precisions.append(4)  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位
# scales = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0表示采用算术刻度，1表示采用对数刻度
scales = np.zeros(params_count)  # 0表示采用算术刻度，1表示采用对数刻度
# codes = [0, 0, 0, 0, 0, 0]  # 决策变量的编码方式，设置两个0表示两个决策变量均使用二进制编码
# precisions = [4, 4, 4, 4, 4, 4]  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位
# scales = [0, 0, 0, 0, 0, 0]  # 0表示采用算术刻度，1表示采用对数刻度

FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

""" ===========遗传算法参数设置==========="""
NIND = 100  # 种群个体数目
MAXGEN = 5  # 最大遗传代数
maxormins = np.array([1])  # 1：目标函数最小化，-1：目标函数最大化
select_style = 'rws'  # 轮盘赌选择
rec_style = 'xovdp'  # 两点交叉
mut_style = 'mutbin'  # 二进制染色体的变异算子
Lind = int(np.sum(FieldD[0, :]))  # 染色体长度
pc = 0.5  # 交叉概率
pm = 1 / Lind  # 变异概率
obj_trace = np.zeros((MAXGEN, 2))
var_trace = np.zeros((MAXGEN, int(Lind)))


y_data = read_file(file_path, region, start_date, end_date)


def model(y, t, rho, phi, epsilon, beta, alpha, theta, gamma_A, gamma_I, gamma_Aq, gamma_Iq, eta, mu, chi, N_e):
    I, I_q, A, A_q, R_1, R_2 = y

    dI = alpha * eta * E - theta * I - gamma_I * I
    dI_q = alpha * eta * E_q + theta * I - gamma_Iq * I_q
    dA = alpha * (1 - eta) * E - mu * A - gamma_A * A
    dA_q = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
    dR_1 = gamma_Iq * I_q + chi * gamma_Aq * A_q
    dR_2 = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

    return [dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2]


def plot_graph(file_name, rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, chi, N_e, z_1, z_2, a, b):
    sol = odeint(model, y0, t, args=(rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_A, gamma_Aq, eta, mu, chi,
                                     N_e, z_1, z_2, a, b))
    # sol = odeint(model_2, y0, t, args=(rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_A, gamma_Aq, gamma_Iq,
    #                                  eta, mu, chi, N_e))
    plt.plot(t, sol[:, 3], '--g', label='Pre_Inf_q')
    plt.plot(t, y_data.now_confirm, 'g', label='Real_Inf_q')
    plt.plot(t, sol[:, 5], '--r', label='Pre_Asy_q')
    plt.plot(t, y_data.now_asy, 'r', label='Real_Asy_q')
    plt.plot(t, sol[:, 6], '--y', label='Pre_Removed_q')
    plt.plot(t, y_data.heal, 'y', label='Real_Removed_q')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.savefig("./img/pic-"+file_name+".png")
    plt.show()
    # plt.savefig("./img/pic-{}.png".format(round))



# 种群染色体矩阵(Chrom)
# 种群表现型矩阵(Phen)
# 种群个体违反约束程度矩阵(CV)
# 种群适应度(FitnV)
def aim(Phen, CV):
    rho = Phen[:, [0]]
    phi = Phen[:, [1]]
    beta = Phen[:, [2]]
    epsilon = Phen[:, [3]]
    alpha = Phen[:, [4]]
    eta = Phen[:, [5]]
    theta = Phen[:, [6]]
    mu = Phen[:, [7]]
    gamma_I = Phen[:, [8]]
    gamma_A = Phen[:, [9]]
    f = []

    for rho_x, phi_x, beta_x, epsilon_x, alpha_x, eta_x, theta_x, mu_x, gamma_I_x, gamma_A_x in \
            zip(rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A):
        # 计算目标函数值
        sol = odeint(model, y0, t, args=(rho_x, phi_x, epsilon_x, beta_x, alpha_x, theta_x,
                                         gamma_I_x, gamma_A_x, gamma_Aq, eta_x, mu_x, chi, N_e[region], z_1, z_2, a, b))
        I_q = sol[:, 3]
        A_q = sol[:, 5]
        R_q = sol[:, 6]

        loss1 = loss_eva(rmse_loss, I_q, y_data.now_confirm.to_numpy())
        loss2 = loss_eva(rmse_loss, A_q, y_data.now_asy.to_numpy())
        loss3 = loss_eva(rmse_loss, R_q, y_data.heal.to_numpy())
        loss = np.mean([loss1, loss2, loss3])
        # loss = np.mean([loss1, loss3])
        f.append([loss])
        # print(f)
    f = np.array(f)
    return f, CV  # 返回目标函数值矩阵


def write_param(log_file: _io.TextIOWrapper):
    log_file.writelines(region)
    temp_str = "rho: " + str(rho) + "\n"
    temp_str += "phi: ", str(phi) + "\n"
    temp_str += "beta: ", str(beta) + "\n"
    temp_str += "epsilon: ", str(epsilon) + "\n"
    temp_str += "alpha: ", str(alpha) + "\n"
    temp_str += "eta: ", str(eta) + "\n"
    temp_str += "theta: ", str(theta) + "\n"
    temp_str += "mu: ", str(mu) + "\n"
    temp_str += "gamma_I: ", str(gamma_I) + "\n"
    temp_str += "gamma_A: ", str(gamma_A) + "\n"
    temp_str += "gamma_Aq: ", str(gamma_Aq) + "\n"
    temp_str += "chi: ", str(chi) + "\n"
    temp_str += "N_e: ", str(N_e) + "\n"
    temp_str += "z_1: ", str(z_1) + "\n"
    temp_str += "z_2: ", str(z_2) + "\n"
    temp_str += "a: ", str(a) + "\n"
    temp_str += "b: ", str(b) + "\n"
    log_file.writelines(temp_str)


def start_GA():
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
    hour = temp_t.tm_hour
    minute = temp_t.tm_min
    log_file_name = model_name + '-'
    log_file_name += region + '-'
    log_file_name += str(MAXGEN) + '-'
    log_file_name += str(int(obj_trace[best_gen, 1])) + '-'
    log_file_name += str(hour) + '_' + str(minute)

    ea.trcplot(obj_trace, [['种群个体平均目标函数值', '种群最优个体目标函数值']],
               save_path="./img/track"+log_file_name+' ')  # 绘制图像

    with open("./log/" + log_file_name + ".txt", mode='w', encoding="utf-8") as log_file:
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

    plot_graph(log_file_name, variable[0, 0], variable[0, 1], variable[0, 2], variable[0, 3], variable[0, 4],
               variable[0, 5], variable[0, 6], variable[0, 7], variable[0, 8], variable[0, 9], gamma_Aq, chi,
               N_e[region], z_1, z_2, a, b)


for i in range(1, 6):
    gamma_Aq = i * 0.01
    start_GA()
