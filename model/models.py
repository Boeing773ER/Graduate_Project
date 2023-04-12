import math
import numpy as np
import time
import geatpy as ea
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from functions import calc_days, read_file, loss_eva, rmse_loss, mse_loss, plot_graph
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool


class SEAIRmodel(ea.Problem):  # 继承Problem父类
    file_path = "../CN_COVID_data/domestic_data.csv"
    region = "上海"
    start_date = "2022-03-10"
    end_date = "2022-04-17"
    days = calc_days(start_date, end_date) - 2
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
    chi = 0
    N_e = {"上海": 2.489e7, "湖北": 5.830e7}
    z_1 = 0.045
    z_2 = 0.026
    a = 28
    b = 5

    y_data = read_file(file_path, region, start_date, end_date)

    def __init__(self, PoolType):
        name = 'SEAIR_Model'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）

        Dim = 15  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * (Dim - 2) + [100, 10]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(2)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            self.pool = ProcessPool(num_cores)  # 设置池的大小

    def model(self, y, t, rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, chi, N_e,
              z_1, z_2, a, b):
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

    def aimFunc(self, pop):  # 目标函数
        Phen = pop.Phen
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
        gamma_Aq = Phen[:, [10]]
        z_1 = Phen[:, [11]]
        z_2 = Phen[:, [12]]
        a = Phen[:, [13]]
        b = Phen[:, [14]]
        loss = []

        # rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, chi, N_e, z_1, z_2, a, b
        for rho_x, phi_x, beta_x, epsilon_x, alpha_x, eta_x, theta_x, mu_x, gamma_I_x, gamma_A_x, gamma_Aq_x, z_1_x, z_2_x, a_x, b_x \
                in zip(rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, z_1, z_2, a, b):
            # 计算目标函数值
            sol = odeint(self.model, self.y0, self.t,
                         args=(rho_x[0], phi_x[0], beta_x[0], epsilon_x[0], alpha_x[0], eta_x[0], theta_x[0], mu_x[0],
                               gamma_I_x[0], gamma_A_x[0], gamma_Aq[0], self.chi, self.N_e[self.region], z_1[0], z_2[0],
                               a[0], b[0]))

            loss1 = loss_eva(rmse_loss, sol[:, 3], self.y_data.now_confirm.to_numpy())  # I_q
            loss2 = loss_eva(rmse_loss, sol[:, 5], self.y_data.now_asy.to_numpy())  # A_q
            loss3 = loss_eva(rmse_loss, sol[:, 6], self.y_data.heal.to_numpy())  # R_q
            loss.append([loss1, loss2, loss3])

        pop.ObjV = np.array(loss).T


def get_field():
    params_count = 15
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
    b11 = [1, 1]
    b12 = [1, 1]
    b13 = [1, 1]
    b14 = [1, 1]
    b15 = [1, 1]

    # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
    ranges = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15]).T
    # 生成自变量的边界矩阵
    borders = np.vstack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15]).T
    varTypes = np.array(np.zeros(params_count))  # 决策变量的类型，0表示连续，1表示离散
    Encoding = 'RI'  # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
    codes = np.zeros(params_count)  # 决策变量的编码方式，0表示决策变量使用二进制编码
    precisions = []
    for i in range(params_count):
        precisions.append(4)  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位
    scales = np.zeros(params_count)  # 0表示采用算术刻度，1表示采用对数刻度
    # return ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)
    # return ea.crtfld(Encoding, varTypes, ranges, borders)
    return varTypes, ranges, borders


if __name__ == '__main__':
    PoolType = "Thread"
    problem = SEAIRmodel(PoolType)
    Encoding = 'RI'
    NIND = 100
    Field = get_field()
    print(Field)
    population = ea.Population(Encoding, Field, NIND)
    # population = ea.Population(Encoding, NIND)
    algorithm = ea.moea_psy_NSGA2_templet(problem, population)
    algorithm.MAXGEN = 1000
    algorithm.trappedValue = 1e-6
    algorithm.maxTrappedCount = 100
    algorithm.logTras = 1
    algorithm.verbose = True
    algorithm.drawing = 1
    [BestIndi, population] = algorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    BestIndi.save()  # 把最优个体的信息保存到文件中
    """=================================输出结果=============================="""
    print('评价次数：%s' % algorithm.evalsNum)
    print('时间已过 %s 秒' % algorithm.passTime)
    # if BestIndi.sizes != 0:
    #     print('最优的目标函数值为：%s' % (BestIndi.ObjV[0][0]))
    #     print('最优的控制变量值为：')
    #     for i in range(BestIndi.Phen.shape[1]):
    #         print(BestIndi.Phen[0, i])
    #     """=================================检验结果==============================="""
    #     problem.test(C=BestIndi.Phen[0, 0], G=BestIndi.Phen[0, 1])
    # else:
    #     print('没找到可行解。')


#
# """ ===========遗传算法参数设置==========="""
# NIND = 100  # 种群个体数目
# MAXGEN = 5  # 最大遗传代数
# maxormins = np.array([1])  # 1：目标函数最小化，-1：目标函数最大化
# select_style = 'rws'  # 轮盘赌选择
# rec_style = 'xovdp'  # 两点交叉
# mut_style = 'mutbin'  # 二进制染色体的变异算子
# Lind = int(np.sum(FieldD[0, :]))  # 染色体长度
# pc = 0.5  # 交叉概率
# pm = 1 / Lind  # 变异概率
# obj_trace = np.zeros((MAXGEN, 2))
# var_trace = np.zeros((MAXGEN, int(Lind)))
#
# y_data = read_file(file_path, region, start_date, end_date)
#
#
# # 种群染色体矩阵(Chrom)
# # 种群表现型矩阵(Phen)
# # 种群个体违反约束程度矩阵(CV)
# # 种群适应度(FitnV)
#
#
#
# def start_GA(iter_round):
#     """=========================开始遗传算法进化========================"""
#     start_time = time.time()  # 开始计时
#     Chrom = ea.crtpc(Encoding, NIND, FieldD)  # 生成种群染色体矩阵
#     variable = ea.bs2ri(Chrom, FieldD)  # 对初始种群进行解码
#     CV = np.zeros((NIND, 1))  # 初始化一个CV矩阵（此时因为未确定个体是否满足约束条件，因此初始化元素为0，暂认为所有个体是可行解个体）
#     ObjV, CV = aim(variable, CV)  # 计算初始种群个体的目标函数值
#     FitnV = ea.ranking(ObjV, CV, maxormins)  # 根据目标函数大小分配适应度值
#     best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
#     # 开始进化
#     for gen in range(MAXGEN):
#         SelCh = Chrom[ea.selecting(select_style, FitnV, NIND - 1), :]  # 选择
#         SelCh = ea.recombin(rec_style, SelCh, pc)  # 重组
#         SelCh = ea.mutate(mut_style, Encoding, SelCh, pm)  # 变异
#         # 把父代精英个体与子代的染色体进行合并，得到新一代种群
#         Chrom = np.vstack([Chrom[best_ind, :], SelCh])
#         Phen = ea.bs2ri(Chrom, FieldD)  # 对种群进行解码(二进制转十进制)
#         ObjV, CV = aim(Phen, CV)  # 求种群个体的目标函数值
#         FitnV = ea.ranking(ObjV, CV, maxormins)  # 根据目标函数大小分配适应度值
#         # 记录
#         best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
#         obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]  # 记录当代种群的目标函数均值
#         obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
#         var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体
#         print(time.ctime())
#         print("Gen:", gen)
#         print(ObjV[best_ind])
#     # 进化完成
#     end_time = time.time()  # 结束计时
#
#     """============================输出结果============================"""
#     best_gen = np.argmin(obj_trace[:, [1]])
#     print("最优解代数：", best_gen)
#
#     temp_t = time.localtime()
#     day = temp_t.tm_mday
#     hour = temp_t.tm_hour
#     minute = temp_t.tm_min
#     log_file_name = str(iter_round) + '-' + model_name + '-'
#     log_file_name += region + '-'
#     log_file_name += str(MAXGEN) + '-'
#     log_file_name += str(int(obj_trace[best_gen, 1])) + '-'
#     log_file_name += str(day) + '_' + str(hour) + '_' + str(minute)
#
#     ea.trcplot(obj_trace, [['Average value of population', 'Population optimal individual value']],
#                save_path="../img/track-"+log_file_name+' ')  # 绘制图像
#     np.savetxt("../log/obj_trace_"+log_file_name+".csv", obj_trace, delimiter=',')
#
#     with open("../log/" + log_file_name + ".txt", mode='w', encoding="utf-8") as log_file:
#         write_param(log_file)
#         temp_str = '最优解的目标函数值：' + str(obj_trace[best_gen, 1])
#         print(temp_str)
#         log_file.writelines(temp_str + "\n")
#         variable = ea.bs2ri(var_trace[[best_gen], :], FieldD)  # 解码得到表现型（即对应的决策变量值）
#         print('最优解的决策变量值为：')
#         log_file.writelines('最优解的决策变量值为：' + "\n")
#         for i in range(variable.shape[1]):
#             temp_str = 'x' + str(i) + '=' + str(variable[0, i])
#             print(temp_str)
#             log_file.writelines(temp_str + "\n")
#         log_file.writelines("\n")
#         print('用时：', end_time - start_time, '秒')
