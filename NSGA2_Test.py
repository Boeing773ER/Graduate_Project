# """Demo.
#
# min f1 = -25 * (x1 - 2)**2 - (x2 - 2)**2 - (x3 - 1)**2 - (x4 - 4)**2 - (x5 - 1)**2
# min f2 = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x4 - 1)**2 + (x5 - 1)**2
# s.t.
# x1 + x2 >= 2
# x1 + x2 <= 6
# x1 - x2 >= -2
# x1 - 3*x2 <= 2
# 4 - (x3 - 3)**2 - x4 >= 0
# (x5 - 3)**2 + x4 - 4 >= 0
# x1,x2,x3,x4,x5 ∈ {0,1,2,3,4,5,6,7,8,9,10}
# """
# import numpy as np
#
# import geatpy as ea
#
#
# class MyProblem(ea.Problem):  # 继承Problem父类
#
#     def __init__(self, M=2):
#         name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
#         Dim = 5  # 初始化Dim（决策变量维数）
#         maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
#         varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
#         lb = [0] * Dim  # 决策变量下界
#         ub = [10] * Dim  # 决策变量上界
#         lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
#         ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
#         # 调用父类构造方法完成实例化
#         ea.Problem.__init__(self,
#                             name,
#                             M,
#                             maxormins,
#                             Dim,
#                             varTypes,
#                             lb,
#                             ub,
#                             lbin,
#                             ubin)
#
#     def evalVars(self, Vars):  # 目标函数
#         x1 = Vars[:, [0]]
#         x2 = Vars[:, [1]]
#         x3 = Vars[:, [2]]
#         x4 = Vars[:, [3]]
#         x5 = Vars[:, [4]]
#         f1 = -25 * (x1 - 2)**2 - (x2 - 2)**2 - (x3 - 1)**2 - (x4 - 4)**2 - (
#             x5 - 1)**2
#         f2 = (x1 - 1)**2 + (x2 - 1)**2 + (x3 - 1)**2 + (x4 - 1)**2 + (x5
#                                                                       - 1)**2
#         #        # 利用罚函数法处理约束条件
#         #        idx1 = np.where(x1 + x2 < 2)[0]
#         #        idx2 = np.where(x1 + x2 > 6)[0]
#         #        idx3 = np.where(x1 - x2 < -2)[0]
#         #        idx4 = np.where(x1 - 3*x2 > 2)[0]
#         #        idx5 = np.where(4 - (x3 - 3)**2 - x4 < 0)[0]
#         #        idx6 = np.where((x5 - 3)**2 + x4 - 4 < 0)[0]
#         #        exIdx = np.unique(np.hstack([idx1, idx2, idx3, idx4, idx5, idx6])) # 得到非可行解的下标
#         #        f1[exIdx] = f1[exIdx] + np.max(f1) - np.min(f1)
#         #        f2[exIdx] = f2[exIdx] + np.max(f2) - np.min(f2)
#         # 利用可行性法则处理约束条件
#         CV = np.hstack([
#             2 - x1 - x2,
#             x1 + x2 - 6,
#             -2 - x1 + x2,
#             x1 - 3 * x2 - 2, (x3 - 3)**2 + x4 - 4,
#             4 - (x5 - 3)**2 - x4
#         ])
#         f = np.hstack([f1, f2])
#         return f, CV
#
#
# if __name__ == '__main__':
#     # 实例化问题对象
#     problem = MyProblem()
#     # 构建算法
#     algorithm = ea.moea_NSGA2_templet(
#         problem,
#         ea.Population(Encoding='BG', NIND=50),
#         MAXGEN=200,  # 最大进化代数
#         logTras=0)  # 表示每隔多少代记录一次日志信息，0表示不记录。
#     algorithm.mutOper.Pm = 0.2  # 修改变异算子的变异概率
#     algorithm.recOper.XOVR = 0.9  # 修改交叉算子的交叉概率
#     # 求解
#     res = ea.optimize(algorithm,
#                       verbose=False,
#                       drawing=1,
#                       outputMsg=True,
#                       drawLog=False,
#                       saveFlag=False)
#     print(res)
import math
from matplotlib import pyplot as plt
import numpy as np
import geatpy as ea
from scipy.integrate import odeint
from model import functions
from model.functions import calc_days, read_file, loss_eva, rmse_loss, mse_loss, plot_graph

from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool

"""==========问题类定义=========="""


class SIRModel(ea.Problem):  # 继承Problem父类
    file_path = "./CN_COVID_data/domestic_data.csv"
    region = "上海"
    start_date = "2022-03-10"
    end_date = "2022-04-17"
    days = calc_days(start_date, end_date) - 2
    y0 = [1, 0, 0, 0, 0, 0]
    t = np.linspace(0, days, days + 1)
    y_data = read_file(file_path, region, start_date, end_date)

    rho = 0.85
    phi = 3.696e-5
    beta = 0.4
    eta = 0.75
    theta = 0.75
    mu = 0.2
    gamma_I = 7e-4
    gamma_A = 1e-4
    gamma_Aq = 0.03
    gamma_Iq = 0.05
    N_e = {"上海": 2.489e7, "湖北": 5.830e7}

    def __init__(self, PoolType):
        name = 'SIR'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 10  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
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

    def model(self, y, t, rho, phi, beta, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, gamma_Iq, N_e):
        I, I_q, A, A_q, R_1, R_2 = y

        dI = (1 - rho) * phi * (I + beta * A) * eta * (N_e - I - I_q - A - A_q - R_1 - R_2) - theta * I - gamma_I * I
        dI_q = rho * phi * (I + beta * A) * eta * (N_e - I - I_q - A - A_q - R_1 - R_2) + theta * I - gamma_Iq * I_q
        dA = (1 - rho) * phi * (I + beta * A) * (1 - eta) * (N_e - I - I_q - A - A_q - R_1 - R_2) - mu * A - gamma_A * A
        dA_q = rho * phi * (I + beta * A) * (1 - eta) * (N_e - I - I_q - A - A_q - R_1 - R_2) + mu * A - gamma_Aq * A_q
        dR_1 = gamma_Iq * I_q + gamma_Aq * A_q
        dR_2 = gamma_A * A + gamma_I * I

        return dI, dI_q, dA, dA_q, dR_1, dR_2

    def aimFunc(self, pop):  # 目标函数
        vars = pop.Phen
        rho = vars[:, [0]]
        phi = vars[:, [1]]
        beta = vars[:, [2]]
        eta = vars[:, [3]]
        theta = vars[:, [4]]
        mu = vars[:, [5]]
        gamma_I = vars[:, [6]]
        gamma_A = vars[:, [7]]
        gamma_Aq = vars[:, [8]]
        gamma_Iq = vars[:, [9]]
        loss1 = []
        loss2 = []
        loss3 = []
        # rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, chi, N_e, z_1, z_2, a, b
        for rho_x, phi_x, beta_x, eta_x, theta_x, mu_x, gamma_I_x, gamma_A_x, gamma_Aq_x, gamma_Iq_x in \
                zip(rho, phi, beta, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, gamma_Iq):
            # 计算目标函数值
            sol = odeint(self.model, self.y0, self.t,
                         args=(rho_x[0], phi_x[0], beta_x[0], eta_x[0], theta_x[0], mu_x[0], gamma_I_x[0], gamma_A_x[0],
                               gamma_Aq_x[0], gamma_Iq_x[0], self.N_e[self.region]))
            loss1.append(loss_eva(rmse_loss, sol[:, 1], self.y_data.now_confirm.to_numpy()))  # I_q
            loss2.append(loss_eva(rmse_loss, sol[:, 3], self.y_data.now_asy.to_numpy()))  # A_q
            loss3.append(loss_eva(rmse_loss, sol[:, 4], self.y_data.heal.to_numpy()))  # R_q
        pop.ObjV = np.array([loss1, loss2, loss3]).T

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值）,这个函数其实可以不要
        referenceObjV = np.array([[0, 0, 0]])
        return referenceObjV

    # def aimFunc(self, pop):  # 目标函数
    #     x1 = pop.Phen[:, [0]]
    #     x2 = pop.Phen[:, [1]]
    #     x3 = pop.Phen[:, [2]]
    #     x4 = pop.Phen[:, [3]]
    #     x5 = pop.Phen[:, [4]]
    #     x6 = pop.Phen[:, [5]]
    #     x7 = pop.Phen[:, [6]]
    #     x8 = pop.Phen[:, [7]]
    #     x9 = pop.Phen[:, [8]]
    #     x10 = pop.Phen[:, [9]]
    #
    #     pop.ObjV = np.zeros((pop.Phen.shape[0], self.M))
    #
    #     pop.ObjV[:, [0]] = x1 ** 4 - 10 * x1 ** 2 + x1 * x2 + x2 ** 4 - x1 ** 2 * x2 ** 2
    #     pop.ObjV[:, [1]] = x2 ** 4 - x1 ** 2 * x2 ** 2 + x1 ** 4 + x1 * x2
    #     pop.ObjV[:, [2]] = x3 ** 4 - x1 ** 2 * x2 ** 2 + x1 ** 4 + x1 * x2
    #     print(pop.ObjV)


class SEIRModel(ea.Problem):  # 继承Problem父类
    file_path = "./CN_COVID_data/domestic_data.csv"
    region = "上海"
    start_date = "2022-03-10"
    end_date = "2022-04-17"
    days = calc_days(start_date, end_date) - 2
    y0 = [0, 0, 1, 0, 0, 0, 0, 0]
    t = np.linspace(0, days, days + 1)
    y_data = read_file(file_path, region, start_date, end_date)

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
    gamma_Iq = 0.05
    chi = 1
    N_e = {"上海": 2.489e7, "湖北": 5.830e7}

    def __init__(self, PoolType):
        name = 'SEIR'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 12  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * Dim  # 决策变量上界
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

    def model(self, y, t, rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, gamma_Iq, chi,
              N_e):
        E, E_q, I, I_q, A, A_q, R_1, R_2 = y
        dE = (1 - rho) * phi * (I + epsilon * E + beta * A) * (
                N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E
        dE_q = rho * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E_q
        dI = alpha * eta * E - theta * I - gamma_I * I
        dI_q = alpha * eta * E_q + theta * I - gamma_Iq * I_q
        dA = alpha * (1 - eta) * E - mu * A - gamma_A * A
        dA_q = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
        dR_1 = gamma_Iq * I_q + chi * gamma_Aq * A_q
        dR_2 = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

        return dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2

    def aimFunc(self, pop):  # 目标函数
        loss1 = []
        loss2 = []
        loss3 = []
        for phen in pop.Phen:
            # 计算目标函数值
            sol = odeint(self.model, self.y0, self.t, args=(*phen, self.chi, self.N_e[self.region]))
            loss1.append(loss_eva(rmse_loss, sol[:, 3], self.y_data.now_confirm.to_numpy()))  # I_q
            loss2.append(loss_eva(rmse_loss, sol[:, 5], self.y_data.now_asy.to_numpy()))  # A_q
            loss3.append(loss_eva(rmse_loss, sol[:, 6], self.y_data.heal.to_numpy()))  # R_q
        pop.ObjV = np.array([loss1, loss2, loss3]).T

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值）,这个函数其实可以不要
        referenceObjV = np.array([[0, 0, 0]])
        return referenceObjV


class SEAIRModel(ea.Problem):  # 继承Problem父类
    file_path = "./CN_COVID_data/domestic_data.csv"
    region = "上海"
    start_date = "2022-03-10"
    end_date = "2022-04-17"
    days = calc_days(start_date, end_date) - 2
    y0 = [0, 0, 1, 0, 0, 0, 0, 0]
    t = np.linspace(0, days, days + 1)
    y_data = read_file(file_path, region, start_date, end_date)

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
    gamma_Iq = 0.05
    chi = 1
    N_e = {"上海": 2.489e7, "湖北": 5.830e7}

    def __init__(self, PoolType):
        name = 'SEAIR'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 15  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0] * Dim  # 决策变量下界
        ub = [1] * (Dim-2) + [100, 10]  # 决策变量上界
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

    def model(self, y, t, rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, z_1, z_2, a, b,
              chi, N_e,):
        E, E_q, I, I_q, A, A_q, R_1, R_2 = y

        gamma_Iq = z_1 + z_2 * math.tanh((t - a) / b)

        dE = (1 - rho) * phi * (I + epsilon * E + beta * A) * (
                    N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E
        dE_q = rho * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E_q
        dI = alpha * eta * E - theta * I - gamma_I * I
        dI_q = alpha * eta * E_q + theta * I - gamma_Iq * I_q
        dA = alpha * (1 - eta) * E - mu * A - gamma_A * A
        dA_q = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
        dR_1 = gamma_Iq * I_q + chi * gamma_Aq * A_q
        dR_2 = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

        return dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2

    def aimFunc(self, pop):  # 目标函数
        loss1 = []
        loss2 = []
        loss3 = []
        for phen in pop.Phen:
            # 计算目标函数值
            sol = odeint(self.model, self.y0, self.t, args=(*phen, self.chi, self.N_e[self.region]))
            loss1.append(loss_eva(rmse_loss, sol[:, 3], self.y_data.now_confirm.to_numpy()))  # I_q
            loss2.append(loss_eva(rmse_loss, sol[:, 5], self.y_data.now_asy.to_numpy()))  # A_q
            loss3.append(loss_eva(rmse_loss, sol[:, 6], self.y_data.heal.to_numpy()))  # R_q
        pop.ObjV = np.array([loss1, loss2, loss3]).T

    def calReferObjV(self):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值）,这个函数其实可以不要
        referenceObjV = np.array([[0, 0, 0]])
        return referenceObjV


def plot_graph(sol, model_name, region, t, y_data, num):
    plt.title(model_name + " COVID " + region)
    plt.plot(t, sol[:, num[0]], '--g', label='Pre_Inf_q')
    plt.plot(t, y_data.now_confirm, 'g', label='Real_Inf_q')
    # plt.plot(t, sol[:, num[1]], '--r', label='Pre_Asy_q')
    # plt.plot(t, y_data.now_asy, 'r', label='Real_Asy_q')
    plt.plot(t, sol[:, num[2]], '--y', label='Pre_Removed_q')
    plt.plot(t, y_data.heal, 'y', label='Real_Removed_q')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    # plt.savefig("../img/pic-"+file_name+".png")
    plt.show()



"""==========执行脚本=========="""
if __name__ == '__main__':
    PoolType = "Thread"
    problem = SEIRModel(PoolType)
    # problem = MyProblem()  # 生成问题对象
    # 构建算法
    Encoding = "RI"
    NIND = 100
    # MAXGEN = 600
    MAXGEN = 200
    print(ea.Population(Encoding=Encoding, NIND=NIND).ChromNum)
    algorithm = ea.moea_NSGA2_templet(problem,
                                      ea.Population(Encoding=Encoding, NIND=NIND),
                                      MAXGEN=MAXGEN,  # 最大进化代数。
                                      logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # algorithm.recOper.XOVR = 0.4
    # 求解
    # algorithm.verbose = True
    # algorithm.drawing = 1
    # [BestIndi, population] = algorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    # BestIndi.save()  # 把最优个体的信息保存到文件中
    # print(type(BestIndi))
    # print(BestIndi.Phen[0])
    res = ea.optimize(algorithm, verbose=True, drawing=1, outputMsg=True, drawLog=True, saveFlag=True,
                      dirName=None)
    # print(res["lastPop"].Chrom)
    # temp_args = res["lastPop"].Chrom[0, :]

    file_path = "./CN_COVID_data/domestic_data.csv"
    region = "上海"
    start_date = "2022-03-10"
    end_date = "2022-06-01"
    days = calc_days(start_date, end_date) - 2
    y0 = [0, 0, 1, 0, 0, 0, 0, 0]
    t = np.linspace(0, days, days + 1)
    y_data = read_file(file_path, region, start_date, end_date)

    for i in range(5):
        temp_args = res["lastPop"].Chrom[0, :]
        # temp_args = res["Vars"][i, :]
        sol = odeint(problem.model, y0, t, args=(*temp_args, problem.chi, problem.N_e[region]))
        plot_graph(sol, problem.name, region, t, y_data, [3, 5, 6])

    for i in range(5):
        # temp_args = res["lastPop"].Chrom[0, :]
        temp_args = res["Vars"][i, :]
        sol = odeint(problem.model, y0, t, args=(*temp_args, problem.chi, problem.N_e[region]))
        plot_graph(sol, problem.name, region, t, y_data, [3, 5, 6])

    # sol = odeint(problem.model, problem.y0, problem.t, args=(*temp_args, problem.N_e[problem.region]))
    # plot_graph(sol, problem.name, problem.region, problem.t, problem.y_data, [1, 3, 4])

