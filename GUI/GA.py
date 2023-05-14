import _io
import math
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import geatpy as ea
import time
import datetime as dt


def calc_days(start, end):
    start_date = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(end, "%Y-%m-%d").date()
    return (end_date - start_date).days


def read_file(file_path, start_date, end_date):
    # print(start_date, end_date)
    data_file = pd.read_csv(file_path)

    sub_data = data_file.loc[data_file.province == "上海", :]
    sub_data = sub_data.loc[sub_data.date > start_date, :]
    sub_data = sub_data.loc[end_date > sub_data.date, :]
    ydata = pd.DataFrame()
    ydata["now_confirm"] = sub_data["now_confirm"]
    ydata["heal"] = sub_data["heal"]
    ydata["now_asy"] = sub_data["now_asy"]
    return ydata


def mse_loss(x: np.ndarray, y: np.ndarray):
    # x: prediction, y: real
    # print(len(x), len(y))
    assert len(x) == len(y)
    loss = np.sum(np.square(x - y)) / len(x)
    return loss


def rmse_loss(x: np.ndarray, y: np.ndarray):
    # print(len(x), len(y))
    assert len(x) == len(y)
    loss = np.sqrt(np.sum(np.square(x - y)) / len(x))
    return loss


def loss_eva(function, x: np.ndarray, y: np.ndarray):
    return function(x, y)


def plot_graph(file_name, sol, model_name, t, y_data, num):
    # plt.title(model_name + " COVID " + region)
    plt.title(model_name + " COVID ")
    plt.plot(t, sol[:, num[0]], '--g', label='Pre_Inf_q')
    plt.plot(t, y_data.now_confirm, 'g', label='Real_Inf_q')
    plt.plot(t, sol[:, num[1]], '--r', label='Pre_Asy_q')
    plt.plot(t, y_data.now_asy, 'r', label='Real_Asy_q')
    plt.plot(t, sol[:, num[2]], '--y', label='Pre_Removed_q')
    plt.plot(t, y_data.heal, 'y', label='Real_Removed_q')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.savefig("../img/pic-"+file_name+".png")
    plt.show()


class SEAIRmodel():

    def __init__(self, file_path, region_population, start_date, end_date, test_end_date):
        self.model_name = "SEAIR"
        self.file_path = file_path
        self.start_date = start_date
        self.end_date = end_date
        self.plot_end_date = test_end_date
        days = calc_days(start_date, end_date) - 2

        self.y0 = [0, 0, 1, 0, 0, 0, 0, 0]
        self.t = np.linspace(0, days, days + 1)

        self.rho = 0.85
        self.phi = 3.696e-5
        self.beta = 0.4
        self.epsilon = 0.5
        self.alpha = 0.2
        self.eta = 0.75
        self.theta = 0.75
        self.mu = 0.2
        self.gamma_I = 7e-4
        self.gamma_A = 1e-4
        self.gamma_Aq = 0.03
        self.z_1 = 0.045
        self.z_2 = 0.026
        self.a = 28
        self.b = 5
        self.chi = 0
        self.N_e = region_population

        # """ ===========变量设置==========="""
        #
        # Dim = 15  # 初始化Dim（决策变量维数）
        # varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        # lb = [0] * Dim  # 决策变量下界
        # ub = [1] * (Dim-2) + [100, 10]  # 决策变量上界
        # lbin = [1] * Dim  # 决策变量下边界
        # ubin = [1] * Dim  # 决策变量上边界
        #
        # ranges = np.vstack([lb, ub])
        # borders = np.vstack([lbin, ubin])
        # print(ranges, borders)
        #
        # """ ===========染色体编码设置==========="""
        # self.Encoding = 'BG'  # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
        # codes = np.zeros(Dim)  # 决策变量的编码方式，0表示决策变量使用二进制编码
        # precisions = []
        # for i in range(Dim):
        #     precisions.append(4)  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位
        # scales = np.zeros(Dim)  # 0表示采用算术刻度，1表示采用对数刻度
        #
        # self.FieldD = ea.crtfld(self.Encoding, varTypes, ranges, borders, precisions, codes, scales)
        """ ===========变量设置==========="""
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

        ranges = np.vstack(
            [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
        borders = np.vstack([b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15]).T  # 生成自变量的边界矩阵
        # print(ranges, borders)
        varTypes = np.array(np.zeros(params_count))  # 决策变量的类型，0表示连续，1表示离散

        """ ===========染色体编码设置==========="""
        self.Encoding = 'BG'  # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
        codes = np.zeros(params_count)  # 决策变量的编码方式，0表示决策变量使用二进制编码
        precisions = []
        for i in range(params_count):
            precisions.append(4)  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位
        scales = np.zeros(params_count)  # 0表示采用算术刻度，1表示采用对数刻度

        self.FieldD = ea.crtfld(self.Encoding, varTypes, ranges, borders, precisions, codes, scales)


        """ ===========遗传算法参数设置==========="""
        self.NIND = 100  # 种群个体数目
        self.MAXGEN = 200  # 最大遗传代数
        self.maxormins = np.array([1])  # 1：目标函数最小化，-1：目标函数最大化
        self.select_style = 'rws'  # 轮盘赌选择
        self.rec_style = 'xovdp'  # 两点交叉
        self.mut_style = 'mutbin'  # 二进制染色体的变异算子
        self.Lind = int(np.sum(self.FieldD[0, :]))  # 染色体长度
        self.pc = 0.5  # 交叉概率
        self.pm = 1 / self.Lind  # 变异概率
        self.obj_trace = np.zeros((self.MAXGEN, 2))
        self.var_trace = np.zeros((self.MAXGEN, int(self.Lind)))

        self.y_data = read_file(file_path, start_date, end_date)

    def model(self, y, t, rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, z_1, z_2, a, b, chi, N_e):
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
    def aim(self, Phen, CV):
        f = []
        for phen in Phen:
            # 计算目标函数值
            sol = odeint(self.model, self.y0, self.t, args=(*phen, self.chi, self.N_e))
            I_q = sol[:, 3]
            A_q = sol[:, 5]
            R_q = sol[:, 6]

            loss1 = loss_eva(rmse_loss, I_q, self.y_data.now_confirm.to_numpy())
            loss2 = loss_eva(rmse_loss, A_q, self.y_data.now_asy.to_numpy())
            loss3 = loss_eva(rmse_loss, R_q, self.y_data.heal.to_numpy())
            loss = np.mean([loss1, loss2, loss3])
            # loss = np.mean([loss1, loss3])
            f.append([loss])
        f = np.array(f)
        return f, CV  # 返回目标函数值矩阵

    def write_param(self, log_file: _io.TextIOWrapper):
        # log_file.writelines(self.region)
        temp_str = "\ninit setting:\n"
        temp_str += "rho: " + str(self.rho) + "\n"
        temp_str += "phi: " + str(self.phi) + "\n"
        temp_str += "beta: " + str(self.beta) + "\n"
        temp_str += "epsilon: " + str(self.epsilon) + "\n"
        temp_str += "alpha: " + str(self.alpha) + "\n"
        temp_str += "eta: " + str(self.eta) + "\n"
        temp_str += "theta: " + str(self.theta) + "\n"
        temp_str += "mu: " + str(self.mu) + "\n"
        temp_str += "gamma_I: " + str(self.gamma_I) + "\n"
        temp_str += "gamma_A: " + str(self.gamma_A) + "\n"
        temp_str += "gamma_Aq: " + str(self.gamma_Aq) + "\n"
        temp_str += "chi: " + str(self.chi) + "\n"
        temp_str += "N_e: " + str(self.N_e) + "\n"
        temp_str += "z_1: " + str(self.z_1) + "\n"
        temp_str += "z_2: " + str(self.z_2) + "\n"
        temp_str += "a: " + str(self.a) + "\n"
        temp_str += "b: " + str(self.b) + "\n"
        log_file.writelines(temp_str)

    def draw_result(self, file_path, log_file_name, start_date, end_date, variable):
        y_data = read_file(file_path, start_date, end_date)
        days = calc_days(start_date, end_date) - 2
        # y0 = [0, 0, 1, 0, 0, 0, 0, 0]
        t = np.linspace(0, days, days + 1)
        variable = variable[0, :]
        sol = odeint(self.model, self.y0, t, args=(*variable, self.chi, self.N_e))
        plot_graph(log_file_name, sol, self.model_name, t, y_data, [3, 5, 6])

    def start_GA(self, iter_round):
        """=========================开始遗传算法进化========================"""
        start_time = time.time()  # 开始计时
        Chrom = ea.crtpc(self.Encoding, self.NIND, self.FieldD)  # 生成种群染色体矩阵
        variable = ea.bs2ri(Chrom, self.FieldD)  # 对初始种群进行解码
        CV = np.zeros((self.NIND, 1))  # 初始化一个CV矩阵（此时因为未确定个体是否满足约束条件，因此初始化元素为0，暂认为所有个体是可行解个体）
        ObjV, CV = self.aim(variable, CV)  # 计算初始种群个体的目标函数值
        FitnV = ea.ranking(ObjV, CV, self.maxormins)  # 根据目标函数大小分配适应度值
        best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
        # 开始进化
        for gen in range(self.MAXGEN):
            SelCh = Chrom[ea.selecting(self.select_style, FitnV, self.NIND - 1), :]  # 选择
            SelCh = ea.recombin(self.rec_style, SelCh, self.pc)  # 重组
            SelCh = ea.mutate(self.mut_style, self.Encoding, SelCh, self.pm)  # 变异
            # 把父代精英个体与子代的染色体进行合并，得到新一代种群
            Chrom = np.vstack([Chrom[best_ind, :], SelCh])
            Phen = ea.bs2ri(Chrom, self.FieldD)  # 对种群进行解码(二进制转十进制)
            ObjV, CV = self.aim(Phen, CV)  # 求种群个体的目标函数值
            FitnV = ea.ranking(ObjV, CV, self.maxormins)  # 根据目标函数大小分配适应度值
            # 记录
            best_ind = np.argmax(FitnV)  # 计算当代最优个体的序号
            self.obj_trace[gen, 0] = np.sum(ObjV) / ObjV.shape[0]  # 记录当代种群的目标函数均值
            self.obj_trace[gen, 1] = ObjV[best_ind]  # 记录当代种群最优个体目标函数值
            self.var_trace[gen, :] = Chrom[best_ind, :]  # 记录当代种群最优个体的染色体
            print(time.ctime())
            print("Gen:", gen)
            print(ObjV[best_ind])
        # 进化完成
        end_time = time.time()  # 结束计时

        """============================输出结果============================"""
        best_gen = np.argmin(self.obj_trace[:, [1]])
        print("最优解代数：", best_gen)

        temp_t = time.localtime()
        day = temp_t.tm_mday
        hour = temp_t.tm_hour
        minute = temp_t.tm_min
        log_file_name = str(iter_round) + '-' + self.model_name + '-'
        # log_file_name += region + '-'
        log_file_name += str(self.MAXGEN) + '-'
        log_file_name += str(int(self.obj_trace[best_gen, 1])) + '-'
        log_file_name += str(day) + '_' + str(hour) + '_' + str(minute)

        ea.trcplot(self.obj_trace, [['Average value of population', 'Population optimal individual value']],
                   save_path="../img/track-"+log_file_name+' ')  # 绘制图像
        np.savetxt("../log/obj_trace_"+log_file_name+".csv", self.obj_trace, delimiter=',')

        with open("../log/" + log_file_name + ".txt", mode='w', encoding="utf-8") as log_file:
            self.write_param(log_file)
            temp_str = '最优解的目标函数值：' + str(self.obj_trace[best_gen, 1])
            print(temp_str)
            log_file.writelines(temp_str + "\n")
            variable = ea.bs2ri(self.var_trace[[best_gen], :], self.FieldD)  # 解码得到表现型（即对应的决策变量值）
            print('最优解的决策变量值为：')
            log_file.writelines('最优解的决策变量值为：' + "\n")
            for i in range(variable.shape[1]):
                temp_str = 'x' + str(i) + '=' + str(variable[0, i])
                print(temp_str)
                log_file.writelines(temp_str + "\n")
            log_file.writelines("\n")
            print('用时：', end_time - start_time, '秒')

        variables = variable[0, :]
        sol = odeint(self.model, self.y0, self.t, args=(*variables, self.chi, self.N_e))
        # 保存数据值csv
        np.savetxt("../log/" + log_file_name + ".csv", sol, delimiter=',', header="E, Eq, I, Iq, A, Aq, R1, R2",
                   comments="")

        self.draw_result(self.file_path, log_file_name, self.start_date, self.plot_end_date, variable)


file_path = "../CN_COVID_data/domestic_data.csv"
region_population = 2.489e7
start_date = "2022-03-10"
end_date = "2022-04-17"
test_end_date = "2022-06-17"
model = SEAIRmodel(file_path, region_population, start_date, end_date, test_end_date)
model.start_GA(1)
