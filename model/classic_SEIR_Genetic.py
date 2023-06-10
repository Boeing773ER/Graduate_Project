import _io
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import geatpy as ea
import time
from functions import calc_days, read_file, loss_eva, rmse_loss, mse_loss, plot_graph


model_name = "Classic_SEIR"
file_path = "../CN_COVID_data/domestic_data.csv"
region = "上海"
start_date = "2022-03-10"
end_date = "2022-04-17"
plot_end_date = "2022-05-03"
"""start_date = "2022-03-20"
end_date = "2022-04-10"
plot_end_date = "2022-04-25" """
days = calc_days(start_date, end_date) - 2

y0 = [0, 646, 4067]
# y0 = [0, 548, 4472]
t = np.linspace(0, days, days + 1)

phi = 3.696e-8
epsilon = 0.5
alpha = 0.2
gamma_I = 0.1
N_e = {"上海": 2.489e7, "湖北": 5.830e7}


""" ===========变量设置==========="""
params_count = 4
x1 = [0, 1]
x2 = [0, 1]  # 第一个决策变量范围
x3 = [0, 1]
x4 = [0, 1]

b1 = [0, 0]  # 第一个决策变量边界，1表示包含范围的边界，0表示不包含
b2 = [0, 0]
b3 = [0, 0]
b4 = [0, 0]

ranges = np.vstack([x1, x2, x3, x4]).T  # 生成自变量的范围矩阵，使得第一行为所有决策变量的下界，第二行为上界
borders = np.vstack([b1, b2, b3, b4]).T  # 生成自变量的边界矩阵
varTypes = np.array(np.zeros(params_count))  # 决策变量的类型，0表示连续，1表示离散

""" ===========染色体编码设置==========="""
Encoding = 'BG'  # 表示采用“实整数编码”，即变量可以是连续的也可以是离散的
codes = np.zeros(params_count)  # 决策变量的编码方式，0表示决策变量使用二进制编码
precisions = []
for i in range(params_count):
    precisions.append(8)  # 决策变量的编码精度，表示二进制编码串解码后能表示的决策变量的精度可达到小数点后6位
scales = np.zeros(params_count)  # 0表示采用算术刻度，1表示采用对数刻度

FieldD = ea.crtfld(Encoding, varTypes, ranges, borders, precisions, codes, scales)

""" ===========遗传算法参数设置==========="""
NIND = 100  # 种群个体数目
MAXGEN = 200  # 最大遗传代数
maxormins = np.array([1])  # 1：目标函数最小化，-1：目标函数最大化
select_style = 'rws'  # 轮盘赌选择
rec_style = 'xovdp'  # 两点交叉
mut_style = 'mutbin'  # 二进制染色体的变异算子
Lind = int(np.sum(FieldD[0, :]))  # 染色体长度
pc = 0.4  # 交叉概率
pm = 1 / Lind  # 变异概率
obj_trace = np.zeros((MAXGEN, 2))
var_trace = np.zeros((MAXGEN, int(Lind)))


y_data = read_file(file_path, region, start_date, end_date)


def model(y, t, phi, epsilon, alpha, gamma_I, N_e):
    E, I, R = y
    dE = phi * (I + epsilon * E) * (N_e - E - I - R) - alpha * E
    dI = alpha * E - gamma_I * I
    dR = gamma_I * I
    return dE, dI, dR


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
    # gamma_Iq = Phen[:, [11]]


    # rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, gamma_Iq, chi, N_e


    # for rho_x, phi_x, beta_x, epsilon_x, alpha_x, eta_x, theta_x, mu_x, gamma_I_x, gamma_A_x, gamma_Aq_x, gamma_Iq_x in \
    #         zip(rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, gamma_Iq):
    #     # 计算目标函数值
    #     sol = odeint(model, y0, t, args=(rho_x[0], phi_x[0], beta_x[0], epsilon_x[0], alpha_x[0], eta_x[0], theta_x[0],
    #                                      mu_x[0], gamma_I_x[0], gamma_A_x[0], gamma_Aq_x[0], gamma_Iq_x[0], chi, N_e[region]))
    f = []
    for phen in Phen:
        # 计算目标函数值
        sol = odeint(model, y0, t, args=(*phen, N_e[region]))
        I_q = sol[:, 1]
        R_q = sol[:, 2]

        loss1 = loss_eva(rmse_loss, I_q, y_data.now_confirm.to_numpy())
        loss3 = loss_eva(rmse_loss, R_q, y_data.heal.to_numpy())
        loss = np.mean([loss1, loss3])
        # loss = loss1
        f.append([loss])
    f = np.array(f)
    return f, CV  # 返回目标函数值矩阵


def write_param(log_file: _io.TextIOWrapper):
    log_file.writelines(region)
    temp_str = "\ninit setting:\n"
    temp_str += "y0: " + str(y0) + "\n"
    temp_str += "start date: " + str(start_date) + "\n"
    temp_str += "end date: " + str(end_date) + "\n"
    temp_str += "phi: " + str(phi) + "\n"
    temp_str += "epsilon: " + str(epsilon) + "\n"
    temp_str += "alpha: " + str(alpha) + "\n"
    temp_str += "gamma_I: " + str(gamma_I) + "\n"
    temp_str += "N_e: " + str(N_e) + "\n"
    log_file.writelines(temp_str)


def draw_result(file_path, log_file_name, region, start_date, end_date, variable):
    y_data = read_file(file_path, region, start_date, end_date)
    days = calc_days(start_date, end_date) - 2
    t = np.linspace(0, days, days + 1)
    param = variable[0, :]
    sol = odeint(model, y0, t, args=(*param, N_e[region]))

    plt.title(model_name + " COVID " + region)
    plt.plot(t, sol[:, 0], '--g', label='Pre_Inf_q')
    plt.plot(t, y_data.now_confirm, 'g', label='Real_Inf_q')
    plt.plot(t, sol[:, 1], '--y', label='Pre_Removed_q')
    plt.plot(t, y_data.heal, 'y', label='Real_Removed_q')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.savefig("../img/pic-" + log_file_name + ".png")
    plt.show()


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
    np.savetxt("../log/obj_trace_" + log_file_name + ".csv", obj_trace, delimiter=',')

    with open("../log/" + log_file_name + ".txt", mode='w', encoding="utf-8") as log_file:
        write_param(log_file)
        temp_str = '最优解的目标函数值：' + str(obj_trace[best_gen, 1])
        print(temp_str)
        log_file.writelines(temp_str + "\n")
        variable = ea.bs2ri(var_trace[[best_gen], :], FieldD)  # 解码得到表现型（即对应的决策变量值）
        print('最优解的决策变量值为：')
        log_file.writelines('最优解的决策变量值为：' + "\n")
        var_name = ["rho", "phi", "beta", "epsilon", "alpha", "eta", "theta", "mu", "gamma_I", "gamma_A", "gamma_Aq",
                    "gamma_Iq"]
        for i in range(variable.shape[1]):
            temp_str = var_name[i] + ': ' + str(variable[0, i])
            print(temp_str)
            log_file.writelines(temp_str + "\n")
        log_file.writelines("\n")
        print('用时：', end_time - start_time, '秒')

    param = variable[0, :]
    sol = odeint(model, y0, t, args=(*param, N_e[region]))

    np.savetxt("../log/" + log_file_name + ".csv", sol, delimiter=',', header="E, Eq, I, Iq, A, Aq, R1, R2",
               comments="")

    # draw_result(file_path, log_file_name, region, start_date, plot_end_date, variable)
    draw_result(file_path, log_file_name, region, start_date, end_date, variable)


for i in range(1):
    # pc = i*0.1
    start_GA(i)

