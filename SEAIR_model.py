"""
dS/dt = - (1 - rho) * phi * (I + epsilon * E + beta * A) * S - rho * phi * (I + epsilon * E + beta * A) * S
dE/dt = (1 - rho) * phi * (I + epsilon * E + beta * A) * S - alpha * E
dE_q/dt = rho * phi * (I + epsilon * E + beta * A) * S - alpha * E_q
dI/dt = alpha * eta * E - theta * I - gamma_I * I
dI_q/dt = alpha * eta * E_q + theta * I - gamma_Iq(t) * I_q
dA/dt = alpha * (1 - eta) * E - mu * A - gamma_A * A
dA_q/dt = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
dR_1/dt = gamma_Iq(t) * I_q + chi * gamma_Aq * A_q
dR_2/dt = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * Aq
"""
import scipy.optimize
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
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
"""

"""
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

file_path = "./CN_COVID_data/domestic_data.csv"
data_file = pd.read_csv(file_path)
sub_data = data_file.loc[data_file.province == "上海", :]
xdata = sub_data["confirm_add"]
# ydata = sub_data["date"]
ydata = np.linspace(0, 20, 21)


def model(y, t, rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_Iq, gamma_A, gamma_Aq, eta, mu, chi, N_e):
    E = y[0]
    E_q = y[1]
    I = y[2]
    I_q = y[3]
    A = y[4]
    A_q = y[5]
    R_1 = y[6]
    R_2 = y[7]
    # rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_Iq, gamma_A, gamma_Aq, eta, mu, chi, N_e = u

    # used a fixed arg for gamma_Iq(t)
    dE = (1 - rho) * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E
    dE_q = rho * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E_q
    dI = alpha * eta * E - theta * I - gamma_I * I
    dI_q = alpha * eta * E_q + theta * I - gamma_Iq * I_q
    dA = alpha * (1 - eta) * E - mu * A - gamma_A * A
    dA_q = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
    dR_1 = gamma_Iq * I_q + chi * gamma_Aq * A_q
    dR_2 = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

    return [dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2]


def model_infected(y, t, E, alpha, eta, theta, gamma_I):
    I = y

    dI = alpha * eta * E - theta * I - gamma_I * I

    return dI


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
gamma_Iq = 0.5
gamma_A = 1e-4
gamma_Aq = 0.03
eta = 0.75
mu = 0.2
chi = 0
N_e = 1e7

sol = odeint(model, y0, t,
             args=(rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_Iq, gamma_A, gamma_Aq, eta, mu, chi, N_e))

sol = odeint(model_infected, 1, t, args=(E, alpha, eta, theta, gamma_I))

# popt, pcov = scipy.optimize.curve_fit(model, xdata, ydata)
popt, pcov = scipy.optimize.curve_fit(model_infected, xdata, ydata)
print(popt)

plt.plot(t, sol[:, 0], 'b', label='Exposed')
plt.plot(t, sol[:, 2], 'g', label='Infected')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

"""
# 自定义函数，curve_fit支持自定义函数的形式进行拟合，这里定义的是指数函数的形式
# 包括自变量x和a，b，c三个参数
def func(x, a, b, c):
    return a * np.exp(-b * x) + c
 
# 产生数据
xdata = np.linspace(0, 4, 50) # x从0到4取50个点
y = func(xdata, 2.5, 1.3, 0.5) # 在x取xdata，a，b，c分别取2.5, 1.3, 0.5条件下，运用自定义函数计算y的值
 
# 在y上产生一些扰动模拟真实数据
np.random.seed(1729)
# 产生均值为0，标准差为1，维度为xdata大小的正态分布随机抽样0.2倍的扰动
y_noise = 0.2 * np.random.normal(size=xdata.size) 
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')
 
# 利用“真实”数据进行曲线拟合
popt, pcov = curve_fit(func, xdata, ydata) # 拟合方程，参数包括func，xdata，ydata，
# 有popt和pcov两个个参数，其中popt参数为a，b，c，pcov为拟合参数的协方差
 
# plot出拟合曲线，其中的y使用拟合方程和xdata求出
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    
 
#     如果参数本身有范围，则可以设置参数的范围，如 0 <= a <= 3,
#     0 <= b <= 1 and 0 <= c <= 0.5:
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5])) # bounds为限定a，b，c参数的范围
 
plt.plot(xdata, func(xdata, *popt), 'g--',
              label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
"""
