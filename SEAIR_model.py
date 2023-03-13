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
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model(y, t, rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_Iq, gamma_A, gamma_Aq, eta, mu, chi, N_e):
    E, E_q, I, I_q, A, A_q, R_1, R_2 = y
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


y0 = [0, 0, 1, 0, 0, 0, 0, 0]
days = 20
t = np.linspace(0, days, days+1)

rho = 0.525
phi = 1e-4
epsilon = 0.5
beta = 0.5
alpha = 0.55
theta = 0.525
gamma_I = 0.5
gamma_Iq = 0.5
gamma_A = 0.5
gamma_Aq = 0.5
eta = 0.575
mu = 0.5
chi = 0
N_e = 1e7

sol = odeint(model, y0, t, args=(rho, phi, epsilon, beta, alpha, theta, gamma_I, gamma_Iq, gamma_A, gamma_Aq, eta, mu, chi, N_e))

plt.plot(t, sol[:, 0], 'b', label='Exposed')
plt.plot(t, sol[:, 2], 'g', label='Infected')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
