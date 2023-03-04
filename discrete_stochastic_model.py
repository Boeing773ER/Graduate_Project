import numpy as np
from numpy.random import binomial
from math import exp

c_0 = 15    # 初始接触率
c_b = 6.0371    # 当前控制策略下的最小接触率
gamma_1 = 0.051 # 接触率的指数递减速率
beta = 0.21 # 每次接触传播的概率
q_0 = 0     # 最初潜伏者的隔离率
q_m = 0.5228    # 在当前控制策略下潜伏者的最大隔离率
gamma_2 = 0.0799    # 潜伏者隔离率的指数增长速率
m = 0.007   # 易感者向疑似者的转移率
b = 0.0397  # 疑似者的检出率
f = 0.01    # 确诊比例: 疑似的潜伏者向隔离的感染者的转移率
sigma = 1 / 2.38    # 潜伏者到感染者的转移率
lambda_ = 1 / 12    # 隔离的未受感染接触者释放回社区的速率
delta_I0 = 1 / 4.01 # 有症状的感染者向隔离的感染者的初始转化率
delta_If = 0.4076   # 最快诊断速度
gamma_3 = 0.0832    # 诊断率的指数递减速率
gamma_I = 0.0497    # 感染者的恢复率
gamma_H = 0.0199    # 隔离感染者的恢复率
alpha = 0   # 因病死亡率
# h represents the length between the time points at which measurements are taken, here h= 1 day.
h = 1

S = [99895]
E = [0]
I = [0]
B = [0]
S_q = [0]
H = [0]
R = [0]


def contact_rate(t):
    return (c_0 - c_b) * exp(-gamma_1 * t) + c_b


def quarantine_rate(t):
    return (q_0 - q_m) * exp(-gamma_2 * t) + q_m


def re_delta_i(t):
    return ((1 / delta_I0) - (1 / delta_If)) * exp(-gamma_3 * t) + (1 / delta_If)


P11 = list()
P12 = list()
P13 = 1 - exp(-m * h)
P21 = 1 - exp(-sigma * h)
P31 = list()
P32 = 1 - exp(-alpha * h)
P33 = 1 - exp(-gamma_I * h)
P41 = 1 - exp(-b * h)
P51 = 1 - exp(-lambda_ * h)
P61 = 1 - exp(-gamma_H * h)


def calc(T):
    for i in range(0, len(T) - 1):
        # TODO:figure out factor I/N in P11&P12
        P11.append(1 - exp(-beta * contact_rate(i) * I[i] / N * h))
        P12.append(1 - exp(-contact_rate(i) * quarantine_rate(i) * (1 - beta) * I[i] / N * h))
        P31.append(1 - exp(-1 / re_delta_i(i)) * h)

        D11 = binomial(S[i], P11[i])
        D12 = binomial(S[i], P12[i])
        D13 = binomial(S[i], P13)
        D21 = binomial(E[i], P21)
        D31 = binomial(I[i], P31[i])
        D32 = binomial(I[i], P32)
        D33 = binomial(I[i], P33)
        D41 = binomial(B[i], P41)
        D51 = binomial(S_q[i], P51)
        D61 = binomial(H[i], P61)
        D62 = binomial(H[i], P32)

        # Removed the imported case in this model
        S.append(S[i] - D11[i] - D12[i] - D13[i] + D51[i] + (1 - f) * D41[i])
        E.append(E[i] + (1 - quarantine_rate(i)) * D11[i] - D21[i])
        I.append(I[i] + D21 - D31 - D32 - D33)
        B.append(B[i] + quarantine_rate(i) * D11 + D13 - D41)
        S_q.append(S_q[i] + D12 - D51)
        H.append(H[i] + D31 + f * D41 - D61 - D62)
        R.append(R[i] + D33 + D61)
    Y = [S, E, I, R]
    return Y
