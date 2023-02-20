import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as spi
import pylab as pl

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


ND = 70  # 时间长度
TS = 1.0  # 按天生成
beta = 1.4247  # 每个感染者每日有效接触人数
gamma = 0  # 治愈率
I0 = 1e-6  # initial infected
INPUT = (1.0 - I0, I0)  # (S0, I0)

# SI模型函数
def diff_eqs1(INP, t):
    Y = np.zeros(2)
    V = INP
    Y[0] = - beta * V[0] * V[1] + gamma * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    return Y


t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)  # 生成日期范围
RES = spi.odeint(diff_eqs1, INPUT, t_range)  # 数值求解微分方程

pl.plot(RES[:, 0], 'b', label='易感者')
pl.plot(RES[:, 1], 'r', label='传染者')
pl.legend(loc=0)
pl.title('SI-nCoV 传播时间曲线')
pl.xlabel('时间(天)')
pl.ylabel('人数比例')
pl.savefig('SI-nCoV 传播时间曲线.png', dpi=900)
pl.show()


# -----------------SIS---------------
beta = 1.4247
gamma = 0.14286
I0 = 1e-6
ND = 70
TS = 1.0
INPUT = (1.0 - I0, I0)

# SIS
def diff_eqs2(INP, t):
    Y = np.zeros((2))
    V = INP
    Y[0] = - beta * V[0] * V[1] + gamma * V[1]
    Y[1] = beta * V[0] * V[1] - gamma * V[1]
    return Y


t_range = np.arange(t_start, t_end + t_inc, t_inc)
RES = spi.odeint(diff_eqs2, INPUT, t_range)
pl.plot(RES[:, 0], '-b', label='易感者')
pl.plot(RES[:, 1], '-r', label='传染者')
pl.legend(loc=0)
pl.title('SIS-nCoV 传播时间曲线')
pl.xlabel('时间(天)')
pl.ylabel('人数比例')
pl.savefig('SIS-nCoV 传播时间曲线.png', dpi=900)  # This does increase the resolution.
pl.show()


# --------------SIR--------------

alpha = 0.000004  # 治愈率
beta = 0.1
TS = 1.0  # 观察间隔
ND = 120.0  # 观察结束日期
S0 = 100000  # 初始易感人数
I0 = 10  # 初始感染人数
INPUT = (S0, I0, 0.0)


def diff_eqs(INP, t):
    Y = np.zeros(3)
    V = INP
    Y[0] = - alpha * V[0] * V[1]
    Y[1] = alpha * V[0] * V[1] - beta * V[1]
    Y[2] = beta * V[1]
    return Y


t_start = 0.0
t_end = ND
t_inc = TS
t_range = np.arange(t_start, t_end + t_inc, t_inc)  # 生成日期范围
RES = spi.odeint(diff_eqs, INPUT, t_range)
pl.plot(RES[:, 0], '-g', label='易感者')
pl.plot(RES[:, 1], '-r', label='传染者')
pl.plot(RES[:, 2], '-k', label='移除者')
pl.legend(loc=0)
pl.title('SIR-nCoV 传播时间曲线')
pl.xlabel('时间(天)')
pl.ylabel('人数')
pl.savefig('SIR-nCoV 传播时间曲线.png', dpi=900)
pl.show()

