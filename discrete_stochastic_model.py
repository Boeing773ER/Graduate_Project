import numpy as np
from numpy.random import binomial
from math import exp

c_0 = 15
c_b = 6.0371
gamma_1 = 0.051
beta = 0.21
q_0 = 0
q_m = 0.5228
gamma_2 = 0.0799
m = 0.007
b = 0.0397
f = 0.01
sigma = 1 / 2.38
lambda_ = 1 / 12
delta_I0 = 1 / 4.01
delta_If = 0.4076
gamma_3 = 0.0832
gamma_I = 0.0497
gamma_H = 0.0199
alpha = 0
# h represents the length between the time points at which measurements are taken, here h= 1 day.
h = 1


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


def calc(N, T, S, E, I, B, S_q, H, R, r, b, a, r2, b2, y):
    for i in range(0, len(T) - 1):
        P11.append(1 - exp(-1 * beta * contact_rate(i) * I[i] / N * h))
        P12.append(1 - exp(-1 * contact_rate(i) * quarantine_rate(i) * (1 - beta) * I[i] / N * h))
        P31.append(1 - exp(-1 * (1 / re_delta_i(i))) * h)

        D11 = binomial(S[i], P11[i])
        D12 = binomial(S[i], P12[i])
        D13 = binomial(S[i], P31)
        D21 = binomial(E[i], P21)
        D31 = binomial(I[i], P31[i])
        D32 = binomial(I[i], P32)
        D33 = binomial(I[i], P33)
        D41 = binomial(B[i], P41)
        D51 = binomial(S_q[i], P51)
        D61 = binomial(H[i], P61)
        D62 = binomial(H[i], P32)

        S.append(S[i] - D11[i] - D12[i] - D13[i] + D51[i] + (1-f)*D41[i])
        E.append(E[i] + (1-quarantine_rate(i))*D11[i] - D21[i] + )
        I.append(I[i] + a * E[i] - y * I[i])
        R.append(R[i] + y * I[i])
    Y = [S, E, I, R]
    return Y
