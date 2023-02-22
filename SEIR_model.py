import matplotlib.pyplot as plt
import pylab as pl

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def calc_SEIR(N, T, S, E, I, R, r, b, a, r2, b2, y):
    for i in range(0, len(T) - 1):
        S.append(S[i] - r * b * S[i] * I[i] / N - r2 * b2 * S[i] * E[i] / N)
        E.append(E[i] + r * b * S[i] * I[i] / N - a * E[i] + r2 * b2 * S[i] * E[i] / N)
        I.append(I[i] + a * E[i] - y * I[i])
        R.append(R[i] + y * I[i])
    Y = [S, E, I, R]
    return Y


# def plot(T, S, E, I, R):
#     plt.figure()
#     plt.title("SEIR-nCoV 传播时间曲线")
#     plt.plot(T, S, color='r', label='易感者')
#     plt.plot(T, E, color='k', label='潜伏者')
#     plt.plot(T, I, color='b', label='传染者')
#     plt.plot(T, R, color='g', label='移除者')
#     plt.grid(False)
#     plt.legend()
#     plt.xlabel("时间(天)")
#     plt.ylabel("人数")
#     pl.savefig('SEIR-nCoV 传播时间曲线.png', dpi=900)
#     plt.show()


# def get_SEIR(N, E, I, S, R, r, b, a, r2, b2, y, days):
#     T = [i for i in range(0, days)]  # 时间
#     calc(N, T, S, E, I, R, r, b, a, r2, b2, y)
#     # plot(T, S, E, I, R)
#     return S, E, I, R
