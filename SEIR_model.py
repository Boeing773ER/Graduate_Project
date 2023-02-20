import matplotlib.pyplot as plt
import pylab as pl

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def calc(T):
    for i in range(0, len(T) - 1):
        S.append(S[i] - r * b * S[i] * I[i] / N - r2 * b2 * S[i] * E[i] / N)
        E.append(E[i] + r * b * S[i] * I[i] / N - a * E[i] + r2 * b2 * S[i] * E[i] / N)
        I.append(I[i] + a * E[i] - y * I[i])
        R.append(R[i] + y * I[i])


def plot(T, S, E, I, R):
    plt.figure()
    plt.title("SEIR-nCoV 传播时间曲线")
    plt.plot(T, S, color='r', label='易感者')
    plt.plot(T, E, color='k', label='潜伏者')
    plt.plot(T, I, color='b', label='传染者')
    plt.plot(T, R, color='g', label='移除者')
    plt.grid(False)
    plt.legend()
    plt.xlabel("时间(天)")
    plt.ylabel("人数")
    pl.savefig('SEIR-nCoV 传播时间曲线.png', dpi=900)
    plt.show()


if __name__ == '__main__':
    N = 100000  # 人口总数
    E = [0]  # 潜伏携带者
    I = [1]  # 传染者
    S = [N - I[0]]  # 易感者
    R = [0]  # 康复者
    r = 20  # 传染者接触人数
    b = 0.03  # 传染者传染概率
    a = 0.1  # 潜伏者患病概率
    r2 = 30  # 潜伏者接触人数
    b2 = 0.03  # 潜伏者传染概率
    y = 0.1  # 康复概率
    T = [i for i in range(0, 120)]  # 时间
    calc(T)
    plot(T, S, E, I, R)
