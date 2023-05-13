import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def model_SI(y, t, r, beta, N):
    S, I = y
    dS = -r * beta * I * S / N
    dI = r * beta * I * S/N
    return dS, dI


def model_SIS(y, t, r, beta, gamma, N):
    S, I = y
    dS = -r * beta * I * S/N + gamma * I
    dI = r * beta * I * S/N - gamma * I
    return dS, dI


def model_SIR(y, t, r, beta, gamma, N):
    S, I, R = y
    dS = -r * beta * I * S/N
    dI = r * beta * I * S/N - gamma * I
    dR = gamma * I
    return dS, dI, dR


def model_SEIR(y, t, r, beta, gamma, alpha, N):
    S, E, I, R = y
    dS = -r * beta * I * S/N
    dE = r * beta * I * S/N - alpha * E
    dI = alpha * E - gamma * I
    dR = gamma * I
    return dS, dE, dI, dR


def plot_SI():
    r = 10
    beta = 0.05
    y0 = [999, 1]
    days = 50
    N = 1000
    t = np.linspace(0, days, days + 1)
    sol = odeint(model_SI, y0, t, args=(r, beta, N))

    plt.title("Model_SI")
    plt.plot(t, sol[:, [0]], 'b', label='Susceptible')
    plt.plot(t, sol[:, [1]], 'r', label='Infected')
    plt.legend(loc='best')
    plt.xlabel('day')
    plt.ylabel("Number")
    plt.grid()
    plt.savefig("../img/pic-SI_model_example.png")
    plt.show()


def plot_SIS():
    r = 10
    beta = 0.05
    gamma = 0.1
    y0 = [999, 1]
    days = 50
    N = 1000
    t = np.linspace(0, days, days + 1)
    sol = odeint(model_SIS, y0, t, args=(r, beta, gamma, N))

    plt.title("Model_SIS")
    plt.plot(t, sol[:, [0]], 'b', label='Susceptible')
    plt.plot(t, sol[:, [1]], 'r', label='Infected')
    plt.legend(loc='best')
    plt.xlabel('day')
    plt.ylabel("Number")
    plt.grid()
    plt.savefig("../img/pic-SIS_model_example.png")
    plt.show()


def plot_SIR():
    r = 10
    beta = 0.05
    gamma = 0.1
    y0 = [999, 1, 0]
    days = 70
    N = 1000
    t = np.linspace(0, days, days + 1)
    sol = odeint(model_SIR, y0, t, args=(r, beta, gamma, N))

    plt.title("Model_SIR")
    plt.plot(t, sol[:, [0]], 'b', label='Susceptible')
    plt.plot(t, sol[:, [1]], 'r', label='Infected')
    plt.plot(t, sol[:, [2]], 'g', label='Removed')
    plt.legend(loc='best')
    plt.xlabel('day')
    plt.ylabel("Number")
    plt.grid()
    plt.savefig("../img/pic-SIR_model_example.png")
    plt.show()


def plot_SEIR():
    r = 10
    beta = 0.25
    gamma = 0.1
    alpha = 0.1
    y0 = [999, 0, 1, 0]
    days = 70
    N = 1000
    t = np.linspace(0, days, days + 1)
    sol = odeint(model_SEIR, y0, t, args=(r, beta, gamma, alpha, N))

    plt.title("Model_SIR")
    plt.plot(t, sol[:, [0]], 'b', label='Susceptible')
    plt.plot(t, sol[:, [1]], 'g', label='Exposed')
    plt.plot(t, sol[:, [2]], 'r', label='Infected')
    plt.plot(t, sol[:, [3]], 'y', label='Removed')
    plt.legend(loc='best')
    plt.xlabel('day')
    plt.ylabel("Number")
    plt.grid()
    plt.savefig("../img/pic-SEIR_model_example.png")
    plt.show()


if __name__ == '__main__':
    # plot_SI()
    # plot_SIS()
    plot_SIR()
    plot_SEIR()
