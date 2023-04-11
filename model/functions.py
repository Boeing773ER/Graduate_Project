import datetime as dt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def calc_days(start, end):
    start_date = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(end, "%Y-%m-%d").date()
    return (end_date - start_date).days


def read_file(file_path, city, start_date, end_date):
    data_file = pd.read_csv(file_path)
    sub_data = data_file.loc[data_file.province == city, :]
    sub_data = sub_data.loc[sub_data.date > start_date, :]
    sub_data = sub_data.loc[end_date > sub_data.date, :]
    # sub_data = data_file.loc[data_file.province == "city", :]
    # sub_data = sub_data.loc["2020-02-11" > sub_data.date, :]
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
    assert len(x) == len(y)
    loss = np.sqrt(np.sum(np.square(x - y)) / len(x))
    return loss


def loss_eva(function, x: np.ndarray, y: np.ndarray):
    return function(x, y)


def plot_graph(file_name, sol, model_name, region, t, y_data):
    plt.title(model_name + "COVID " + region)
    plt.plot(t, sol[:, 3], '--g', label='Pre_Inf_q')
    plt.plot(t, y_data.now_confirm, 'g', label='Real_Inf_q')
    plt.plot(t, sol[:, 5], '--r', label='Pre_Asy_q')
    plt.plot(t, y_data.now_asy, 'r', label='Real_Asy_q')
    plt.plot(t, sol[:, 6], '--y', label='Pre_Removed_q')
    plt.plot(t, y_data.heal, 'y', label='Real_Removed_q')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.savefig("../img/pic-"+file_name+".png")
    plt.show()
