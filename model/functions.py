import datetime as dt
import numpy as np
import pandas as pd


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
