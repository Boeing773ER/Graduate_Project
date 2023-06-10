import datetime as dt
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from pylab import mpl


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


def seir_model(y, t, rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, gamma_Iq, chi, N_e):
    E, E_q, I, I_q, A, A_q, R_1, R_2 = y

    dE = (1 - rho) * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E
    dE_q = rho * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E_q
    dI = alpha * eta * E - theta * I - gamma_I * I
    dI_q = alpha * eta * E_q + theta * I - gamma_Iq * I_q
    dA = alpha * (1 - eta) * E - mu * A - gamma_A * A
    dA_q = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
    dR_1 = gamma_Iq * I_q + chi * gamma_Aq * A_q
    dR_2 = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

    return dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2


def sir_model(y, t, rho, phi, beta, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, gamma_Iq, N_e):
    I, I_q, A, A_q, R_1, R_2 = y

    dI = (1 - rho) * phi * (I + beta * A) * eta * (N_e - I - I_q - A - A_q - R_1 - R_2) - theta * I - gamma_I * I
    dI_q = rho * phi * (I + beta * A) * eta * (N_e - I - I_q - A - A_q - R_1 - R_2) + theta * I - gamma_Iq * I_q
    dA = (1 - rho) * phi * (I + beta * A) * (1 - eta) * (N_e - I - I_q - A - A_q - R_1 - R_2) - mu * A - gamma_A * A
    dA_q = rho * phi * (I + beta * A) * (1 - eta) * (N_e - I - I_q - A - A_q - R_1 - R_2) + mu * A - gamma_Aq * A_q
    dR_1 = gamma_Iq * I_q + gamma_Aq * A_q
    dR_2 = gamma_A * A + gamma_I * I

    return dI, dI_q, dA, dA_q, dR_1, dR_2


def seiar_model(y, t, rho, phi, beta, epsilon, alpha, eta, theta, mu, gamma_I, gamma_A, gamma_Aq, z_1, z_2, a, b, chi, N_e):
    E, E_q, I, I_q, A, A_q, R_1, R_2 = y

    gamma_Iq = z_1 + z_2 * math.tanh((t - a) / b)

    dE = (1 - rho) * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E
    dE_q = rho * phi * (I + epsilon * E + beta * A) * (N_e - E - E_q - I - I_q - A - A_q - R_1 - R_2) - alpha * E_q
    dI = alpha * eta * E - theta * I - gamma_I * I
    dI_q = alpha * eta * E_q + theta * I - gamma_Iq * I_q
    dA = alpha * (1 - eta) * E - mu * A - gamma_A * A
    dA_q = alpha * (1 - eta) * E_q + mu * A - gamma_Aq * A_q
    dR_1 = gamma_Iq * I_q + chi * gamma_Aq * A_q
    dR_2 = gamma_A * A + gamma_I * I + (1 - chi) * gamma_Aq * A_q

    return dE, dE_q, dI, dI_q, dA, dA_q, dR_1, dR_2


def classic_seir_model(y, t, phi, epsilon, alpha, gamma_I, N_e):
    E, I, R = y
    dE = phi * (I + epsilon * E) * (N_e - E - I - R) - alpha * E
    dI = alpha * E - gamma_I * I
    dR = gamma_I * I
    return dE, dI, dR


def classic_sir_model(y, t, phi, gamma_I, N_e):
    I, R = y
    dI = phi * (N_e - I - R) - gamma_I * I
    dR = gamma_I * I
    return dI, dR


def calc_days(start, end):
    start_date = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(end, "%Y-%m-%d").date()
    return (end_date - start_date).days


def read_file(file_path, city, start_date, end_date):
    data_file = pd.read_csv(file_path)
    sub_data = data_file.loc[data_file.province == city, :]
    sub_data = sub_data.loc[sub_data.date > start_date, :]
    sub_data = sub_data.loc[end_date > sub_data.date, :]
    ydata = pd.DataFrame()
    ydata["now_confirm"] = sub_data["now_confirm"]
    ydata["heal"] = sub_data["heal"]
    ydata["now_asy"] = sub_data["now_asy"]
    return ydata


def plot_graph(sol, model_name, region, t, y_data, num):

    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.title(model_name + " COVID " + region)
    plt.plot(t, sol[:, num[0]], '--r', label='确诊预测值')
    plt.plot(t, y_data.now_confirm, 'r', label='确诊统计值')
    plt.plot(t, sol[:, num[1]], '--y', label='无症状预测值')
    plt.plot(t, y_data.now_asy, 'y', label='无症状统计值')
    plt.plot(t, sol[:, num[2]], '--g', label='治愈预测值')
    plt.plot(t, y_data.heal, 'g', label='治愈统计值')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def plot_final_graph(sol, model_name, region, t, y_data, num, mid, min_y, max_y):

    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    plt.title(model_name + " COVID " + region)
    # plt.axvline(mid, min_y, max_y)
    # plt.axvline(20, 0, 300000, c="b")
    # plt.axvline(21, 0, 300000, c="b", ls="--")
    # plt.axvline(27, 0, 300000, c="b", ls="--")
    # plt.axvline(34, 0, 300000, c="b", ls="--")
    plt.plot(t, sol[:, num[0]], '--r', label='确诊预测值')
    plt.plot(t, y_data.now_confirm, 'r', label='确诊统计值')
    plt.plot(t, sol[:, num[1]], '--y', label='无症状预测值')
    plt.plot(t, y_data.now_asy, 'y', label='无症状统计值')
    plt.plot(t, sol[:, num[2]], '--g', label='治愈预测值')
    plt.plot(t, y_data.heal, 'g', label='治愈统计值')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def calc_verify_RMSE(file_path, city, start_date, end_date, test_end_date, model, y0, params, inf_index, asy_index, rem_index):
    training_days = calc_days(start_date, end_date)-2
    test_days = calc_days(end_date, test_end_date)
    print(test_days)
    entire_days = calc_days(start_date, test_end_date)-2

    real_data = read_file(file_path, city, start_date, test_end_date)
    t = np.linspace(0, entire_days, entire_days + 1)
    pre_data = odeint(model, y0, t, args=(*params, ))

    print(len(real_data), len(pre_data), t)

    plot_final_graph(pre_data, "SEIAR_Variable", city, t, real_data, [inf_index, asy_index, rem_index], 37, 0, 300000)

    inf_x = pre_data[-test_days:, inf_index]
    inf_y = real_data.now_confirm[-test_days:]
    print(len(inf_x), len(inf_y))
    assert len(inf_x) == len(inf_y)
    inf_loss = np.sqrt(np.sum(np.square(inf_x - inf_y)) / len(inf_x))

    asy_x = pre_data[-test_days:, asy_index]
    asy_y = real_data.now_asy[-test_days:]
    assert len(asy_x) == len(asy_y)
    asy_loss = np.sqrt(np.sum(np.square(asy_x - asy_y)) / len(asy_x))

    rem_x = pre_data[-test_days:, rem_index]
    rem_y = real_data.heal[-test_days:]
    assert len(rem_x) == len(rem_y)
    rem_loss = np.sqrt(np.sum(np.square(rem_x - rem_y)) / len(rem_x))
    loss_mean = np.mean([inf_loss, asy_loss, rem_loss])

    return [inf_loss, asy_loss, rem_loss, loss_mean]


def calc_training_RMSE(file_path, city, start_date, end_date, model, y0, params, inf_index, asy_index, rem_index):
    days = calc_days(start_date, end_date)-2
    real_data = read_file(file_path, city, start_date, end_date)
    t = np.linspace(0, days, days + 1)
    pre_data = odeint(model, y0, t, args=(*params, ))

    plot_graph(pre_data, "SIAR", city, t, real_data, [inf_index, asy_index, rem_index])

    print(pre_data)
    print(real_data)

    inf_x = pre_data[:, inf_index]
    inf_y = real_data.now_confirm
    print(len(inf_x), len(inf_y))
    assert len(inf_x) == len(inf_y)
    inf_loss = np.sqrt(np.sum(np.square(inf_x - inf_y)) / len(inf_x))

    asy_x = pre_data[:, asy_index]
    asy_y = real_data.now_asy
    assert len(asy_x) == len(asy_y)
    asy_loss = np.sqrt(np.sum(np.square(asy_x - asy_y)) / len(asy_x))

    rem_x = pre_data[:, rem_index]
    rem_y = real_data.heal
    assert len(rem_x) == len(rem_y)
    rem_loss = np.sqrt(np.sum(np.square(rem_x - rem_y)) / len(rem_x))
    loss_mean = np.mean([inf_loss, asy_loss, rem_loss])

    return [inf_loss, asy_loss, rem_loss, loss_mean]


def old_func(model, y0, phen, file_path, city, start_date, end_date):
    data_file = pd.read_csv(file_path)
    sub_data = data_file.loc[data_file.province == city, :]
    sub_data = sub_data.loc[sub_data.date > start_date, :]
    sub_data = sub_data.loc[end_date > sub_data.date, :]
    y_data = pd.DataFrame()
    y_data["now_confirm"] = sub_data["now_confirm"]
    y_data["heal"] = sub_data["heal"]
    y_data["now_asy"] = sub_data["now_asy"]

    days = calc_days(start_date, end_date) - 2
    t = np.linspace(0, days, days + 1)

    sol = odeint(model, y0, t, args=(*phen, ))
    I_q = sol[:, 3]
    A_q = sol[:, 5]
    R_q = sol[:, 6]

    loss1 = loss_eva(rmse_loss, I_q, y_data.now_confirm.to_numpy())
    loss2 = loss_eva(rmse_loss, A_q, y_data.now_asy.to_numpy())
    loss3 = loss_eva(rmse_loss, R_q, y_data.heal.to_numpy())
    loss = np.mean([loss1, loss2, loss3])
    # loss = np.mean([loss1, loss3])
    print(loss1, loss2, loss3, loss)


def plot_compare(file_path, city, start_date, end_date, test_end_date, base_name, noval_name, base_data, noval_data, index):
    training_days = calc_days(start_date, end_date) - 2
    test_days = calc_days(end_date, test_end_date)
    entire_days = calc_days(start_date, test_end_date) - 2
    y_data = read_file(file_path, city, start_date, test_end_date)
    t = np.linspace(0, entire_days, entire_days + 1)

    plt.title(noval_name + " compare to " + base_name)
    # plt.axvline(mid, min_y, max_y)
    """plt.axvline(19, 0, 300000, c="b")
    plt.axvline(20, 0, 300000, c="b", ls="--")
    plt.axvline(26, 0, 300000, c="b", ls="--")
    plt.axvline(33, 0, 300000, c="b", ls="--")"""
    plt.axvline(36, 0, 300000, c="b")
    plt.axvline(37, 0, 300000, c="b", ls="--")
    plt.axvline(43, 0, 300000, c="b", ls="--")
    plt.axvline(50, 0, 300000, c="b", ls="--")

    plt.plot(t, noval_data[:, index[0]], '--r', label=noval_name+'确诊预测值')
    plt.plot(t, base_data[:, 1], ':r', label=base_name+'确诊预测值')
    plt.plot(t, y_data.now_confirm, 'r', label='确诊统计值')

    # plt.plot(t, noval_data[:, index[1]], '--y', label=noval_name+'无症状预测值')
    # plt.plot(t, y_data.now_asy, 'y', label='无症状统计值')

    plt.plot(t, noval_data[:, index[2]], '--g', label=noval_name+'治愈预测值')
    plt.plot(t, base_data[:, 2], ':g', label=base_name+'治愈预测值')
    plt.plot(t, y_data.heal, 'g', label='治愈统计值')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


def triple_compare(file_path, city, start_date, end_date, test_end_date, base_name, model1_name, model2_name,
                   base_data, model1_data, model2_data, index):
    training_days = calc_days(start_date, end_date) - 2
    test_days = calc_days(end_date, test_end_date)
    entire_days = calc_days(start_date, test_end_date) - 2
    y_data = read_file(file_path, city, start_date, test_end_date)
    t = np.linspace(0, entire_days, entire_days + 1)

    plt.title(model1_name + " " + model2_name + " " + base_name)
    """plt.axvline(19, 0, 300000, c="b")
    plt.axvline(20, 0, 300000, c="b", ls="--")
    plt.axvline(26, 0, 300000, c="b", ls="--")
    plt.axvline(33, 0, 300000, c="b", ls="--")"""
    plt.axvline(36, 0, 300000, c="b")
    plt.axvline(37, 0, 300000, c="b", ls="--")
    plt.axvline(43, 0, 300000, c="b", ls="--")
    plt.axvline(50, 0, 300000, c="b", ls="--")

    # plt.plot(t, model1_data[:, index[0]], '--r', label=model1_name + '确诊预测值')
    # plt.plot(t, model2_data[:, index[0]], '-.r', label=model2_name + '确诊预测值')
    # plt.plot(t, base_data[:, 1], ':r', label=base_name + '确诊预测值')
    # plt.plot(t, y_data.now_confirm, 'r', label='确诊统计值')

    plt.plot(t, model1_data[:, index[1]], '--y', label=model1_name + '无症状预测值')
    plt.plot(t, model2_data[:, index[1]], '-.y', label=model2_name + '无症状预测值')
    plt.plot(t, base_data[:, 3], ':y', label=base_name + '无症状预测值')
    plt.plot(t, y_data.now_asy, 'y', label='无症统计测值')

    # plt.plot(t, model1_data[:, index[2]], '--g', label=model1_name + '治愈预测值')
    # plt.plot(t, model2_data[:, index[2]], '-.g', label=model2_name + '治愈预测值')
    # plt.plot(t, base_data[:, 2], ':g', label=base_name + '治愈预测值')
    # plt.plot(t, y_data.heal, 'g', label='治愈统计值')

    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()


mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

file_path = "../CN_COVID_data/domestic_data.csv"
region = "上海"
start_date = "2022-03-10"
end_date = "2022-04-17"
plot_end_date = "2022-06-17"

# dI, dI_q, dA, dA_q, dR_1, dR_2

# y0 = [64, 646, 37, 370, 4067, 406]
y0 = [1, 0, 0, 0, 0, 0]

# y0 = [0, 0, 1, 0, 0, 0, 0, 0]


sir_index = [1, 3, 4]

# SIR_params_old = [0.0019532442165659525, 6.103888176768602e-05, 6.103888176768602e-05, 0.0004883110541414881,
#           0.03338826832692425, 0.5002136360861869, 0.7556613562839529, 0.004028566196667277, 0.003174021851919673,
#           0.9498870780687297, 24890000.0]

# SIR_params_old = [0.12897515717512056, 0.00012207776353537203, 0.0, 0.00018311664530305805, 0.35292681438076057,
#               0.01715192577671977, 0.0012207776353537203, 0.4949032533723982, 0.0022584386254043826, 0.0, 24890000.0]

SIR_params = [6.1031431187061336e-05, 6.1031431187061336e-05, 0.007873054623130912, 0.750320415013732,
              6.1031431187061336e-05, 0.0008544400366188587, 0.9998779371376258, 6.1031431187061336e-05,
              0.000366188587122368, 0.01367104058590174, 24890000]

SIR_params2 = [0.12897515717512056, 0.00012207776353537203, 0.0, 0.00018311664530305805, 0.35292681438076057,
               0.01715192577671977, 0.0012207776353537203, 0.4949032533723982, 0.0022584386254043826, 0.0, 24890000]

index = [3, 5, 6]

# SEIR_params = [0.4640175791979491, 0.00012207776353537203, 0.006958432521516206, 0.00024415552707074406,
#                0.0014038942806567783, 0.035158395898187145, 0.5002746749679546, 0.12842580723921138, 0.5148629677104315,
#                0.5271928218275042, 0.004699993896111823, 0.0, 0, 24890000.0]

SEIR_params = [0.7500152578577968, 6.1031431187061336e-05, 0.06499847421422032, 0.00024412572474824534,
               0.0039060115959719255, 0.009765028989929814, 0.9999389685688129, 0.2642660970399756, 0.9999389685688129,
               6.1031431187061336e-05, 0.5000305157155935, 0.031248092767775404, 0, 24890000]

SEIR_params2 = [0.589635597875847, 0.0005493499359091741, 0.021546725263993163, 0.0, 0.0009155832265152902,
                0.058597326496978575, 0.6294329487883782, 0.7761093816761276, 0.0012818165171214063, 0.30537752548373315,
                0.0017090886894952084, 0.06256485381187817, 0, 24890000]

SEIR_params3 = [0.00012206286237412267, 6.1031431187061336e-05, 0.0007934086054317974,  6.1031431187061336e-05,
                0.03021055843759536, 0.0017699115044247787, 0.9995727799816906, 0.0028684772657918828,
                0.4969179127250534, 0.023863289594140982, 0.0010375343301800427, 0.023497101007018614, 0, 24890000]

# SEIAR_params_old = [0.8942806567783678, 0.0004883110541414881, 0.0010376609900506623, 0.00012207776353537203,
#                 0.007751937984496124, 0.012512970762375633, 0.7001159738753586, 0.7068912897515718, 0.01593114814136605,
#                 0.0006103888176768602, 0.5088201184154306, 0.7068912897515718, 0.6245498382469633, 60.81482011301051,
#                 7.337855055656858, 0, 24890000.0]

SEIAR_params = [0.24998474214220323, 0.000183094293561184, 0.008544400366188587, 6.1031431187061336e-05,
               0.0009765028989929814, 0.031248092767775404, 0.9999389685688129, 0.1603906011595972, 0.8153799206591394,
               0.004333231614281355, 0.0003051571559353067, 0.6718339945071712, 0.6420506560878853, 77.46670010881414,
               0.3411076270475231, 0, 24890000]
SEIAR_params2 = [0.7663716814159292, 6.1031431187061336e-05, 0.0008544400366188587, 0.0003051571559353067,
                 0.014281354897772353, 0.00732377174244736, 0.9981080256332011, 0.02642660970399756, 0.9911504424778761,
                 0.00012206286237412267, 0.642722001830943, 0.709490387549588, 0.6813548977723528, 76.07576744483238,
                 3.1417606982368604, 0, 24890000]

Classic_SEIR_params = [1e-06, 0.04434674945215322, 1.5258788948813162e-05, 0.02914075531705651, 24890000]

Classic_SEIR_params2 = [3.5166740155467835e-06, 0.007040679402346317, 2.3417174641660043e-05, 0.02998461551975745,
                        24890000]
Classic_SEIR_params3 = [0.46173553569811926, 0.04637224937698059, 1.0274350566608081e-05, 0.019530854973712153, 24890000]


Classic_SIR_params = [6.1031431187061336e-05, 0.0091547146780592, 24890000]


# result = calc_training_RMSE(file_path, region, start_date, end_date, sir_model, y0, SIR_params, *sir_index)
# result = calc_verify_RMSE(file_path, region, start_date, end_date, plot_end_date, sir_model, y0, SIR_params, *sir_index)

# SIAR
y0 = [500, 548, 2800, 2793, 4472, 0]
# y0 = [1, 7, 1, 2, 3383, 0]
# result = calc_training_RMSE(file_path, region, "2022-03-20", "2022-04-10", sir_model, [500, 548, 2800, 2793, 4472, 0],
#                             SIR_params, *sir_index)
# result = calc_verify_RMSE(file_path, region, "2022-03-20", "2022-04-10", "2022-04-24", sir_model,
#                           [500, 548, 2800, 2793, 4472, 0], SIR_params, *sir_index)

# result = calc_training_RMSE(file_path, region, "2022-03-10", "2022-04-17", sir_model, [1, 646, 1, 370, 4067, 0],
#                             SIR_params2, *sir_index)
# result = calc_verify_RMSE(file_path, region, "2022-03-10", "2022-04-17", "2022-05-01", sir_model,
#                           [1, 646, 1, 370, 4067, 0], SIR_params2, *sir_index)


# SEIAR
# y0 = [0, 0, 500, 548, 2800, 2793, 4472, 0]
# result = calc_training_RMSE(file_path, region, "2022-03-20", "2022-04-10", seir_model,
#                             [0, 0, 500, 548, 2800, 2793, 4472, 0], SEIR_params, *index)

# result = calc_verify_RMSE(file_path, region, "2022-03-20", "2022-04-10", "2022-04-24", seir_model,
#                           [0, 0, 500, 548, 2800, 2793, 4472, 0], SEIR_params, *index)
# result = calc_training_RMSE(file_path, region, "2022-03-10", "2022-04-17", seir_model,
#                             [0, 0, 500, 646, 2800, 370, 4067, 0], SEIR_params3, *index)

# result = calc_verify_RMSE(file_path, region, "2022-03-10", "2022-04-17", "2022-05-01", seir_model,
#                           [0, 0, 500, 646, 2800, 370, 4067, 0], SEIR_params3, *index)


# SEIAR_Variable
y0 = [0, 0, 500, 548, 2800, 2793, 4472, 0]
# result = calc_training_RMSE(file_path, region, "2022-03-20", "2022-04-10", seiar_model,
#                             [0, 0, 500, 548, 2800, 2793, 4472, 0], SEIAR_params, *index)

# result = calc_verify_RMSE(file_path, region, "2022-03-20", "2022-04-10", "2022-04-24", seiar_model,
#                             [0, 0, 500, 548, 2800, 2793, 4472, 0], SEIAR_params, *index)

# [0, 0, 500, 646, 2800, 370, 4067, 0]
# result = calc_training_RMSE(file_path, region, "2022-03-10", "2022-04-17", seiar_model,
#                             [0, 0, 500, 646, 2800, 370, 4067, 0], SEIAR_params2, *index)

result = calc_verify_RMSE(file_path, region, "2022-03-10", "2022-04-17", "2022-05-01", seiar_model,
                            [0, 0, 500, 646, 2800, 370, 4067, 0], SEIAR_params2, *index)

# classic SEIR
# [0, 548, 4472]
# result = calc_training_RMSE(file_path, region, "2022-03-20", "2022-04-10", classic_seir_model, [0, 548, 4472],
#                             Classic_SEIR_params2, *[1, 1, 2])

# result = calc_verify_RMSE(file_path, region, "2022-03-20", "2022-04-10", "2022-04-24", classic_seir_model,
#                           [0, 548, 4472], Classic_SEIR_params2, *[1, 1, 2])

# result = calc_training_RMSE(file_path, region, "2022-03-10", "2022-04-17", classic_seir_model, [0, 646, 4067],
#                             Classic_SEIR_params3, *[1, 1, 2])

# result = calc_verify_RMSE(file_path, region, "2022-03-10", "2022-04-17", "2022-05-01", classic_seir_model,
#                           [0, 646, 4067], Classic_SEIR_params3, *[1, 1, 2])

# classic SIR
# y0 = [548, 4472]
# [6.1031431187061336e-05, 0.00592004882514495, 24890000]
# result = calc_training_RMSE(file_path, region, "2022-03-20", "2022-04-10", classic_sir_model, [548, 4472],
#                             Classic_SIR_params, *[0, 0, 1])
# result = calc_verify_RMSE(file_path, region, "2022-03-20", "2022-04-10", "2022-04-24", classic_sir_model, [548, 4472],
#                             Classic_SIR_params, *[0, 0, 1])
# [3348.809797249851, 108774.79910805445, 652.9488111546764, 37592.18590548632]
# [25494.99312526286, 87680.02355931165, 1447.17962563284, 38207.398770069114]

# [9.14563728e-06, 1.17245048e-02, 24890000]
# result = calc_training_RMSE(file_path, region, "2022-03-10", "2022-04-17", classic_sir_model, [646, 4067],
#                             [9.14563728e-06, 1.17245048e-02, 24890000], *[0, 0, 1])

# result = calc_verify_RMSE(file_path, region, "2022-03-10", "2022-04-17", "2022-05-01", classic_sir_model, [646, 4067],
#                             [9.14563728e-06, 1.17245048e-02, 24890000], *[0, 0, 1])



"""entire_days = calc_days("2022-03-20", "2022-04-24") - 2
t = np.linspace(0, entire_days, entire_days + 1)

siar_data = odeint(sir_model, [500, 548, 2800, 2793, 4472, 0], t, args=(*SIR_params, ))
seiar_data = odeint(seir_model, [0, 0, 500, 548, 2800, 2793, 4472, 0], t, args=(*SEIR_params, ))
seiar_v_data = odeint(seiar_model, [0, 0, 500, 548, 2800, 2793, 4472, 0], t, args=(*SEIAR_params, ))
classic_sir_data = odeint(classic_sir_model, [646, 4067], t, args=(*[9.14563728e-06, 1.17245048e-02, 24890000], ))
classic_seir_data = odeint(classic_seir_model, [0, 646, 4067], t, args=(*Classic_SEIR_params3, ))

# plot_compare(file_path, region, "2022-03-20", "2022-04-10", "2022-04-24", "SEIR", "SEIAR_Variable",
#              classic_seir_data, seiar_v_data, index)
triple_compare(file_path, region, "2022-03-20", "2022-04-10", "2022-04-24", "SIAR", "SEIAR", "SEIAR_Varaible",
             siar_data, seiar_data, seiar_v_data, index)"""


"""entire_days = calc_days("2022-03-10", "2022-05-01") - 2
t = np.linspace(0, entire_days, entire_days + 1)

siar_data = odeint(sir_model, [1, 646, 1, 370, 4067, 0], t, args=(*SIR_params2, ))
seiar_data = odeint(seir_model, [0, 0, 500, 646, 2800, 370, 4067, 0], t, args=(*SEIR_params3, ))
seiar_v_data = odeint(seiar_model, [0, 0, 500, 646, 2800, 370, 4067, 0], t, args=(*SEIAR_params2, ))
classic_sir_data = odeint(classic_sir_model, [646, 4067], t, args=(*Classic_SIR_params, ))
classic_seir_data = odeint(classic_seir_model, [0, 646, 4067], t, args=(*Classic_SEIR_params3, ))

# plot_compare(file_path, region, "2022-03-10", "2022-04-17", "2022-05-01", "SEIR", "SEIAR_Variable",
#              classic_seir_data, seiar_v_data, index)
triple_compare(file_path, region, "2022-03-10", "2022-04-17", "2022-05-01", "SIAR", "SEIAR", "SEIAR_Varaible",
             siar_data, seiar_data, seiar_v_data, index)"""

# result = calc_training_RMSE(file_path, region, start_date, end_date, seiar_model, y0, SEIAR_params, *index)
# result = calc_verify_RMSE(file_path, region, start_date, end_date, plot_end_date, seiar_model, y0, SEIAR_params, *index)
print(result)

# old_func(seiar_model, y0, SEIAR_params, file_path, region, start_date, end_date)
