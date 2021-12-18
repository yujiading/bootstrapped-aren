import pathlib
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from bootstrapped_aren.driver_index_tracking import BestPara
from bootstrapped_aren.driver_index_tracking import DriverIndexTrackSp500Aren


def test_driver_index_track_sp500_aren_saving_data():
    driver = DriverIndexTrackSp500Aren(bootstrap_replicates_lst=[None, 8, 16, 32, 64, 128, 256],  # increase
                                       soft_J_percentage_lst=[1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6],
                                       # None if no bootstrap # decrease
                                       is_fit_intercept=False,
                                       is_center_data=True,
                                       # if is_fit_intercept is True and is_center_data is False:
                                       #     fit with intercept, but penalize intercept, not accurate, see The Elements
                                       #     of Statistical Learning page 64
                                       # if is_fit_intercept is False and is_center_data is False:
                                       #     fit without intercept
                                       # if is_fit_intercept is False and is_center_data is True:
                                       #     fit with intercept, best option
                                       # please change only above
                                       # do not change below
                                       percent_money_on_each_stock=1,  # if 100%, no limit on constraints
                                       max_feature_selected=None,
                                       # do not change below
                                       n_alphas=10,  # from 0 to 1
                                       n_lambdas=100,  # from 0.001*max_lam to max_lam
                                       start_date='2020-09-01',
                                       end_date='2021-09-01',
                                       train_size=0.7,
                                       val_size=0.2,
                                       test_size=0.1)

    _ = driver.run_once(bootstrap_replicates=None)


def test_driver_index_track_sp500_aren_exportung_results():
    # percent_money_on_each_stock = 0.1  # M = 1, 0.3, 0.2, 0.1 # if 100%, no limit on constraints
    # max_feature_selected = None  # None, 200, 150, 100, 50
    data_dir_path = pathlib.Path(__file__).parent / '../data'
    for max_feature_selected in [None, 200, 150, 100, 50]:
        filename = data_dir_path / f'out_max{max_feature_selected}.csv'
        file_path = pathlib.Path(filename)
        table_all = pd.DataFrame()
        for percent_money_on_each_stock in [1, 0.3, 0.2, 0.1]:
            driver = DriverIndexTrackSp500Aren(bootstrap_replicates_lst=[None, 8, 16, 32, 64, 128, 256],
                                               soft_J_percentage_lst=[1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6],
                                               is_fit_intercept=False,
                                               is_center_data=True,
                                               # do not change above
                                               # change below to export results
                                               percent_money_on_each_stock=percent_money_on_each_stock,
                                               max_feature_selected=max_feature_selected,
                                               # please change only above
                                               # do not change below
                                               n_alphas=10,  # from 0 to 1
                                               n_lambdas=100,  # from 0.001*max_lam to max_lam
                                               start_date='2020-09-01',
                                               end_date='2021-09-01',
                                               train_size=0.7,
                                               val_size=0.2,
                                               test_size=0.1)

            table, latex = driver.run_all()
            table_all = table_all.append(table)
        table_all.to_csv(file_path)


def test_driver_index_track_sp500_aren_plot_many():
    data_dir_path = pathlib.Path(__file__).parent / '../data'
    filename = data_dir_path / f'plot_data.pickle'
    file_path = pathlib.Path(filename)
    if not file_path.exists():
        parameters = [
            BestPara(Stocks=None, M=1, m=64, S=0.95),
            BestPara(Stocks=None, M=0.3, m=32, S=0.75),
            BestPara(Stocks=None, M=0.2, m=64, S=0.75),
            BestPara(Stocks=None, M=0.1, m=32, S=0.9),
            BestPara(Stocks=200, M=1, m=128, S=0.9),
            BestPara(Stocks=200, M=0.3, m=32, S=0.95),
            BestPara(Stocks=200, M=0.2, m=32, S=0.95),
            BestPara(Stocks=200, M=0.1, m=32, S=0.9),
            BestPara(Stocks=150, M=1, m=64, S=1),
            BestPara(Stocks=150, M=0.3, m=64, S=1),
            BestPara(Stocks=150, M=0.2, m=32, S=0.85),
            BestPara(Stocks=150, M=0.1, m=32, S=0.9),
            BestPara(Stocks=100, M=1, m=64, S=0.8),
            BestPara(Stocks=100, M=0.3, m=64, S=0.8),
            BestPara(Stocks=100, M=0.2, m=32, S=0.95),
            BestPara(Stocks=100, M=0.1, m=256, S=1),
        ]
        parameters_with_data = []
        for parameter in tqdm(parameters):
            is_fit_intercept = False  # True or False
            is_center_data = True  # True or False
            percent_money_on_each_stock = parameter.M  # M = 1, 0.3, 0.2, 0.1 # if 100%, no limit on constraints
            max_feature_selected = parameter.Stocks  # None, 200, 150, 100, 50
            bootstrap_replicates = parameter.m  # None, 8, 16, 32, 64, 128
            soft_J_percentage = parameter.S  # 1, 0.9, 0.8, 0.7, 0.6
            driver = DriverIndexTrackSp500Aren(bootstrap_replicates_lst=[None, 8, 16, 32, 64, 128, 256],
                                               # # increaing list
                                               soft_J_percentage_lst=[1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6],
                                               # None if no bootstrap # decrease
                                               # do not change above
                                               # change below to export results
                                               is_fit_intercept=is_fit_intercept,
                                               is_center_data=is_center_data,
                                               percent_money_on_each_stock=percent_money_on_each_stock,
                                               max_feature_selected=max_feature_selected,
                                               # please change only above
                                               # do not need to change below
                                               n_alphas=10,  # from 0 to 1
                                               n_lambdas=100,  # from 0.001*max_lam to max_lam
                                               start_date='2020-09-01',
                                               end_date='2021-09-01',
                                               train_size=0.7,
                                               val_size=0.2,
                                               test_size=0.1)
            parameter.Obj = {}
            x_test, y_test_pred, y_test_real = driver.get_price(bootstrap_replicates=bootstrap_replicates,
                                                                soft_J_percentage=soft_J_percentage)
            parameter.Obj["bootstrap"] = (x_test, y_test_pred, y_test_real)
            x_test, y_test_pred, y_test_real = driver.get_price(bootstrap_replicates=None, soft_J_percentage=None)
            parameter.Obj["aren"] = (x_test, y_test_pred, y_test_real)
            parameters_with_data.append(parameter)
        with open(filename, 'wb') as handle:
            pickle.dump(parameters_with_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, 'rb') as handle:
            parameters_with_data = pickle.load(handle)
    plt.rcParams["figure.figsize"] = (8, 10)
    fig = plt.figure()
    # parameters_with_data = parameters_with_data[0:4] #+ parameters_with_data[-1:]
    i = 1
    for parameter in tqdm(parameters_with_data):
        plt.subplot(4, 4, i)
        # plt.subplot(1, 4, i)
        percent_money_on_each_stock = parameter.M
        max_feature_selected = parameter.Stocks
        is_plot_acc_over_pred = True
        if is_plot_acc_over_pred:
            x_test, y_test_pred, y_test_real = parameter.Obj["bootstrap"]
            plt.plot(x_test, y_test_real / y_test_pred, 'r-', linewidth=0.5)
            x_test, y_test_pred, y_test_real = parameter.Obj["aren"]
            plt.plot(x_test, y_test_real / y_test_pred, 'b--', linewidth=0.5)
            plt.ylim(0.987, 1.016)
        else:
            x_test, y_test_pred, y_test_real = parameter.Obj["bootstrap"]
            plt.plot(x_test, y_test_real, 'g:', linewidth=0.5)
            plt.plot(x_test, y_test_pred, 'r-', linewidth=0.5)
            x_test, y_test_pred, y_test_real = parameter.Obj["aren"]
            plt.plot(x_test, y_test_pred, 'b--', linewidth=0.5)
        # plt.ylabel(f"Stocks≤{max_feature_selected}")
        if i == 1:
            plt.ylabel(f"No limit")
        if i in [5, 9, 13]:
            plt.ylabel(f"Stocks≤{max_feature_selected}")
        if i in [13, 14, 15, 16]:
            plt.xlabel(f"M={percent_money_on_each_stock}")
        if i not in [1, 5, 9, 13]:
            plt.yticks([])
        # plt.xlabel(f"M={percent_money_on_each_stock}")
        plt.xticks([])
        i += 1
    plt.tight_layout()
    plt.show()
