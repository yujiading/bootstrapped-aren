import pathlib

from bootstrapped_argen.driver_index_tracking import DriverIndexTrackSp500Aren


def test_driver_index_track_sp500_aren_saving_data():
    driver = DriverIndexTrackSp500Aren(bootstrap_replicates_lst=[None, 8, 16, 32, 64, 128],  # increase
                                       soft_J_percentage_lst=[1, 0.9, 0.8, 0.7, 0.6],  # None if no bootstrap # decrease
                                       is_fit_intercept=True,
                                       # above settings are for saving and loading data
                                       # please change only above
                                       # do not need to change below
                                       percent_money_on_each_stock=1,  # if 100%, no limit on constraints
                                       max_feature_selected=None,
                                       # please change only above
                                       # do not need to change below
                                       n_alphas=10,  # from 0 to 1
                                       n_lambdas=100,  # from 0.001*max_lam to max_lam
                                       start_date='2020-09-01',
                                       end_date='2021-09-01',
                                       train_size=0.7,
                                       val_size=0.2,
                                       test_size=0.1)

    _ = driver.run_once(bootstrap_replicates=None)


def test_driver_index_track_sp500_aren_exportung_results():
    is_fit_intercept = True  # True or False
    # percent_money_on_each_stock = 0.1  # M = 1, 0.3, 0.2, 0.1 # if 100%, no limit on constraints
    # max_feature_selected = None  # None, 200, 150, 100, 50

    for max_feature_selected in [None, 200, 150, 100, 50]:
        for percent_money_on_each_stock in [1, 0.3, 0.2, 0.1]:
            driver = DriverIndexTrackSp500Aren(bootstrap_replicates_lst=[None, 8, 16, 32, 64, 128],
                                               soft_J_percentage_lst=[1, 0.9, 0.8, 0.7, 0.6],
                                               # None if no bootstrap # decrease
                                               # do not change above
                                               # change below to export results
                                               is_fit_intercept=is_fit_intercept,
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

            table, latex = driver.run_all()
            data_dir_path = pathlib.Path(__file__).parent / '../data'
            filename = data_dir_path / f'out_intercept{is_fit_intercept}_M{percent_money_on_each_stock}_max{max_feature_selected}.csv'
            file_path = pathlib.Path(filename)
            table.to_csv(file_path)


def test_driver_index_track_sp500_aren_plot():
    is_fit_intercept = False  # True or False
    percent_money_on_each_stock = 0.3  # M = 1, 0.3, 0.2, 0.1 # if 100%, no limit on constraints
    max_feature_selected = None  # None, 200, 150, 100, 50
    bootstrap_replicates = 32  # None, 8, 16, 32, 64, 128
    soft_J_percentage = 0.8  # 1, 0.9, 0.8, 0.7, 0.6
    driver = DriverIndexTrackSp500Aren(bootstrap_replicates_lst=[None, 8, 16, 32, 64, 128],  # # increaing list
                                       soft_J_percentage_lst=[1, 0.9, 0.8, 0.7, 0.6],  # None if no bootstrap # decrease
                                       # do not change above
                                       # change below to export results
                                       is_fit_intercept=is_fit_intercept,
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

    _, _, _ = driver.run_once(bootstrap_replicates=bootstrap_replicates, soft_J_percentage=soft_J_percentage,
                              is_plot_price=True)
