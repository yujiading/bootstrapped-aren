from bootstrapped_argen.driver_index_tracking import DriverIndexTrackSp500Aren


def test_driver_index_track_sp500_aren_run():
    driver = DriverIndexTrackSp500Aren(bootstrap_replicates_lst=[None, 8, 16, 32, 64, 128, 256],  # # increaing list
                                       soft_J_percentage_lst=[1, 0.9, 0.8, 0.7, 0.6],
                                       # None if no bootstrapping # decreasing list
                                       percent_money_on_each_stock=1,  # is 100%, no limit on constraints
                                       max_feature_selected=None,
                                       is_fit_intercept=False,
                                       # please change only above
                                       # do not need to change below
                                       n_alphas=10,  # from 0 to 1
                                       n_lambdas=100,  # from 0.001*max_lam to max_lam
                                       start_date='2020-09-01',
                                       end_date='2021-09-01',
                                       train_size=0.7,
                                       val_size=0.2,
                                       test_size=0.1)

    table = driver.run_all(bootstrap_replicates_lst=None, soft_J_percentage_lst=None)
    print(table)
