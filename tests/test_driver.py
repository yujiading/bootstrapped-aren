import math

from bootstrapped_argen.driver import DriverSimulation, DriverIndexTrackSp500Aren, DriverIndexTrackSp500Lasso


def test_driver_simulation():
    lam_1_lst = [math.exp(-i) for i in range(5)]
    driver = DriverSimulation(n_features=4,
                              nonzero_index=2,
                              n_samples=10,
                              lam_1_lst=lam_1_lst,
                              lam_2_lst=[0.5],
                              n_replicates=15,
                              bootstrap_replicates=10)
    ret_dict = driver.run(is_consistent=False)
    print(ret_dict)


def test_driver_simulation_full_dimension():
    lam_1_lst = [math.exp(-i) for i in range(16)]
    driver = DriverSimulation(n_features=16,
                              nonzero_index=8,
                              n_samples=1000,
                              lam_1_lst=lam_1_lst,
                              lam_2_lst=[0.5],
                              n_replicates=256,
                              bootstrap_replicates=128)
    ret_dict = driver.run(is_consistent=False)
    print(ret_dict)


def test_driver_index_track_sp500_aren():
    driver = DriverIndexTrackSp500Aren(bootstrap_replicates=None,
                                       n_alphas=2, # from 0 to 1
                                       n_lambdas=2,  # from e^-1 to e^5
                                       start_date='2018-09-01',
                                       end_date='2021-09-01',
                                       train_size=0.7,
                                       val_size=0.2,
                                       test_size=0.1,
                                       lower_bound=0.0058,
                                       upper_bound=0.1)
    mse, portfolio_return, cumulative_return, annual_average_return, \
    annual_volatility, daily_tracking_error, daily_tracking_error_volatility = driver.run
    print(f"portfolio_return is {portfolio_return}")
    print(f"mse is {mse}")
    print(f"cumulative_return is {cumulative_return}")
    print(f"annual_average_return is {annual_average_return}")
    print(f"annual_volatility is {annual_volatility}")
    print(f"daily_tracking_error is {daily_tracking_error}")
    print(f"daily_tracking_error_volatility is {daily_tracking_error_volatility}")


def test_driver_index_track_sp500_lasso():
    driver = DriverIndexTrackSp500Lasso(bootstrap_replicates=10,
                                        n_lambdas=10,
                                        start_date='2018-09-01',
                                        end_date='2021-09-01',
                                        train_size=0.7,
                                        val_size=0.2,
                                        test_size=0.1)
    mse, portfolio_return, cumulative_return, annual_average_return, \
    annual_volatility, daily_tracking_error, daily_tracking_error_volatility = driver.run
    print(f"portfolio_return is {portfolio_return}")
    print(f"mse is {mse}")
    print(f"cumulative_return is {cumulative_return}")
    print(f"annual_average_return is {annual_average_return}")
    print(f"annual_volatility is {annual_volatility}")
    print(f"daily_tracking_error is {daily_tracking_error}")
    print(f"daily_tracking_error_volatility is {daily_tracking_error_volatility}")
