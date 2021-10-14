from bootstrapped_argen.driver_index_tracking import DriverIndexTrackSp500Aren, DriverIndexTrackSp500Lasso


def test_driver_index_track_sp500_aren():
    driver = DriverIndexTrackSp500Aren(bootstrap_replicates=None,
                                       n_alphas=10,  # from 0 to 1
                                       n_lambdas=15,  # from e^1 to e^15
                                       start_date='2020-09-01',
                                       end_date='2021-09-01',
                                       train_size=0.7,
                                       val_size=0.2,
                                       test_size=0.1)
    mse, portfolio_return, cumulative_return, annual_average_return, \
    annual_volatility, daily_tracking_error = driver.run
    print(f"portfolio_return is {portfolio_return}")
    print(f"mse is {mse}")
    print(f"cumulative_return is {cumulative_return}")
    print(f"annual_average_return is {annual_average_return}")
    print(f"annual_volatility is {annual_volatility}")
    print(f"daily_tracking_error is {daily_tracking_error}")


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
