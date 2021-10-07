import itertools
import math
import pathlib
from operator import add
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from generalized_elastic_net import GeneralizedElasticNet
from sklearn import linear_model
from tqdm import tqdm

from bootstrapped_argen.library.bootstrapped_regressor import BootstrappedRegressor
from bootstrapped_argen.library.simulation import Simulation


class DriverSimulation:
    def __init__(self, n_features: int, nonzero_index: int, n_samples: int, bootstrap_replicates, n_replicates: int,
                 lam_1_lst: List,
                 lam_2_lst: List):
        """
        nonzero_index: relevent variables with indices less than r
        bootstrap_replicates: number of bootstrap replications, if None, do not apply bootstrapping
        """
        self.n_features = n_features
        self.nonzero_index = nonzero_index
        self.n_samples = n_samples
        self.bootstrap_replicates = bootstrap_replicates
        self.lam_1_lst = lam_1_lst
        self.lam_2_lst = lam_2_lst
        self.n_replicates = n_replicates

    def get_data(self, is_consistent):
        while True:
            sim = Simulation(n_features=self.n_features, nonzero_index=self.nonzero_index, n_samples=self.n_samples)
            X, y, w, eps, is_cons = sim.run
            if is_cons is is_consistent:
                return X, y, w

    def get_reg(self, lam_1, lam_2):
        k = self.n_features + 1
        sigma = np.diag([1] * k)
        wvec = np.ones(k)
        lowbo = -1e5 * np.ones(k)
        upbo = np.inf * np.ones(k)
        argen = GeneralizedElasticNet(lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec=wvec, sigma=sigma)
        return argen

    def J_to_J_count(self, J):
        J_count = [0] * self.n_features
        for index in J:
            J_count[index] += 1
        return J_count

    @staticmethod
    def add_lists(list1, list2):
        return list(map(add, list1, list2))

    def run(self, is_consistent):
        parameter_grid = list(itertools.product(self.lam_1_lst, self.lam_2_lst))
        J_count_dict = {key: [0] * self.n_features for key in parameter_grid}
        for i in tqdm(range(self.n_replicates)):
            X, y, w = self.get_data(is_consistent=is_consistent)
            # print(w)
            for lam_1, lam_2 in parameter_grid:
                argen = self.get_reg(lam_1=lam_1, lam_2=lam_2)
                reg = BootstrappedRegressor(regressor=argen, bootstrap_replicates=self.bootstrap_replicates)
                reg.fit_subset(X, y)
                J_count = self.J_to_J_count(J=reg.J)
                J_count_dict[(lam_1, lam_2)] = self.add_lists(J_count_dict[(lam_1, lam_2)], J_count)
        J_prob_dict = {key: [item / self.n_replicates for item in value] for key, value in J_count_dict.items()}
        J_prob_array = np.array(list(J_prob_dict.values())).T
        plt.imshow(J_prob_array, cmap='gray')
        plt.xticks(range(len(parameter_grid)), np.round(parameter_grid, 2))
        plt.yticks(range(self.n_features), list(range(self.n_features)))
        plt.xlabel('parameters')
        plt.ylabel('variable index')
        plt.title(f'consistent: {is_consistent}, boostrap: {self.bootstrap_replicates}')
        plt.show()
        return J_prob_dict


class DriverIndexTrackSp500Aren:
    def __init__(self, bootstrap_replicates, n_lambdas: int, n_alphas: int, start_date='2021-09-14',
                 end_date='2021-09-16', train_size=0.7, val_size=0.2, test_size=0.1, lower_bound=-1e5,
                 upper_bound=np.inf):
        """
        bootstrap_replicates: number of bootstrap replications, if None, do not apply bootstrapping

        lambda * (alpha * ||w||_1 + 0.5 * (1-alpha) * ||w||_2^2)

        lambda = e^, 0<=alpha<=1
        """
        self.bootstrap_replicates = bootstrap_replicates
        self.n_lambdas = n_lambdas
        self.n_alphas = n_alphas
        self.start_date = start_date
        self.end_date = end_date
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_samples = None
        self.n_features = None
        self.X = None
        self.y = None
        self.val_reg = None
        self.val_mse = None
        self.val_lam = None
        self.val_alpha = None
        self.get_data()

    def get_data(self):
        full_period = '5y'  # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        data_dir_path = pathlib.Path(__file__).parent / '../data'
        filename = data_dir_path / f'sp500_{full_period}.csv'
        file_path = pathlib.Path(filename)
        if not file_path.exists():
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            df = table[0]
            stock_symbol = df['Symbol'].to_list()
            stock_symbol = ['^GSPC'] + stock_symbol
            full_stock_data = yf.download(tickers=stock_symbol, period=full_period)
            adj_close_data = full_stock_data['Adj Close']
            adj_close_data.to_csv(filename)
        else:
            adj_close_data = pd.read_csv(filename, index_col=0)
        adj_close_data = adj_close_data[self.start_date:self.end_date]
        adj_close_data = adj_close_data.dropna(axis='index', how='all').dropna(axis='columns')
        self.y = adj_close_data['^GSPC']
        self.X = adj_close_data.drop(columns='^GSPC')
        self.n_samples, self.n_features = self.X.shape
        print(f"Get {self.n_samples} days' stock prices for sp500 and {self.n_features} companies")

    @property
    def train_test_val_split(self):
        n_samples_train = int(self.n_samples * self.train_size)
        n_samples_val = int(self.n_samples * self.val_size)
        X_train = self.X.iloc[: n_samples_train, :]
        y_train = self.y.iloc[: n_samples_train]
        X_val = self.X.iloc[n_samples_train:n_samples_train + n_samples_val, :]
        y_val = self.y.iloc[n_samples_train:n_samples_train + n_samples_val]
        X_test = self.X.iloc[n_samples_train + n_samples_val:, :]
        y_test = self.y.iloc[n_samples_train + n_samples_val:]
        return X_train, y_train, X_val, y_val, X_test, y_test

    def get_reg(self, lam_1, lam_2, lower_bound, upper_bound, n_features):
        # k = self.n_features + 1
        k = n_features
        sigma = np.diag([1] * k)
        wvec = np.ones(k)
        lowbo = lower_bound * np.ones(k)
        upbo = upper_bound * np.ones(k)
        argen = GeneralizedElasticNet(lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec=wvec, sigma=sigma)
        return argen

    @property
    def get_lam_list(self):
        # start_power = -1
        # stop_power = 5
        # if self.n_lambdas <= stop_power + 2:
        #     lst = np.linspace(start=start_power, stop=stop_power, num=self.n_lambdas, dtype=int)
        # else:
        #     lst = np.arange(-1, self.n_lambdas - 1)
        # return math.e ** lst
        return [400000]

    # total 298 10000 297, 50000  206, 70000, 177, 90000, 155, 110000 141, 200000 88, 400000 45

    @property
    def get_alpha_list(self):
        # return np.linspace(start=0 + 1 / self.n_alphas, stop=1, num=self.n_alphas, dtype=float)
        return [1]

    def fit(self, X_train, y_train, X_val, y_val):
        alpha_list = self.get_alpha_list
        lam_list = self.get_lam_list
        val_mse = np.inf
        val_reg = None
        val_lam = None
        val_alpha = None
        for alpha in tqdm(alpha_list):
            for lam in lam_list:
                elastic_net = self.get_reg(lam_1=alpha * lam, lam_2=lam * 0.5 * (1 - alpha), lower_bound=-1e5,
                                           upper_bound=np.inf, n_features=self.n_features)
                reg = BootstrappedRegressor(regressor=elastic_net, bootstrap_replicates=self.bootstrap_replicates,
                                            aren_fit_intercept=False)

                reg.fit_subset(X_train, y_train)
                arls = self.get_reg(lam_1=0, lam_2=0, lower_bound=self.lower_bound,
                                    upper_bound=self.upper_bound, n_features=len(reg.J))
                reg.fit_regression(X_train, y_train, regressor=arls, fit_intercept=False)
                mse = reg.score(X=X_val, y=y_val)
                # portfolio_return = self.get_portfolio_return(J=reg.J, coef_=reg.coef_, X_test=X_val)
                # mse = self.get_daily_tracking_error(portfolio_return=portfolio_return,
                #                                     index_return=y_val)
                if mse < val_mse:
                    val_mse = mse
                    val_reg = reg
                    val_lam = lam
                    val_alpha = alpha
        self.val_reg = val_reg
        self.val_mse = val_mse
        self.val_lam = val_lam
        self.val_alpha = val_alpha
        return self

    @staticmethod
    def get_portfolio_return(J, coef_, X_test):
        X_test = X_test.iloc[:, J]
        coef_ = coef_ / sum(coef_)
        portfolio_return = np.matmul(X_test, coef_)
        return portfolio_return

    @staticmethod
    def get_cumulative_return(portfolio_return):
        one_plus_return = 1 + portfolio_return
        return np.cumprod(one_plus_return)[-1] - 1

    @staticmethod
    def get_annual_average_return(portfolio_return):
        average_return = np.average(portfolio_return)
        return (1 + average_return) ** 252 - 1

    @staticmethod
    def get_annual_volatility(portfolio_return):
        return np.sqrt(252) * np.std(portfolio_return)

    @staticmethod
    def get_daily_tracking_error(portfolio_return, index_return):
        assert len(portfolio_return) == len(
            index_return), "two vectors need to be the same length"
        excess_return = portfolio_return - index_return
        res = np.std(excess_return)
        return res

    @staticmethod
    def get_daily_tracking_error_volatility(portfolio_return, index_return):
        assert len(portfolio_return) == len(
            index_return), "two vectors need to be the same length"
        return np.std(np.abs(portfolio_return - index_return))

    @property
    def run(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.train_test_val_split
        self.fit(X_train, y_train, X_val, y_val)
        mse = self.val_reg.score(X=X_test, y=y_test)
        portfolio_return = self.get_portfolio_return(J=self.val_reg.J,
                                                     coef_=self.val_reg.coef_, X_test=X_test)
        cumulative_return = self.get_cumulative_return(portfolio_return=portfolio_return)
        annual_average_return = self.get_annual_average_return(portfolio_return=portfolio_return)
        annual_volatility = self.get_annual_volatility(portfolio_return=portfolio_return)
        daily_tracking_error = self.get_daily_tracking_error(portfolio_return=portfolio_return, index_return=y_test)
        daily_tracking_error_volatility = self.get_daily_tracking_error_volatility(portfolio_return=portfolio_return,
                                                                                   index_return=y_test)
        return mse, portfolio_return, cumulative_return, annual_average_return, annual_volatility, daily_tracking_error, daily_tracking_error_volatility


class DriverIndexTrackSp500Lasso(DriverIndexTrackSp500Aren):
    def __init__(self, bootstrap_replicates, n_lambdas: int, start_date='2021-09-14',
                 end_date='2021-09-16', train_size=0.7, val_size=0.2, test_size=0.1):
        """
        bootstrap_replicates: number of bootstrap replications, if None, do not apply bootstrapping

        lambda * ||w||_1

        lambda = e^
        """
        super().__init__(bootstrap_replicates=bootstrap_replicates, n_lambdas=n_lambdas, n_alphas=1,
                         start_date=start_date, end_date=end_date, train_size=train_size, val_size=val_size,
                         test_size=test_size, lower_bound=None, upper_bound=None)

    def get_reg(self, lam_1, lam_2, lower_bound, upper_bound):
        lasso = linear_model.Lasso(alpha=lam_1, fit_intercept=False)
        return lasso

    @property
    def get_lam_list(self):
        if self.n_lambdas <= 7:
            lst = np.linspace(start=-1, stop=5, num=self.n_lambdas, dtype=int)
        else:
            lst = np.arange(-1, self.n_lambdas - 1)
        return math.e ** lst
