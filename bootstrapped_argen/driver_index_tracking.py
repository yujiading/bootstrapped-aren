import math
import pathlib
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from generalized_elastic_net import GeneralizedElasticNet
from sklearn import linear_model
from tqdm import tqdm

from bootstrapped_argen.library.bootstrapped_regressor import BootstrappedRegressor


class DriverIndexTrackSp500Aren:
    def __init__(self,
                 bootstrap_replicates,
                 n_alphas: int,
                 n_lambdas: int = None,
                 start_date='2021-09-14',
                 end_date='2021-09-16',
                 train_size=0.7,
                 val_size=0.2,
                 test_size=0.1):
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
        self.n_samples = None
        self.n_features = None
        self.X = None
        self.y = None
        self.val_reg = None
        self.val_mse = None
        self.val_lam = None
        self.val_alpha = None
        self.bootstrapped_reg_dict = None
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

    def get_lam_list(self, alpha):
        start_power = 10
        stop_power = 13 - int(math.log(alpha))
        if self.n_lambdas is None:
            n_lambdas = (stop_power - start_power) * 4 + 1
        else:
            n_lambdas = self.n_lambdas
        lst = np.linspace(start=start_power, stop=stop_power, num=n_lambdas, dtype=float)
        return math.e ** lst

    # total 298, 10000 297, 50000  206, 70000, 177, 90000, 155, 110000 141, 200000 88, 400000 45, 4000000 6

    @property
    def get_alpha_list(self):
        return np.linspace(start=0 + 1 / self.n_alphas, stop=1, num=self.n_alphas, dtype=float)

    @staticmethod
    def get_portfolio_return(J, coef_, X_test):
        X_test = X_test.iloc[:, J]
        # coef_ = coef_ / sum(coef_)
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

    def bootstrapped_feature_select_hyperparameters(self, X_train, y_train):
        data_dir_path = pathlib.Path(__file__).parent / '../data'
        filename = data_dir_path / f'bodict_borepli{self.bootstrap_replicates}_{self.n_lambdas}lams_{self.n_alphas}alphas_{self.start_date}_{self.end_date}_train{self.train_size}_val{self.val_size}_test{self.test_size}.pickle'
        file_path = pathlib.Path(filename)
        if not file_path.exists():
            print('file does not exist')
            bootstrapped_reg_dict = {}
            alpha_list = self.get_alpha_list
            for alpha in tqdm(alpha_list, total=len(alpha_list)):
                lam_list = self.get_lam_list(alpha=alpha)
                for lam in tqdm(lam_list, total=len(lam_list)):
                    elastic_net = self.get_reg(lam_1=alpha * lam, lam_2=lam * 0.5 * (1 - alpha), lower_bound=-1e5,
                                               upper_bound=np.inf, n_features=self.n_features)
                    reg = BootstrappedRegressor(bootstrapped_feature_select_regressor=elastic_net,
                                                bootstrap_replicates=self.bootstrap_replicates,
                                                argen_fit_intercept=False)
                    reg.bootstrapped_feature_select(X_train, y_train)
                    bootstrapped_reg_dict[(alpha, lam)] = reg
            self.bootstrapped_reg_dict = bootstrapped_reg_dict
            with open(filename, 'wb') as handle:
                pickle.dump(bootstrapped_reg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print('file exists and load...')
            with open(filename, 'rb') as handle:
                self.bootstrapped_reg_dict = pickle.load(handle)
        return self

    def bootstrapped_feature_select_best_hyperparameter(self, X_train, y_train, X_val, y_val):
        if self.bootstrapped_reg_dict is None:
            raise ValueError('Need to run bootstrapped_feature_select_hyperparameters')
        alpha_list = self.get_alpha_list
        # lam_list = self.get_lam_list
        val_mse = np.inf
        val_reg = None
        val_lam = None
        val_alpha = None
        for alpha in alpha_list:
            lam_list = self.get_lam_list(alpha)
            for lam in lam_list:
                reg = self.bootstrapped_reg_dict[(alpha, lam)]
                arls = self.get_reg(lam_1=0, lam_2=0, lower_bound=0,
                                    upper_bound=np.inf, n_features=len(reg.J))
                reg.fit(X_train, y_train, regressor=arls, fit_intercept=False)
                # mse = reg.score(X=X_val, y=y_val)
                portfolio_return = self.get_portfolio_return(J=reg.J, coef_=reg.coef_, X_test=X_val)
                mse = self.get_daily_tracking_error(portfolio_return=portfolio_return,
                                                    index_return=y_val)
                print(f"alpha={alpha}, lam=e^{math.log(lam)}, te={mse}, subset={len(reg.J)}")
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

    @property
    def run(self):
        X_train, y_train, X_val, y_val, X_test, y_test = self.train_test_val_split
        self.bootstrapped_feature_select_hyperparameters(X_train, y_train)
        self.bootstrapped_feature_select_best_hyperparameter(X_train, y_train, X_val, y_val)
        mse = self.val_reg.score(X=X_test, y=y_test)
        print(len(self.val_reg.J))
        portfolio_return = self.get_portfolio_return(J=self.val_reg.J,
                                                     coef_=self.val_reg.coef_, X_test=X_test)
        cumulative_return = self.get_cumulative_return(portfolio_return=portfolio_return)
        annual_average_return = self.get_annual_average_return(portfolio_return=portfolio_return)
        annual_volatility = self.get_annual_volatility(portfolio_return=portfolio_return)
        daily_tracking_error = self.get_daily_tracking_error(portfolio_return=portfolio_return, index_return=y_test)
        return mse, portfolio_return, cumulative_return, annual_average_return, annual_volatility, daily_tracking_error


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
                         test_size=test_size)

    def get_reg(self, lam_1, lam_2, lower_bound, upper_bound, n_features):
        lasso = linear_model.Lasso(alpha=lam_1, fit_intercept=False)
        return lasso
