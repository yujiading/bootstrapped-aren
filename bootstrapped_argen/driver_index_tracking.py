import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from generalized_elastic_net import GeneralizedElasticNet
from sklearn import linear_model
from tqdm import tqdm
from typing import Union
from bootstrapped_argen.library.bootstrapped_regressor import BootstrappedRegressor


class DriverIndexTrackSp500Aren:
    def __init__(self,
                 bootstrap_replicates,
                 n_alphas: int,  # 0 to 1
                 n_lambdas: int,  # 0.001*max_lam to max_lam
                 percent_money_on_each_stock: float = 1,
                 max_feature_selected=None,
                 start_date='2021-09-14',
                 end_date='2021-09-16',
                 train_size=0.7,
                 val_size=0.2,
                 test_size=0.1,
                 eps=1e-8):
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
        self.percent_money_on_each_stock = percent_money_on_each_stock
        self.max_feature_selected = max_feature_selected
        self.eps = eps
        self.n_samples = None
        self.n_features = None
        self.X = None  # returns
        self.y = None  # returns
        self.y_price = None
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
        self.y_price = adj_close_data['^GSPC']
        self.y = self.y_price.pct_change().iloc[1:]
        X_price = adj_close_data.drop(columns='^GSPC')
        self.X = X_price.pct_change().iloc[1:]
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

    def get_reg(self, lam_1, lam_2, lower_bound: Union[float, np.ndarray], upper_bound: Union[float, np.ndarray],
                n_features):
        # k = self.n_features + 1
        k = n_features
        sigma = np.diag([1] * k)
        wvec = np.ones(k)
        if isinstance(lower_bound, (float, int)):
            lowbo = lower_bound * np.ones(k)
        else:
            lowbo = lower_bound
        if isinstance(upper_bound, (float, int)):
            upbo = upper_bound * np.ones(k)
        else:
            upbo = upper_bound
        argen = GeneralizedElasticNet(lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec=wvec, sigma=sigma)
        return argen

    def get_lam_list(self, alpha):
        # max lam # https://stats.stackexchange.com/questions/174897/choosing-the-range-and-grid-density-for-regularization-parameter-in-lasso
        Z = np.dot(self.X.T, self.y)
        max_lam = np.max(np.abs(Z)) * 2 / alpha
        lst = np.linspace(start=0.001 * max_lam, stop=max_lam, num=self.n_lambdas, dtype=float)
        return lst

    @property
    def get_alpha_list(self):
        return np.linspace(start=0 + 1 / self.n_alphas, stop=1, num=self.n_alphas, dtype=float)

    # alpha=0.1, lam = 0.9,0.89,0.88,0.87,0.86:0;  0.8,0.85,1; 0.75:3; 0.5: 30; 0.4:50; 0.2:88; 0.1:186;0.09:191; 0.01:289; 0.001:295

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

    def bootstrapped_feature_select_all_hyperparameters(self, X_train, y_train):
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
                    elastic_net = self.get_reg(lam_1=alpha * lam, lam_2=lam * 0.5 * (1 - alpha), lower_bound=-self.eps,
                                               upper_bound=np.inf, n_features=self.n_features)
                    reg = BootstrappedRegressor(bootstrapped_feature_select_regressor=elastic_net,
                                                bootstrap_replicates=self.bootstrap_replicates,
                                                argen_fit_intercept=False,
                                                cpu=8,
                                                zero_thresh=1e-8,
                                                is_soft_J=True)
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

    def get_s0_t0(self, J, coef_min, coef_max, X_train):
        R_min = X_train.iloc[:, J].min().min()
        R_max = X_train.iloc[:, J].max().max()
        R_factor = (1 + R_min) / (1 + R_max)
        factor = self.percent_money_on_each_stock * R_factor * (len(J) - 1) / (1 - self.percent_money_on_each_stock)
        if factor < 1:
            raise ValueError('cannot use current percent_money_on_each_stock, try a bigger one')
        s1, t2 = coef_min, coef_max
        t1 = factor * s1
        s2 = t2 / factor
        d1 = t1 - s1
        d2 = t2 - s2
        if d1 >= d2:
            s0 = s1
            t0 = t1
        else:
            s0 = s2
            t0 = t2
        return s0, t0

    def bootstrapped_feature_select_fit_one_hyperparameter(self, reg: BootstrappedRegressor, X_train, y_train, X_val,
                                                           y_val, is_soft_J):
        if is_soft_J:
            J = reg.J_soft
        else:
            J = reg.J
        arls = self.get_reg(lam_1=0, lam_2=0, lower_bound=0,
                            upper_bound=np.inf, n_features=len(J))
        reg.fit(X_train, y_train, regressor=arls, fit_intercept=False, is_soft_J=is_soft_J)
        if self.percent_money_on_each_stock < 1:
            coef_max = np.max(reg.coef_)
            coef_min = np.min(reg.coef_)
            s0, t0 = self.get_s0_t0(J=J, coef_min=coef_min, coef_max=coef_max, X_train=X_train)
            # print(s0, t0)
            arls = self.get_reg(lam_1=0, lam_2=0, lower_bound=s0,
                                upper_bound=t0, n_features=len(J))
            reg.fit(X_train, y_train, regressor=arls, fit_intercept=False, is_soft_J=is_soft_J)
        # mse = reg.score(X=X_val, y=y_val)
        portfolio_return = self.get_portfolio_return(J=J, coef_=reg.coef_, X_test=X_val)
        te = self.get_daily_tracking_error(portfolio_return=portfolio_return,
                                           index_return=y_val)
        return te

    def bootstrapped_feature_select_best_hyperparameter(self, X_train, y_train, X_val, y_val, is_soft_J):
        if self.bootstrapped_reg_dict is None:
            raise ValueError('Need to run bootstrapped_feature_select_hyperparameters')
        alpha_list = self.get_alpha_list
        # lam_list = self.get_lam_list
        val_mse = np.inf
        val_reg = None
        val_lam = None
        val_alpha = None
        for alpha in tqdm(alpha_list):
            lam_list = self.get_lam_list(alpha)
            for lam in lam_list:
                reg = self.bootstrapped_reg_dict[(alpha, lam)]
                if is_soft_J:
                    J = reg.J_soft
                else:
                    J = reg.J
                if not J:  # is J is empty, skip it
                    continue
                if self.max_feature_selected is not None:
                    if self.max_feature_selected <= len(J):
                        continue
                mse = self.bootstrapped_feature_select_fit_one_hyperparameter(reg=reg, X_train=X_train, y_train=y_train,
                                                                              X_val=X_val, y_val=y_val,
                                                                              is_soft_J=is_soft_J)
                # print(f"alpha={alpha}, lam={lam}, te={mse}, subset={len(J)}")
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

    def plot_price(self, J, X_train, X_val, X_test):
        train_return_pred = self.get_portfolio_return(J=J,
                                                      coef_=self.val_reg.coef_, X_test=X_train)
        val_return_pred = self.get_portfolio_return(J=J,
                                                    coef_=self.val_reg.coef_, X_test=X_val)
        test_return_pred = self.get_portfolio_return(J=J,
                                                     coef_=self.val_reg.coef_, X_test=X_test)
        return_pred = np.concatenate((train_return_pred, val_return_pred, test_return_pred))
        price_pred = np.concatenate(([self.y_price[0]], (return_pred + 1) * self.y_price[:-1]))
        nreal = len(train_return_pred) + 1
        npred = len(test_return_pred) + len(val_return_pred)
        plt.plot(range(nreal + npred), self.y_price, 'b--', linewidth=0.7)
        plt.plot(range(nreal), price_pred[:nreal], 'g--', linewidth=0.7)
        plt.plot(range(nreal, npred + nreal), price_pred[nreal:nreal + npred], 'r--', linewidth=0.7)
        plt.show()

    def run(self, is_soft_J, is_plot_price):
        X_train, y_train, X_val, y_val, X_test, y_test = self.train_test_val_split
        self.bootstrapped_feature_select_all_hyperparameters(X_train, y_train)
        self.bootstrapped_feature_select_best_hyperparameter(X_train, y_train, X_val, y_val, is_soft_J=is_soft_J)
        # print(f"min_coef={np.min(self.val_reg.coef_)}, max_coef={np.max(self.val_reg.coef_)}")
        mse = self.val_reg.score(X=X_test, y=y_test, is_soft_J=is_soft_J)
        if is_soft_J:
            J = self.val_reg.J_soft
        else:
            J = self.val_reg.J
        size_J = len(J)
        portfolio_return = self.get_portfolio_return(J=J,
                                                     coef_=self.val_reg.coef_, X_test=X_test)
        daily_tracking_error = self.get_daily_tracking_error(portfolio_return=portfolio_return, index_return=y_test)
        if is_plot_price:
            self.plot_price(J=J, X_train=X_train, X_val=X_val, X_test=X_test)
        return size_J, mse, daily_tracking_error

    # def load_saved_date_fix_alpha(self, alpha):
    #     data_dir_path = pathlib.Path(__file__).parent / '../data'
    #     filename = data_dir_path / f'bodict_{alpha}alpha_{self.n_lambdas}lams_{self.n_alphas}alphas_{self.start_date}_{self.end_date}_train{self.train_size}_val{self.val_size}_test{self.test_size}_low{self.fit_low_bound}_up{self.fit_up_bound}_max{self.max_feature_selected}.pickle'
    #     file_path = pathlib.Path(filename)
    #     if not file_path.exists():
    #         print('file does not exist')
    #         X_train, y_train, X_val, y_val, X_test, y_test = self.train_test_val_split
    #         bootstrap_replicates_list = ['None', 8, 16, 32, 64, 128]
    #         result_dict = {}
    #         for bootstrap_replicates in tqdm(bootstrap_replicates_list):
    #             data_dir_path = pathlib.Path(__file__).parent / '../data'
    #             sourcename = data_dir_path / f'bodict_borepli{bootstrap_replicates}_{self.n_lambdas}lams_{self.n_alphas}alphas_{self.start_date}_{self.end_date}_train{self.train_size}_val{self.val_size}_test{self.test_size}.pickle'
    #             file_path = pathlib.Path(sourcename)
    #             if not file_path.exists():
    #                 raise ValueError(f'{sourcename} does not exist...')
    #             else:
    #                 print('file exists and load...')
    #                 with open(sourcename, 'rb') as handle:
    #                     bootstrapped_reg_dict = pickle.load(handle)
    #                 result_dict[bootstrap_replicates] = {}
    #                 dict_keys = list(bootstrapped_reg_dict.keys())
    #                 lam_list = [x[1] for x in dict_keys if x[0] == alpha]
    #                 result_dict[bootstrap_replicates]['lam_list'] = lam_list
    #                 mse_list = []
    #                 J_list = []
    #                 for lam in tqdm(lam_list):
    #                     reg = bootstrapped_reg_dict[(alpha, lam)]
    #                     J_list.append(reg.J)
    #                     mse = self.bootstrapped_feature_select_fit_one_hyperparameter(reg=reg, X_train=X_train,
    #                                                                                   y_train=y_train,
    #                                                                                   X_val=X_val, y_val=y_val)
    #                     mse_list.append(mse)
    #                 result_dict[bootstrap_replicates]['mse_list'] = mse_list
    #                 result_dict[bootstrap_replicates]['J_list'] = J_list
    #         with open(filename, 'wb') as handle:
    #             pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     else:
    #         print('file exists and load...')
    #         with open(filename, 'rb') as handle:
    #             result_dict = pickle.load(handle)
    #     return result_dict
    #
    # def plot_saved_data_fixed_alpha(self, alpha):
    #     result_dict = self.load_saved_date_fix_alpha(alpha=alpha)
    #     for bootstrap_replicates in result_dict:
    #         x = np.log(result_dict[bootstrap_replicates]['lam_list'])
    #         y = result_dict[bootstrap_replicates]['mse_list']
    #         plt.plot(x, y, label=f'm={bootstrap_replicates}', linewidth=0.8)
    #     plt.legend()
    #     plt.xlabel(f'$\\ln\\lambda, \\alpha={alpha}$')
    #     plt.ylabel('Tracking error')
    #     plt.show()
    #
    #     for bootstrap_replicates in result_dict:
    #         J_list = result_dict[bootstrap_replicates]['J_list']
    #         x = np.log(result_dict[bootstrap_replicates]['lam_list'])
    #         y = [len(item) for item in J_list]
    #         plt.plot(x, y, label=f'm={bootstrap_replicates}', linewidth=0.8)
    #     plt.legend()
    #     plt.xlabel(f'$\\ln\\lambda, \\alpha={alpha}$')
    #     plt.ylabel('Number of selected stocks')
    #     plt.show()
    #
    #     for bootstrap_replicates in result_dict:
    #         J_list = result_dict[bootstrap_replicates]['J_list']
    #         te_list = result_dict[bootstrap_replicates]['mse_list']
    #         x = np.log(result_dict[bootstrap_replicates]['lam_list'])
    #         y = [len(J_list[i]) * te_list[i] for i in range(len(J_list))]
    #         plt.plot(x, y, label=f'm={bootstrap_replicates}', linewidth=0.8)
    #     plt.legend()
    #     plt.xlabel(f'$\\ln\\lambda, \\alpha={alpha}$')
    #     plt.ylabel('Number of selected stocks times tracking error')
    #     plt.show()
    #
    # @property
    # def load_saved_date_best_J(self):
    #     data_dir_path = pathlib.Path(__file__).parent / '../data'
    #     filename = data_dir_path / f'bodict_bestJ_{self.n_lambdas}lams_{self.n_alphas}alphas_{self.start_date}_{self.end_date}_train{self.train_size}_val{self.val_size}_test{self.test_size}_low{self.fit_low_bound}_up{self.fit_up_bound}_max{self.max_feature_selected}.pickle'
    #     file_path = pathlib.Path(filename)
    #     if not file_path.exists():
    #         print('file does not exist')
    #         X_train, y_train, X_val, y_val, X_test, y_test = self.train_test_val_split
    #         bootstrap_replicates_list = ['None', 8, 16, 32, 64, 128]
    #         result_dict = {}
    #         for bootstrap_replicates in tqdm(bootstrap_replicates_list):
    #             data_dir_path = pathlib.Path(__file__).parent / '../data'
    #             sourcename = data_dir_path / f'bodict_borepli{bootstrap_replicates}_{self.n_lambdas}lams_{self.n_alphas}alphas_{self.start_date}_{self.end_date}_train{self.train_size}_val{self.val_size}_test{self.test_size}.pickle'
    #             file_path = pathlib.Path(sourcename)
    #             if not file_path.exists():
    #                 raise ValueError(f'{sourcename} does not exist...')
    #             else:
    #                 print('file exists and load...')
    #                 with open(sourcename, 'rb') as handle:
    #                     bootstrapped_reg_dict = pickle.load(handle)
    #                 dict_keys = list(bootstrapped_reg_dict.keys())
    #                 val_mse = np.inf
    #                 val_reg = None
    #                 for alpha, lam in dict_keys:
    #                     reg = bootstrapped_reg_dict[(alpha, lam)]
    #                     if self.max_feature_selected is not None:
    #                         if self.max_feature_selected <= len(reg.J):
    #                             continue
    #                     mse = self.bootstrapped_feature_select_fit_one_hyperparameter(reg=reg, X_train=X_train,
    #                                                                                   y_train=y_train,
    #                                                                                   X_val=X_val, y_val=y_val)
    #                     print(f"alpha={alpha}, lam=e^{math.log(lam)}, te={mse}, subset={len(reg.J)}")
    #                     if mse < val_mse:
    #                         val_mse = mse
    #                         val_reg = reg
    #                 result_dict[bootstrap_replicates] = val_reg.J
    #         with open(filename, 'wb') as handle:
    #             pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     else:
    #         print('file exists and load...')
    #         with open(filename, 'rb') as handle:
    #             result_dict = pickle.load(handle)
    #     return result_dict
    #
    # def plot_saved_data_best_J(self):
    #     bootstrap_replicates_list = ['None', 8, 16, 32, 64, 128]
    #     result_dict = self.load_saved_date_best_J
    #     J_map_list = np.zeros((self.n_features, len(bootstrap_replicates_list)))
    #
    #     for col in range(len(bootstrap_replicates_list)):
    #         J = result_dict[bootstrap_replicates_list[col]]
    #         for row in range(self.n_features):
    #             if row in J:
    #                 J_map_list[row, col] = 1
    #     plt.imshow(J_map_list, cmap='hot', interpolation='nearest', aspect='auto')
    #     plt.xlabel(f'm')
    #     plt.ylabel('Stock index')
    #     plt.show()


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
