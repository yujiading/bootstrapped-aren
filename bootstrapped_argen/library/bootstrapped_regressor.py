from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
from generalized_elastic_net import GeneralizedElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from tqdm import tqdm
import copy


class BootstrappedRegressor:
    def __init__(self,
                 bootstrap_replicates: int,
                 bootstrapped_feature_select_regressor,
                 argen_fit_intercept: bool = True,
                 cpu: int = 8):
        self.bootstrap_replicates = bootstrap_replicates
        self.bootstrapped_feature_select_regressor = bootstrapped_feature_select_regressor
        self.argen_fit_intercept = argen_fit_intercept
        self.cpu = cpu
        self.intercept_ = None
        self.coef_ = None
        self.J = None

    @staticmethod
    def _get_nonzero_index(lst: List, zero_thresh=1.01e-8):
        index = [i for i, ele in enumerate(lst) if abs(ele) > zero_thresh]
        return index

    @staticmethod
    def _get_intersect_two_lists(lst_one: List, lst_two: List):
        if lst_one and lst_two:
            return list(set(lst_one) & set(lst_two))
        else:
            if lst_one:
                return lst_one
            else:
                return lst_two

    @staticmethod
    def _get_intersect_lists(lsts: List[List]):
        ret = []
        for lst in lsts:
            ret = BootstrappedRegressor._get_intersect_two_lists(lst_one=ret, lst_two=lst)
        return ret

    def get_J(self, i, X, y, is_bootstrap):
        n = X.shape[0]
        if is_bootstrap:
            X_star, y_star = resample(X, y, replace=True, n_samples=n)
        else:
            X_star = X
            y_star = y
        bootstrapped_feature_select_regressor = copy.deepcopy(self.bootstrapped_feature_select_regressor)
        if isinstance(bootstrapped_feature_select_regressor, GeneralizedElasticNet) and self.argen_fit_intercept:
            X_star = np.concatenate((np.ones((n, 1)), X_star), axis=1)
        reg = bootstrapped_feature_select_regressor.fit(X_star, y_star)
        if isinstance(bootstrapped_feature_select_regressor, GeneralizedElasticNet) and self.argen_fit_intercept:
            w = reg.coef_[1:]
        else:
            w = reg.coef_
        J_ = self._get_nonzero_index(lst=w)
        return J_

    def bootstrapped_feature_select(self, X, y):
        if self.bootstrap_replicates is None:
            J = self.get_J(X=X, y=y, is_bootstrap=False, i=1)
        else:
            parallel = True
            if parallel:
                bootstrap_iterate = list(range(self.bootstrap_replicates))
                with Pool(self.cpu) as p:
                    is_tqdm = False
                    if is_tqdm:
                        J_lst = list(tqdm(p.imap(partial(self.get_J,
                                                         X=X,
                                                         y=y,
                                                         is_bootstrap=True), bootstrap_iterate, chunksize=2),
                                          total=len(bootstrap_iterate)))
                    else:
                        J_lst = list(p.imap(partial(self.get_J,
                                                   X=X,
                                                   y=y,
                                                   is_bootstrap=True), bootstrap_iterate, chunksize=2))

                J = self._get_intersect_lists(J_lst)
            else:
                J = []
                for i in tqdm(range(self.bootstrap_replicates)):
                    J_ = self.get_J(X=X, y=y, is_bootstrap=True, i=1)
                    J = self._get_intersect_two_lists(lst_one=J, lst_two=J_)
        self.J = J
        return self

    def fit(self, X, y, regressor=None, fit_intercept: bool = True):
        if self.J is None:
            raise ValueError('Need to run bootstrapped_feature_select')
        X_J = X.iloc[:, self.J]
        if regressor is None:
            reg_ls = LinearRegression(fit_intercept=fit_intercept).fit(X_J, y)
            self.intercept_ = reg_ls.intercept_
            self.coef_ = reg_ls.coef_
        else:
            if isinstance(regressor, GeneralizedElasticNet):
                if fit_intercept:
                    n = X_J.shape[0]
                    X_J = np.concatenate((np.ones((n, 1)), X_J), axis=1)
                    reg_ls = regressor.fit(X_J, y)
                    self.intercept_ = reg_ls.coef_[0]
                    self.coef_ = reg_ls.coef_[1:]
                else:
                    reg_ls = regressor.fit(X_J, y)
                    self.intercept_ = 0
                    self.coef_ = reg_ls.coef_
            else:
                reg_ls = regressor.bootstrapped_feature_select_hyperparameters(X_J, y)
                self.intercept_ = reg_ls.intercept_
                self.coef_ = reg_ls.coef_
        return self

    def predict(self, X):
        """
        X has same columns as X in fit
        """
        if self.J is None:
            raise ValueError('Need to run bootstrapped_feature_select')
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError('Need to run fit')
        X_J = X.iloc[:, self.J]
        return np.matmul(X_J, self.coef_) + self.intercept_

    def score(self, X, y):
        """
        X has same columns as X in fit
        """
        y_pred = self.predict(X=X)
        mse = mean_squared_error(y, y_pred)
        return mse
