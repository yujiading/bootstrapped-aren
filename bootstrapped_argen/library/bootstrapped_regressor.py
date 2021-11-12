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
                 cpu: int = 8,
                 zero_thresh=1e-8,
                 is_soft_J: bool = True):
        self.bootstrap_replicates = bootstrap_replicates
        self.bootstrapped_feature_select_regressor = bootstrapped_feature_select_regressor
        self.argen_fit_intercept = argen_fit_intercept
        self.cpu = cpu
        self.zero_thresh = zero_thresh
        self.is_soft_J = is_soft_J
        self.intercept_ = None
        self.coef_ = None
        self.J = None
        self.J_soft = None

    def _get_nonzero_index(self, lst: List):
        index = [i for i, ele in enumerate(lst) if abs(ele) > self.zero_thresh]
        return index

    # @staticmethod
    # def _get_intersect_two_lists(lst_one: List, lst_two: List):
    #     if lst_one and lst_two:
    #         return list(set(lst_one) & set(lst_two))
    #     else:
    #         if lst_one:
    #             return lst_one
    #         else:
    #             return lst_two
    #
    # @staticmethod
    # def _get_intersect_lists(lsts: List[List]):
    #     ret = []
    #     for lst in lsts:
    #         ret = BootstrappedRegressor._get_intersect_two_lists(lst_one=ret, lst_two=lst)
    #     return ret
    @staticmethod
    def _get_intersect_lists(lsts: List[List], percentage: float):
        length = len(lsts)
        count_dict = {}
        for lst in lsts:
            for item in lst:
                if item in count_dict:
                    count_dict[item] += 1
                else:
                    count_dict[item] = 1
        return [item for item in count_dict if count_dict[item] / length >= percentage]

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
            self.J = self.get_J(X=X, y=y, is_bootstrap=False, i=1)
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
                                                         is_bootstrap=True), bootstrap_iterate, chunksize=10),
                                          total=len(bootstrap_iterate)))
                    else:
                        J_lst = list(p.imap(partial(self.get_J,
                                                    X=X,
                                                    y=y,
                                                    is_bootstrap=True), bootstrap_iterate, chunksize=10))

            else:
                J_lst = []
                for i in tqdm(range(self.bootstrap_replicates)):
                    J_ = self.get_J(X=X, y=y, is_bootstrap=True, i=1)
                    J_lst.append(J_)
            self.J = self._get_intersect_lists(J_lst, percentage=1)
            if self.is_soft_J:
                self.J_soft = self._get_intersect_lists(J_lst, percentage=0.8)
        return self

    def fit(self, X, y, is_soft_J, regressor=None, fit_intercept: bool = True):
        if self.J is None:
            raise ValueError('Need to run bootstrapped_feature_select')
        if is_soft_J:
            J = self.J_soft
        else:
            J = self.J
        X_J = X.iloc[:, J]
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
                reg_ls = regressor.fit(X_J, y)
                self.intercept_ = reg_ls.intercept_
                self.coef_ = reg_ls.coef_
        return self

    def predict(self, X, is_soft_J):
        """
        X has same columns as X in fit
        """
        if self.J is None:
            raise ValueError('Need to run bootstrapped_feature_select')
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError('Need to run fit')
        if is_soft_J:
            J = self.J_soft
        else:
            J = self.J
        X_J = X.iloc[:, J]
        return np.matmul(X_J, self.coef_) + self.intercept_

    def score(self, X, y, is_soft_J):
        """
        X has same columns as X in fit
        """
        y_pred = self.predict(X=X, is_soft_J=is_soft_J)
        mse = mean_squared_error(y, y_pred)
        return mse
