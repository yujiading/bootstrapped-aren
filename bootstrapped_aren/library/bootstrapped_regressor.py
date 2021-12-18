import copy
from functools import partial
from multiprocessing import Pool
from typing import List, Union

import numpy as np
import pandas as pd
from generalized_elastic_net import GeneralizedElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
from tqdm import tqdm


class BootstrappedRegressor:
    def __init__(self,
                 bootstrap_replicates_lst: List,
                 bootstrapped_feature_select_regressor,
                 soft_J_percentage_lst: List = None,  # None if no bootstrapping
                 argen_fit_intercept: bool = True,
                 cpu: int = 8,
                 zero_thresh=1e-8):
        self.bootstrap_replicates_lst = bootstrap_replicates_lst
        self.bootstrapped_feature_select_regressor = bootstrapped_feature_select_regressor
        self.soft_J_percentage_lst = soft_J_percentage_lst
        self.argen_fit_intercept = argen_fit_intercept
        self.cpu = cpu
        self.zero_thresh = zero_thresh
        self.intercept_ = None
        self.coef_ = None
        self.J = None

    def _get_nonzero_index(self, lst: List):
        index = [i for i, ele in enumerate(lst) if abs(ele) > self.zero_thresh]
        return index

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

    def get_J(self, i: int, X: pd.DataFrame, y: pd.Series, is_bootstrap: bool):
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

    def bootstrapped_feature_select(self, X: pd.DataFrame, y: pd.Series):
        self.J = {}  # dictionary with key (bootstrap_replicates, soft_J_percentage)
        if None in self.bootstrap_replicates_lst:
            # no bootstrapping, so (bootstrap_replicates,soft_J_percentage) = (None,None)
            self.J[None, None] = self.get_J(X=X, y=y, is_bootstrap=False, i=1)
        bootstrap_replicates_lst_remove_none = [x for x in self.bootstrap_replicates_lst if x is not None]
        if bootstrap_replicates_lst_remove_none:
            bootstrap_replicates_max = max(bootstrap_replicates_lst_remove_none)
            parallel = True
            if parallel:
                bootstrap_iterate = list(range(bootstrap_replicates_max))
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
                for i in tqdm(range(bootstrap_replicates_max)):
                    J_ = self.get_J(X=X, y=y, is_bootstrap=True, i=1)
                    J_lst.append(J_)
            for bootstrap_replicates in bootstrap_replicates_lst_remove_none:
                for soft_J_percentage in self.soft_J_percentage_lst:
                    self.J[bootstrap_replicates, soft_J_percentage] = self._get_intersect_lists(
                        lsts=J_lst[:bootstrap_replicates],
                        percentage=soft_J_percentage)
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series, bootstrap_replicates: Union[int, None],
            soft_J_percentage: float = None,
            regressor=None, fit_intercept: bool = True):
        if self.J is None:
            raise ValueError('Need to run bootstrapped_feature_select')
        J = self.J[bootstrap_replicates, soft_J_percentage]
        if not J:
            print("J is empty")
            self.coef_ = None
            self.intercept_ = None
            return self
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

    def predict(self, X: pd.DataFrame, bootstrap_replicates: int, soft_J_percentage: float = None):
        """
        X has same columns as X in fit
        """
        if self.J is None:
            raise ValueError('Need to run bootstrapped_feature_select')
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError('Need to run fit')
        J = self.J[bootstrap_replicates, soft_J_percentage]
        X_J = X.iloc[:, J]
        return np.matmul(X_J, self.coef_) + self.intercept_

    def score(self, X: pd.DataFrame, y: pd.Series, bootstrap_replicates: int, soft_J_percentage: float = None):
        """
        X has same columns as X in fit
        """
        y_pred = self.predict(X=X, bootstrap_replicates=bootstrap_replicates, soft_J_percentage=soft_J_percentage)
        mse = mean_squared_error(y, y_pred)
        return mse
