from typing import List
from functools import partial,partialmethod
import numpy as np
from generalized_elastic_net import GeneralizedElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from tqdm import tqdm
from multiprocessing import Pool


class BootstrappedRegressor:
    def __init__(self, regressor, bootstrap_replicates: int, cpu: int = 3):
        self.regressor = regressor
        self.bootstrap_replicates = bootstrap_replicates
        self.cpu = cpu
        self.intercept_ = None
        self.coef_ = None
        self.J = None

    @staticmethod
    def get_nonzero_index(lst: List):
        index = [i for i, ele in enumerate(lst) if ele != 0]
        return index

    @staticmethod
    def get_intersect_two_lists(lst_one: List, lst_two: List):
        if lst_one and lst_two:
            return list(set(lst_one) & set(lst_two))
        else:
            if lst_one:
                return lst_one
            else:
                return lst_two

    def get_intersect_lists(self, lsts: List[List]):
        ret = []
        for lst in lsts:
            ret = self.get_intersect_two_lists(lst_one=ret, lst_two=lst)
        return ret

    def get_J(self, i, X, y, is_bootstrap):
        n = X.shape[0]
        if is_bootstrap:
            X_star, y_star = resample(X, y, replace=True, n_samples=n)
        else:
            X_star = X
            y_star = y
        if isinstance(self.regressor, GeneralizedElasticNet):
            X_star = np.concatenate((np.ones((n, 1)), X_star), axis=1)
        reg = self.regressor.fit(X_star, y_star)
        if isinstance(self.regressor, GeneralizedElasticNet):
            w = reg.coef_[1:]
        else:
            w = reg.coef_
        J_ = self.get_nonzero_index(lst=w)
        return J_

    def fit(self, X, y):
        if self.bootstrap_replicates is None:
            J = self.get_J(X=X, y=y, is_bootstrap=False, i=1)
        else:
            parallel = True
            if parallel:
                bootstrap_iterate = list(range(self.bootstrap_replicates))
                with Pool(self.cpu) as p:
                    J_lst = list(tqdm(p.imap(partial(self.get_J,
                                                     X=X,
                                                     y=y,
                                                     is_bootstrap=True), bootstrap_iterate), total=len(bootstrap_iterate)))
                J = self.get_intersect_lists(J_lst)
            else:
                J = []
                for i in tqdm(range(self.bootstrap_replicates)):
                    J_ = self.get_J(X=X, y=y, is_bootstrap=True, i=1)
                    J = self.get_intersect_two_lists(lst_one=J, lst_two=J_)
        X_J = X[:, J]
        reg_ls = LinearRegression().fit(X_J, y)
        self.intercept_ = reg_ls.intercept_
        self.coef_ = reg_ls.coef_
        self.J = J
        return self

    def predict(self, X):
        return np.matmul(X, self.coef_) + self.intercept_
