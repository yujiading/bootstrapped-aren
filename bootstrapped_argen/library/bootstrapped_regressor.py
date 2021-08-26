import numpy as np
from sklearn.utils import resample
from typing import List
from sklearn.linear_model import LinearRegression


class BootstrappedRegressor:
    def __init__(self, regressor, n_replicates: int):
        self.regressor = regressor
        self.n_replicates = n_replicates
        self.intercept_ = None
        self.coef_ = None
        self.J = None

    @staticmethod
    def get_nonzero_index(lst: List):
        index = [i for i, ele in enumerate(lst) if ele != 0]
        return index

    @staticmethod
    def get_intersection(lst_one: List, lst_two: List):
        if lst_one and lst_two:
            return list(set(lst_one) & set(lst_two))
        else:
            if lst_one:
                return lst_one
            else:
                return lst_two

    def fit(self, X, y):
        n = X.shape[0]
        J = []
        for i in range(self.n_replicates):
            X_star, y_star = resample(X, y, replace=True, n_samples=n)
            reg = self.regressor.fit(X=X_star, y=y_star)
            w = reg.coef_
            J_ = self.get_nonzero_index(lst=w)
            J = self.get_intersection(lst_one=J, lst_two=J_)
        X_J = X[:, J]
        reg_ls = LinearRegression().fit(X_J, y)
        self.intercept_ = reg_ls.intercept_
        self.coef_ = reg_ls.coef_
        self.J = J
        return self

    def predict(self, X):
        return np.matmul(X, self.coef_) + self.intercept_
