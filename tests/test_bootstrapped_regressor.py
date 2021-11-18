import numpy as np
import pandas as pd
import pytest
from generalized_elastic_net import GeneralizedElasticNet
from sklearn.linear_model import LinearRegression

from bootstrapped_argen.library.bootstrapped_regressor import BootstrappedRegressor


@pytest.fixture
def X():
    return pd.DataFrame([[1, 1], [1, 2], [2, 2], [2, 3]])


@pytest.fixture
def y(X):
    # y = 1 * x_0 + 2 * x_1 + 3
    return pd.Series(np.dot(X, np.array([1, 2])) + 3)


def test_bootstrapped_regressor_ls(X, y):
    reg = BootstrappedRegressor(bootstrapped_feature_select_regressor=LinearRegression(),
                                bootstrap_replicates_lst=[None, 100],
                                soft_J_percentage_lst=[1, 0.9])
    reg.bootstrapped_feature_select(X=X, y=y)
    print(reg.J)
    reg.fit(X, y, bootstrap_replicates=None, soft_J_percentage=None)
    print(reg.intercept_, reg.coef_)
    reg.fit(X, y, bootstrap_replicates=100, soft_J_percentage=1)
    print(reg.intercept_, reg.coef_)
    reg.fit(X, y, bootstrap_replicates=100, soft_J_percentage=0.9)
    print(reg.intercept_, reg.coef_)


def test_bootstrapped_regressor_argen(X, y):
    K = X.shape[1] + 1
    lam_1 = 1000
    lam_2 = 0
    sigma = np.diag([1] * K)
    wvec = np.ones(K)
    lowbo = -1e5 * np.ones(K)
    upbo = np.inf * np.ones(K)
    argen = GeneralizedElasticNet(lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec=wvec, sigma=sigma)
    reg = BootstrappedRegressor(bootstrapped_feature_select_regressor=argen, bootstrap_replicates_lst=[None, 100],
                                soft_J_percentage_lst=[1, 0.9])
    reg.bootstrapped_feature_select(X=X, y=y)
    print(reg.J)
    reg.fit(X, y, bootstrap_replicates=None, soft_J_percentage=None)
    print(reg.intercept_, reg.coef_)
    reg.fit(X, y, bootstrap_replicates=100, soft_J_percentage=1)
    print(reg.intercept_, reg.coef_)
    reg.fit(X, y, bootstrap_replicates=100, soft_J_percentage=0.9)
    print(reg.intercept_, reg.coef_)
