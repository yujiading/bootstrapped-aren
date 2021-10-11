from bootstrapped_argen.library.bootstrapped_regressor import BootstrappedRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from generalized_elastic_net import GeneralizedElasticNet
import pytest


@pytest.fixture
def X():
    return np.array([[1, 1], [1, 2], [2, 2], [2, 3]])


@pytest.fixture
def y(X):
    # y = 1 * x_0 + 2 * x_1 + 3
    return np.dot(X, np.array([1, 2])) + 3


def test_bootstrapped_regressor_ls(X, y):
    reg = BootstrappedRegressor(bootstrapped_feature_select_regressor=LinearRegression(), bootstrap_replicates=100)
    reg.fit(X, y)
    print(reg.intercept_, reg.coef_, reg.J)


def test_bootstrapped_regressor_argen(X, y):
    K = X.shape[1] + 1
    lam_1 = 0.0034
    lam_2 = 2
    sigma = np.diag([1] * K)
    wvec = np.ones(K)
    lowbo = -1e5 * np.ones(K)
    upbo = np.inf * np.ones(K)
    argen = GeneralizedElasticNet(lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec=wvec, sigma=sigma)
    reg = BootstrappedRegressor(bootstrapped_feature_select_regressor=argen, bootstrap_replicates=100)
    reg.fit(X, y)
    print(reg.intercept_, reg.coef_, reg.J)
