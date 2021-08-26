from bootstrapped_argen.library.bootstrapped_regressor import BootstrappedRegressor
import numpy as np
from sklearn.linear_model import LinearRegression


def test_bootstrapped_regressor():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    reg = BootstrappedRegressor(regressor=LinearRegression(), n_replicates=100)
    reg.fit(X, y)
    print(reg.intercept_, reg.coef_, reg.J)
