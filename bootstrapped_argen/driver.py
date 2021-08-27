from bootstrapped_argen.library.simulation import Simulation
from bootstrapped_argen.library.bootstrapped_regressor import BootstrappedRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from generalized_elastic_net import GeneralizedElasticNet
import pytest
from typing import List
from tqdm import tqdm
import itertools
from operator import add
import matplotlib.pyplot as plt

class Driver:
    def __init__(self, n_features: int, nonzero_index: int, n_samples: int, bootstrap_replicates, n_replicates: int,
                 lam_1_lst: List,
                 lam_2_lst: List):
        """
        nonzero_index: relevent variables with indices less than r
        bootstrap_replicates: number of bootstrap replications, if None, do not apply bootstrapping
        """
        self.n_features = n_features
        self.nonzero_index = nonzero_index
        self.n_samples = n_samples
        self.bootstrap_replicates = bootstrap_replicates
        self.lam_1_lst = lam_1_lst
        self.lam_2_lst = lam_2_lst
        self.n_replicates = n_replicates

    def get_data(self, is_consistent):
        while True:
            sim = Simulation(n_features=self.n_features, nonzero_index=self.nonzero_index, n_samples=self.n_samples)
            X, y, w, eps, is_cons = sim.run
            if is_cons is is_consistent:
                return X, y, w

    def get_reg(self, lam_1, lam_2):
        k = self.n_features + 1
        sigma = np.diag([1] * k)
        wvec = np.ones(k)
        lowbo = -1e5 * np.ones(k)
        upbo = np.inf * np.ones(k)
        argen = GeneralizedElasticNet(lam_1=lam_1, lam_2=lam_2, lowbo=lowbo, upbo=upbo, wvec=wvec, sigma=sigma)
        return argen

    def J_to_J_count(self, J):
        J_count = [0] * self.n_features
        for index in J:
            J_count[index] += 1
        return J_count

    @staticmethod
    def add_lists(list1, list2):
        return list(map(add, list1, list2))

    def run(self, is_consistent):
        parameter_grid = list(itertools.product(self.lam_1_lst, self.lam_2_lst))
        J_count_dict = {key: [0] * self.n_features for key in parameter_grid}
        for i in tqdm(range(self.n_replicates)):
            X, y, w, = self.get_data(is_consistent=is_consistent)
            # print(w)
            for lam_1, lam_2 in tqdm(parameter_grid):
                argen = self.get_reg(lam_1=lam_1, lam_2=lam_2)
                reg = BootstrappedRegressor(regressor=argen, bootstrap_replicates=self.bootstrap_replicates)
                reg.fit(X, y)
                J_count = self.J_to_J_count(J=reg.J)
                J_count_dict[(lam_1, lam_2)] = self.add_lists(J_count_dict[(lam_1, lam_2)], J_count)
        J_prob_dict = {key: [item / self.n_replicates for item in value] for key, value in J_count_dict.items()}
        J_prob_array = np.array(list(J_prob_dict.values())).T
        plt.imshow(J_prob_array, cmap='gray')
        plt.xticks(range(len(parameter_grid)), np.round(parameter_grid, 2))
        plt.yticks(range(self.n_features), list(range(self.n_features)))
        plt.xlabel('parameters')
        plt.ylabel('variable index')
        plt.title(f'consistent: {is_consistent}, boostrap: {self.bootstrap_replicates}')
        plt.show()
        return J_prob_dict
