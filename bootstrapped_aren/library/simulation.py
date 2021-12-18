import numpy as np


class Simulation:
    def __init__(self, n_features: int, nonzero_index: int, n_samples: int):
        """
        nonzero_index: relevent variables with indices less than r
        """
        self.n_features = n_features
        self.nonzero_index = nonzero_index
        self.n_samples = n_samples

    @staticmethod
    def _is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    @property
    def _covariance_matrix(self):
        G = np.random.randn(self.n_features, self.n_features)
        Q = G.dot(G.T)
        scale_vector = np.diag(Q) ** (-1 / 2)
        scale_matrix = np.diag(scale_vector)
        cov = np.matmul(np.matmul(scale_matrix, Q), scale_matrix)
        try:
            self._is_pos_def(cov)
        except ValueError:
            print('covariance matrix needs to be positive semi definite')
        return cov, Q

    def samples(self, cov):
        mean = np.zeros(self.n_features)
        X = np.random.multivariate_normal(mean=mean, cov=cov, size=self.n_samples)
        return X

    @property
    def weights(self):
        w = np.random.randn(self.nonzero_index)
        w = w / np.linalg.norm(w)

        scaling = np.random.uniform(1 / 3, 1, self.nonzero_index)
        w = np.sign(w) * (np.abs(w) + scaling)
        # np.all(np.abs(w) >= 1/3)
        w = np.concatenate((w, np.zeros(self.n_features - self.nonzero_index)))
        return w

    def noise(self, X, w):
        sigma = 0.1 * np.linalg.norm(X.dot(w))
        eps = np.random.randn(self.n_samples) * sigma
        return eps

    @staticmethod
    def response(X, w, eps):
        return X.dot(w) + eps

    def is_consistent(self, Q, w):
        Q_JJ = Q[:self.nonzero_index, :self.nonzero_index]
        Q_JcJ = Q[self.nonzero_index:, :self.nonzero_index]
        w_J = w[:self.nonzero_index]
        lst = np.matmul(Q_JcJ, np.linalg.inv(Q_JJ))
        lst = np.matmul(lst, np.sign(w_J))
        lst_norm = np.linalg.norm(lst, ord=np.inf)
        if lst_norm <= 1:
            return True
        else:
            return False

    @property
    def run(self):
        cov, Q = self._covariance_matrix
        X = self.samples(cov=cov)
        w = self.weights
        eps = self.noise(X=X, w=w)
        y = self.response(X=X, w=w, eps=eps)
        is_consistent = self.is_consistent(Q=Q, w=w)
        return X, y, w, eps, is_consistent
