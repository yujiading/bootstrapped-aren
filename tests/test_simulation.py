from bootstrapped_aren.library.simulation import Simulation


def test_simulation():
    sim = Simulation(n_features=6, nonzero_index=3, n_samples=10)
    X, y, w, eps, is_consistent = sim.run
    print(X, y, w, eps, is_consistent)
