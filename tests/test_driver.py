from bootstrapped_argen.driver import Driver
import math


def test_driver():
    lam_1_lst = [math.exp(-i) for i in range(5)]
    driver = Driver(n_features=4,
                    nonzero_index=2,
                    n_samples=10,
                    lam_1_lst=lam_1_lst,
                    lam_2_lst=[0.5],
                    n_replicates=15,
                    bootstrap_replicates=10)
    ret_dict = driver.run(is_consistent=False)
    print(ret_dict)



def test_driver_full_dimension():
    lam_1_lst = [math.exp(-i) for i in range(16)]
    driver = Driver(n_features=16,
                    nonzero_index=8,
                    n_samples=1000,
                    lam_1_lst=lam_1_lst,
                    lam_2_lst=[0.5],
                    n_replicates=256,
                    bootstrap_replicates=128)
    ret_dict = driver.run(is_consistent=False)
    print(ret_dict)