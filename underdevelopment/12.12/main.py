import numpy as np
from fista import FISTA
from calculate_MC import compute_MC_and_grad_MC
import matplotlib.pyplot as plt

def main():

    seed = 42  # set the seed for reproductability
    # Define network
    o_dict = {
        "1": {"2": 1, "3": 1,"9" : 1, "6": 5,"7" : 5},
        "8": {"2": 1, "3": 1,"9" : 1, "6": 5,"7" : 5}
    }

    r_dict = {
        "2": {"4": 1},
        "3": {"5": 1},
        "9": {"10": 1}
    }

    s_dict = {
        "4": {"2": 1, "3": 3,"9" : 1, "6": 6,"7" : 6},
        "5": {"2": 2, "3": 1,"9" : 1, "6": 6,"7" : 6},
        "10": {"2": 2, "3": 1,"9" : 1, "6": 6,"7" : 6}
    }

    d_dict = {
        "6": {},
        "7" : {}
    }
    # Parameters
    n_size = 3  # number of deliveries to be made
    rs_size = 3  # number of delivery spots
    K = 2
    theta_p = 1   
    theta_c = 1

    z_rs_bar = np.zeros(rs_size)
    z_rs_bar[0] = 1
    z_rs_bar[1] = 1  # Total deliveries to match number of deliveries to be made
    z_rs_bar[2] = 1

    y_od_bar = 2  # Drivers per OD pair
    

    c_rs = np.zeros((rs_size, K))
    for r_s in range(rs_size):
        c_rs[r_s, 1] = 10
        c_rs[r_s, 0] = c_rs[r_s, 1] + 2

 
    class Subalgorithm:
        def __init__(self, o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed):
            self.o_dict = o_dict
            self.r_dict = r_dict
            self.s_dict = s_dict
            self.d_dict = d_dict
            self.n_size = n_size
            self.K = K
            self.theta_p = theta_p
            self.theta_c = theta_c
            self.z_rs_bar = z_rs_bar
            self.y_od_bar = y_od_bar
            self.c_rs = c_rs
            self.seed = seed

        def run(self, v):
            # Pass v as an argument!
            return compute_MC_and_grad_MC(
                self.o_dict, self.r_dict, self.s_dict, self.d_dict,
                self.n_size, self.K, self.theta_p, self.theta_c,
                self.z_rs_bar, self.y_od_bar, self.c_rs, self.seed, v
            )


    subalgorithm = Subalgorithm(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs,seed)

    # Initial guess for v
    v0 = np.zeros(rs_size)
    v0 +=15
    #no matter the inital value, it converges to the same solution, this is nice !
    # but the solution is negative.

    # FISTA optimization
    fista = FISTA(subalgorithm, L0=1.0, eta=1.1, epsilon=1e-2, max_iter=100)
    optimal_v = fista.optimize(v0)


    print("Optimal v:", optimal_v)

    plt.figure(figsize=(6, 3))
    plt.plot(fista.MC_history, marker='o', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('MC Value')
    plt.title('MC Value through Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()