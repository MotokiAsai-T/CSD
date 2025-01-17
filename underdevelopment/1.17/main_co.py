import numpy as np
import random
import time
from fista_co import FISTA_co
from MCAmo import MCA
from calculate_MC_copy import compute_MC_and_grad_MC_return_intermediate
from calculate_MC_copy import compute_y_distrib
from calculate_MC_copy import compute_z_rs_1
from calculate_MC_copy import compute_z_rs_0
import matplotlib.pyplot as plt



    
    # Parameters
def main_co():
    seed = 42
    # Each r connected to each s node
    '''
    o_dict = {
        "1": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6},
        "8": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6},
        "13": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6},
        "14": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6},
        "15": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6}

    }
    r_dict = {
        "2": {"4": 1, "5":1,"10":1},
        "3": {"4": 1, "5":1,"10":1},
        "9": {"4": 1, "5":1,"10":1},
        "11" : {"4": 1, "5":1,"10":1}
    }
    s_dict = {
        "4": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 6,"7" : 6,"12" : 6},
        "5": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 6,"7" : 6,"12" : 6},
        "10": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 6,"7" : 6,"12" : 6} 
    }
    d_dict = {
        "6": {},
        "7" : {},
        "12": {}
    }
    '''
    origins = [f"O{i}" for i in range(1,6)]
    r_nodes = [f"R{i}" for i in range(1,11)]
    s_nodes = [f"S{i}" for i in range(1,11)]
    destinations = [f"D{i}" for i in range(1,6)]

    
    o_dict = {}
    or_cost_choices = [3,4,5]
    od_cost_choices = [6,7,8]
    for o in origins:
        links = {}
        for r in r_nodes:
            links[r] = random.choice(or_cost_choices)  
        for d in destinations:
            links[d] = random.choice(od_cost_choices)  
        o_dict[o] = links


    r_dict = {}
    rs_cost_choices = [3,4,5]
    for r in r_nodes:
        links = {}
        for s in s_nodes:
            links[s] = random.choice(rs_cost_choices)  # R->S
        r_dict[r] = links


    s_dict = {}
    sr_cost_choices = [3,4,5]
    sd_cost_choices = [7,8,9]
    for s in s_nodes:
        links = {}
        for r in r_nodes:
            links[r] = random.choice(sr_cost_choices)
        for d in destinations:
            links[d] = random.choice(sd_cost_choices)
        s_dict[s] = links

    d_dict = {d:{} for d in destinations}
    
    
    n_size = 4 # number of columns -1
    #example, n=2 -> n[0] (first s in the column), n[1] last n from o to d.
    r_size=len(r_dict)
    s_size=len(s_dict)
    o_size=len(o_dict)
    d_size=len(d_dict)
    K = 2
    theta_p = 5
    theta_c = 5 

    z_rs_bar = np.zeros((r_size,s_size))
    z_rs_bar += 20
    y_od_bar = np.zeros((o_size,d_size))
    y_od_bar += 5

    # Set c_rs and v 
    c_rs = np.zeros((r_size, s_size, K))
    shipper_do_cost_choices = [4,5,6]
    shipper_dont_cost_choices = [10,11,12]
    for rr in range(r_size):
        for ss in range(s_size):
            c_rs[rr, ss, 1] = random.choice(shipper_do_cost_choices) #doesn't have an impact on distribution, and final price.
            c_rs[rr, ss, 0] = random.choice(shipper_dont_cost_choices) # cost difference # has an impact higher difference, more incentives to do deliveries.

    #c_rs *= 10
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
            return MCA(self.o_dict, self.d_dict, self.r_dict, self.s_dict, 
                                          self.n_size, self.z_rs_bar, self.y_od_bar,  self.theta_p, self.theta_c,
                                          self.K, self.c_rs, v)
            
        
        def run_fr_sol(self, v):
            return compute_MC_and_grad_MC_return_intermediate(self.o_dict, self.r_dict, self.s_dict, self.d_dict,
                                                              self.n_size, self.K, self.theta_p, self.theta_c,
                                                              self.z_rs_bar, self.y_od_bar, self.c_rs, self.seed, v)

    subalgorithm = Subalgorithm(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed)

    v0 = np.zeros((r_size,s_size))
    v0 +=1

    start_time = time.time()
    # tighter epsilon, more iterations for better convergence
    fista_co = FISTA_co(subalgorithm, L0=1, eta=5, epsilon=1e-3, max_iter=2000)
    optimal_v = fista_co.optimize(v0)
   
    # MC_fin, _, C_rs, sum_exp, theta_c_val, P_od_rs,P_sd_rs, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = subalgorithm.run_fr_sol(optimal_v)

    # z_rs_1_values = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, optimal_v, theta_c_val)
    # z_rs_0_values = compute_z_rs_0(z_rs_bar, C_rs, sum_exp,theta_c_val)

    # y_od_rs, y_rs = compute_y_distrib(
    #     r_size=r_size,
    #     n_size=n_size,
    #     s_size=s_size,
    #     o_size=len(o_nodes),
    #     d_size=len(d_nodes),
    #     y_od_bar=y_od_bar*np.ones((len(o_nodes), len(d_nodes))),
    #     P_od_rs=P_od_rs,
    #     P_sd_rs=P_sd_rs
    # )
    end_time = time.time()
    print("Time:", np.round((end_time - start_time),3), "seconds")
    # print("MC*: ",MC_fin)
    # print("v*:", np.round(optimal_v,4))
    # print("y_rs:")
    # print(np.round(y_rs,4))
    # print("z_rs_1:")
    # print(np.round(z_rs_1_values,4))
    

    # plt.figure(figsize=(6, 3))
    # plt.plot(fista.MC_history, marker='o', linestyle='-')
    # plt.xlabel('Iterations')
    # plt.ylabel('MC Value')
    # plt.title('MC Value through Iterations')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return end_time - start_time

if __name__ == "__main__":
    main_co()