import numpy as np
import random
import time
from fista import FISTA
from calculate_MC import compute_MC_and_grad_MC
from calculate_MC import compute_MC_and_grad_MC_return_intermediate
from calculate_MC import compute_y_distrib
from calculate_MC import compute_z_rs_1
from calculate_MC import compute_z_rs_0
import matplotlib.pyplot as plt

def main():

    seed = 42  # set the seed for reproductability
    # Define network
    
    # o_dict = {
    #     "1": {"2": 1, "3": 1,"9" : 1, "6": 5,"7" : 5},
    #     "8": {"2": 1, "3": 1,"9" : 1, "6": 5,"7" : 5}
    # }

    # r_dict = {
    #     "2": {"4": 1},
    #     "3": {"5": 1},
    #     "9": {"10": 1}
    # }

    # s_dict = {
    #     "4": {"2": 1, "3": 3,"9" : 1, "6": 6,"7" : 6},
    #     "5": {"2": 2, "3": 1,"9" : 1, "6": 6,"7" : 6},
    #     "10": {"2": 2, "3": 1,"9" : 1, "6": 6,"7" : 6}
    # }

    # d_dict = {
    #     "6": {},
    #     "7" : {}
    # }

    
    # Setting num node
    o_num_node = 20
    r_num_node = 25
    s_num_node = 25
    d_num_node = 20
    all_num_node = o_num_node + r_num_node + s_num_node + d_num_node

    # Generate cost dict

    o_dict = {}
    or_cost_choices = [3,4,5]
    od_cost_choices = [4,5,6]
    for o in range(1, o_num_node+1):
        # 各起点ノードに対応するコスト辞書を作成
        cost_dict = {}
        # リンクo-rのコスト
        for r in range(o_num_node+1, o_num_node + r_num_node+1):
            cost_dict[str(r)] = random.choice(or_cost_choices)
        # リンクo-dのコスト
        for d in range(all_num_node - d_num_node+1, all_num_node+1):
            cost_dict[str(d)] = random.choice(od_cost_choices)

        # 辞書に追加
        o_dict[str(o)] = cost_dict

    r_dict = {}
    rs_cost_choices = [4,5,6]
    for r in range(o_num_node+1, o_num_node + r_num_node+1):
        r_dict[str(r)] = {str(r_num_node+r): random.choice(rs_cost_choices)}

    s_dict = {}
    sr_cost_choices = [2,3,4]
    sd_cost_choices = [4,5,6]
    for s in range(o_num_node+r_num_node+1, all_num_node-d_num_node+1):
        cost_dict = {}
        # リンクs-rのコスト
        for r in range(o_num_node+1, o_num_node + r_num_node+1):
            cost_dict[str(r)] = random.choice(sr_cost_choices)
        # リンクs-dのコスト
        for d in range(all_num_node - d_num_node+1, all_num_node+1):
            cost_dict[str(d)] = random.choice(sd_cost_choices)

        s_dict[str(s)] = cost_dict

    d_dict = {}
    for d in range(all_num_node - d_num_node+1, all_num_node+1):
        d_dict[str(d)] = {}

    
    # Parameters
    n_size = 5  # number of deliveries to be made
    rs_size = r_num_node  # number of delivery spots
    K = 2
    theta_p = 1   
    theta_c = 1

    z_rs_bar = np.zeros(rs_size)
    for i in range(rs_size):
        z_rs_bar[i] = 1
    

    y_od_bar = 2  # Drivers per OD pair
    

    c_rs = np.zeros((rs_size, K))
    for r_s in range(rs_size):
        c_rs[r_s, 1] = 5
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
            return compute_MC_and_grad_MC(self.o_dict, self.r_dict, self.s_dict, self.d_dict,
                                          self.n_size, self.K, self.theta_p, self.theta_c,
                                          self.z_rs_bar, self.y_od_bar, self.c_rs, self.seed, v)
        
        def run_fr_sol(self, v):
            return compute_MC_and_grad_MC_return_intermediate(self.o_dict, self.r_dict, self.s_dict, self.d_dict,
                                                              self.n_size, self.K, self.theta_p, self.theta_c,
                                                              self.z_rs_bar, self.y_od_bar, self.c_rs, self.seed, v)

    subalgorithm = Subalgorithm(o_dict, r_dict, s_dict, d_dict, n_size, K, theta_p, theta_c, z_rs_bar, y_od_bar, c_rs, seed)

    v0 = np.zeros(rs_size)
    v0 +=5
    start_time = time.time()
    fista = FISTA(subalgorithm, L0=1.0, eta=1.1, epsilon=1e-5, max_iter=1000)
    optimal_v = fista.optimize(v0)
   

    _, _, C_rs, sum_exp, theta_c_val, P_od_rs, P_od_no_rs, P_sd_rs, node_to_index, r_to_s_mapping, d_nodes, o_nodes, r_nodes = subalgorithm.run_fr_sol(optimal_v)
    z_rs_1_values = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, optimal_v, theta_c_val)
    z_rs_0_values = compute_z_rs_0(z_rs_bar, C_rs, sum_exp, optimal_v, theta_c_val)

    y_od_rs, y_rs = compute_y_distrib(
        r_size=len(r_nodes),
        n_size=n_size,
        s_size=len(r_to_s_mapping), 
        o_size=len(o_nodes),
        d_size=len(d_nodes),
        y_od_bar=y_od_bar*np.ones((len(o_nodes), len(d_nodes))),
        P_od_rs=P_od_rs,
        P_od_no_rs=P_od_no_rs,
        P_sd_rs=P_sd_rs,
        node_to_index=node_to_index,
        r_to_s_mapping=r_to_s_mapping,
        d_nodes=d_nodes,
        o_nodes=o_nodes,
        r_nodes=r_nodes
    )
    end_time = time.time()
    print("v*:", np.round(optimal_v,4))
    print("Time:", np.round(end_time - start_time,3), "seconds")
    #print("y_od_rs distribution:")
    #print(y_od_rs)
    print("y_rs:")
    print(np.round(y_rs,4))
    #print("z_rs_0:")
    #print(z_rs_0_values)
    print("z_rs_1:")
    print(np.round(z_rs_1_values,4))

    plt.figure(figsize=(6, 3))
    plt.plot(fista.MC_history, marker='o', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('MC Value')
    plt.title('MC Value through Iterations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

