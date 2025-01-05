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
    #     "1": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6},
    #     "8": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6},
    #     "13": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6},
    #     "14": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6},
    #     "15": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 3,"7" : 3,"12" : 6}

    # }
    # r_dict = {
    #     "2": {"4": 1, "5":1,"10":1},
    #     "3": {"4": 1, "5":1,"10":1},
    #     "9": {"4": 1, "5":1,"10":1},
    #     "11" : {"4": 1, "5":1,"10":1}
    # }
    # s_dict = {
    #     "4": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 6,"7" : 6,"12" : 6},
    #     "5": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 6,"7" : 6,"12" : 6},
    #     "10": {"2": 1, "3": 1,"9" : 1,"11" : 1, "6": 6,"7" : 6,"12" : 6} 
    # }
    # d_dict = {
    #     "6": {},
    #     "7" : {},
    #     "12": {}
    # }

    
    # Setting num node
    o_num_node = 3
    r_num_node = 10
    s_num_node = 10
    d_num_node = 3
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
    rs_cost_choices = [3,4,5]
    for r in range(o_num_node+1, o_num_node + r_num_node+1):
        cost_dict = {}
        for s in range(o_num_node+r_num_node+1, all_num_node-d_num_node+1):
            cost_dict[str(s)] = random.choice(rs_cost_choices)
        r_dict[str(r)] = cost_dict


    s_dict = {}
    sr_cost_choices = [3,4,5]
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
    n_size= 5 #sum(len(v) for v in r_dict.values())  # number of deliveries to be made
    r_size= r_num_node #len(r_dict)
    s_size= s_num_node #len(s_dict)
    o_size= o_num_node #len(o_dict)
    d_size= d_num_node #len(d_dict)
    K = 2
    theta_p = 5
    theta_c = 5

    
    z_rs_bar = np.zeros((r_size,s_size))
    z_rs_bar += 500
    y_od_bar = np.zeros((o_size,d_size))
    y_od_bar += 2000

    # Set c_rs and v 
    c_rs = np.zeros((r_size, s_size, K))
    shipper_do_cost_choices = [4,5,6]
    shipper_dont_cost_choices = [10,11,12]
    for rr in range(r_size):
        for ss in range(s_size):
            c_rs[rr, ss, 1] = random.choice(shipper_do_cost_choices)
            c_rs[rr, ss, 0] = random.choice(shipper_dont_cost_choices) # cost difference

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

    # Start from zero to let algorithm find direction
    v0 = np.zeros((r_size,s_size))
    v0 += 1

    start_time = time.time()
    # tighter epsilon, more iterations for better convergence
    fista = FISTA(subalgorithm, L0=1.0, eta=1.1, epsilon=1e-1, max_iter=1000)
    optimal_v = fista.optimize(v0)
   
    MC_fin, _, C_rs, sum_exp, theta_c_val, P_od_rs, P_sd_rs, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes = subalgorithm.run_fr_sol(optimal_v)

    z_rs_1_values = compute_z_rs_1(z_rs_bar, C_rs, sum_exp, optimal_v, theta_c_val)
    z_rs_0_values = compute_z_rs_0(z_rs_bar, C_rs, sum_exp, optimal_v, theta_c_val)

    y_od_rs, y_rs = compute_y_distrib(
        r_size=r_size,
        n_size=n_size,
        s_size=s_size,
        o_size=len(o_nodes),
        d_size=len(d_nodes),
        y_od_bar=y_od_bar*np.ones((len(o_nodes), len(d_nodes))),
        P_od_rs=P_od_rs,
        P_sd_rs=P_sd_rs, 
        node_to_index=node_to_index,
        d_nodes=d_nodes,
        o_nodes=o_nodes,
        r_nodes=r_nodes,
        s_nodes=s_nodes
    )
    
    end_time = time.time()
    print("Time:", np.round((end_time - start_time),5), "seconds")
    
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
    main()