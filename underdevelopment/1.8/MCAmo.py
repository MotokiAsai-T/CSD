import numpy as np
import math
import copy
import time


# 関数

def get_node_order_and_t(o_dict, r_dict, s_dict, d_dict):

    d_nodes = list(d_dict.keys())
    o_nodes = list(o_dict.keys())
    r_nodes = list(r_dict.keys())
    s_nodes = list(s_dict.keys())

    node_order = d_nodes + o_nodes + r_nodes + s_nodes
    node_to_index = {node: i for i, node in enumerate(node_order)}

    # value_order = r_nodes + s_nodes
    # value_to_index = {node: i for i, node in enumerate(value_order)}

    # val = np.zeros((len(r_nodes), len(s_nodes)))
    # for r_node, links in r_dict.items():
    #     for dest_node, value in links.items():
    #         val[value_to_index[r_node], value_to_index[dest_node]] = v

    r_node_index = {node: i for i, node in enumerate(r_nodes)}
    s_node_index = {node: i for i, node in enumerate(s_nodes)}
    

    t_size = len(node_order)
    t = np.zeros((t_size, t_size))

    for o_node, links in o_dict.items():
        for dest_node, cost in links.items():
            t[node_to_index[o_node], node_to_index[dest_node]] = cost

    for r_node, links in r_dict.items():
        for dest_node, cost in links.items():
            t[node_to_index[r_node], node_to_index[dest_node]] = cost

    for s_node, links in s_dict.items():
        for dest_node, cost in links.items():
            t[node_to_index[s_node], node_to_index[dest_node]] = cost

    return t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, r_node_index, s_node_index


def calc_exp_sd(r_nodes, s_nodes, s_idx, d_idx, s_node, t, theta_p, node_to_index, r_node_index, s_node_index, mu_sd_before, v):
    val_matrix = np.zeros((len(r_nodes), len(s_nodes)))
    for r_idx2, r_node2 in enumerate(r_nodes):
        for s_idx2, s_node2 in enumerate(s_nodes):
            t_sr = t[node_to_index[s_node], node_to_index[r_node2]]
            t_rs = t[node_to_index[r_node2], node_to_index[s_node2]]
            mask = (t_rs > 0) #  boolean array, size  r*s. true only when there is a connection.
            val = v[r_idx2, s_idx2]
            val_matrix[r_idx2][s_idx2] = np.exp( -theta_p * ( t_sr + t_rs - val + mu_sd_before[s_idx][d_idx] ))
    sum_val = (val_matrix[mask]).sum()
    return sum_val, val_matrix


def calc_P_sd_rs(P_sd_rs_after, node_to_index, r_nodes, s_nodes, t, s_idx, d_idx, val_matrix, Z_val):
    for r_idx2, r_node2 in enumerate(r_nodes):
        for s_idx2, s_node2 in enumerate(s_nodes):
            t_rs = t[node_to_index[r_node2], node_to_index[s_node2]]
            mask = (t_rs > 0)
            P_sd_rs_after[r_idx2][s_idx2][s_idx][d_idx] = (val_matrix[r_idx2][s_idx2] * mask) / Z_val




### アルゴリズム

def MCA(o_dict, d_dict, r_dict, s_dict, n_size, theta_p, v):

    # 全ノードのインデックスリストと全ネットワークのコスト行列の生成
    t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, r_node_index, s_node_index = get_node_order_and_t(o_dict, r_dict, s_dict, d_dict)

    # ノードサイズの取得
    o_size = len(o_nodes)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)

    # データセット
    Z_sd = np.zeros((n_size, s_size, d_size))
    mu_sd = np.zeros((n_size, s_size, d_size))
    P_sd_rs = np.zeros((n_size, r_size, s_size, s_size, d_size))
    d_mu_sd = np.zeros((n_size, r_size, s_size, s_size, d_size))

    # 初期値生成
    for s_idx, s_node in enumerate(s_nodes):
        mu_sd[0, s_idx, :] = t[node_to_index[s_node], :d_size]

    # アルゴリズム
    
    # mu_sd と P_sd_rs の計算
    for n in range(1, n_size):
        mu_sd_before = mu_sd[n-1] # (s_size, d_size)
        mu_sd_after = mu_sd[n] # (s_size, d_size)
        Z_sd_after = Z_sd[n] # (s_size, d_size)
        P_sd_rs_after = P_sd_rs[n] # (r_size, s_size, s_size, d_size)
 
        for s_idx, s_node in enumerate(s_nodes):
            s_index = node_to_index[s_node]

            exp_direct_sd = np.exp( -theta_p * t[s_index, :d_size] )

            for d_idx in range(d_size):
                sum_val, val_matrix = calc_exp_sd(r_nodes, s_nodes, s_idx, d_idx, s_node, t, theta_p, node_to_index, r_node_index, s_node_index, mu_sd_before, v)

                Z_val = exp_direct_sd[d_idx] + sum_val
                Z_sd_after[s_idx][d_idx] = Z_val
                mu_sd_after[s_idx][d_idx] = (-1/theta_p) * math.log(Z_val)

                calc_P_sd_rs(P_sd_rs_after, node_to_index, r_nodes, s_nodes, t, s_idx, d_idx, val_matrix, Z_val)

    # d_mu_sd の計算
    for n in range(1, n_size):
        P_sd_rs_n = P_sd_rs[n] # (r_size, s_size, s_size, d_size)
        d_mu_sd_before = d_mu_sd[n-1] # (r_size, s_size, s_size, d_size)
        d_mu_sd_after = d_mu_sd[n] # (r_size, s_size, s_size, d_size)

        for s_idx in range(s_size):
            for d_idx in range(d_size):

                

