import numpy as np
import math
import copy
import time
import random

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


# def calc_mu_sd_P_sd_rs(n_size, d_size, mu_sd, Z_sd, P_sd_rs, node_to_index, t, s_nodes, r_nodes, d_nodes, theta_p, v):
#     for n in range(1, n_size):
#         mu_sd_before = mu_sd[n-1] # (s_size, d_size)
#         mu_sd_after = mu_sd[n] # (s_size, d_size)
#         Z_sd_after = Z_sd[n] # (s_size, d_size)
#         P_sd_rs_after = P_sd_rs[n] # (r_size, s_size, s_size, d_size)
 
#         for s_idx, s_node in enumerate(s_nodes):
#             s_index = node_to_index[s_node]

#             exp_direct_sd = np.exp( -theta_p * t[s_index, :d_size] )

#             for d_idx in range(d_size):
#                 sum_val, val_matrix = calc_exp_sd(r_nodes, s_nodes, s_idx, d_idx, s_node, t, theta_p, node_to_index, mu_sd_before, v)

#                 Z_val = exp_direct_sd[d_idx] + sum_val
#                 Z_sd_after[s_idx][d_idx] = Z_val
#                 mu_sd_after[s_idx][d_idx] = (-1/theta_p) * math.log(Z_val)

#                 calc_P_sd_rs(P_sd_rs_after, node_to_index, r_nodes, s_nodes, t, s_idx, d_idx, val_matrix, Z_val)


def calc_mu_sd_P_sd_rs(n_size, d_size, mu_sd, Z_sd, P_sd_rs, node_to_index, t, s_nodes, r_nodes, d_nodes, theta_p, v):
    r_indices = [node_to_index[r_node] for r_node in r_nodes]
    s_indices = [node_to_index[s_node] for s_node in s_nodes]
    for n in range(1, n_size):
        mu_sd_before = mu_sd[n-1] # (s_size, d_size)
        mu_sd_after = mu_sd[n] # (s_size, d_size)
        Z_sd_after = Z_sd[n] # (s_size, d_size)
        P_sd_rs_after = P_sd_rs[n] # (r_size, s_size, s_size, d_size)
 
        for s_idx, s_node in enumerate(s_nodes):
            s_index = node_to_index[s_node]

            t_sr = t[s_index,r_indices]
            t_rs = t[np.ix_(r_indices, s_indices)]
            mask = (t_rs > 0)

            exp_direct_sd = np.exp(-theta_p * t[s_index, :d_size])

            for d_idx in range(d_size):
                val_mat = np.exp(-theta_p * (t_sr[:, None] + t_rs - v + mu_sd_before[np.newaxis, :, d_idx]))

                sum_val = (val_mat[mask]).sum()
                Z_val = exp_direct_sd[d_idx] + sum_val
                Z_sd_after[s_idx, d_idx] = Z_val
                mu_sd_after[s_idx, d_idx] = (-1/theta_p)*math.log(Z_val)
                inv_Z = 1.0/Z_val

                P_sd_rs_after[:, :, s_idx, d_idx] = (val_mat * mask) * inv_Z


def generate_t(r_nodes, s_nodes, s_index, s_node, t, node_to_index):
    t_sr = np.zeros(len(r_nodes))
    t_rs = np.zeros((len(r_nodes), len(s_nodes)))
    for r_idx2, r_node2 in enumerate(r_nodes):
        t_sr[r_idx2] = t[s_index, node_to_order[r_node2]]
        for s_idx2, s_node2 in enumerate(s_nodes):
            t_sr[s_idx2][r_idx2] = t[node_to_index[s_node], node_to_index[r_node2]]
            t_rs[r_idx2][s_idx2] = t[node_to_index[r_node2], node_to_index[s_node2]]
    return t_sr, t_rs

                

# def calc_exp_sd(r_nodes, s_nodes, s_idx, d_idx, s_node, t, theta_p, node_to_index, mu_sd_before, v):
#     val_matrix = np.zeros((len(r_nodes), len(s_nodes)))
#     for r_idx2, r_node2 in enumerate(r_nodes):
#         for s_idx2, s_node2 in enumerate(s_nodes):
#             t_sr = t[node_to_index[s_node], node_to_index[r_node2]]
#             t_rs = t[node_to_index[r_node2], node_to_index[s_node2]]
#             mask = (t_rs > 0) #  boolean array, size  r*s. true only when there is a connection.
#             val = v[r_idx2, s_idx2]
#             val_matrix[r_idx2][s_idx2] = np.exp( -theta_p * ( t_sr + t_rs - val + mu_sd_before[s_idx][d_idx] ))
#     sum_val = (val_matrix[mask]).sum()
#     return sum_val, val_matrix


# def calc_P_sd_rs(P_sd_rs_after, node_to_index, r_nodes, s_nodes, t, s_idx, d_idx, val_matrix, Z_val):
    
#     for r_idx2, r_node2 in enumerate(r_nodes):
#         for s_idx2, s_node2 in enumerate(s_nodes):
#             t_rs = t[node_to_index[r_node2], node_to_index[s_node2]]
#             mask = (t_rs > 0)
#             P_sd_rs_after[r_idx2][s_idx2][s_idx][d_idx] = (val_matrix[r_idx2][s_idx2] * mask) / Z_val


def calc_d_mu_sd(n_size, P_sd_rs, d_mu_sd, s_size, d_size):
    for n in range(1, n_size):
        P_sd_rs_n = P_sd_rs[n] # (r_size, s_size, s_size, d_size)
        d_mu_sd_before = d_mu_sd[n-1] # (r_size, s_size, s_size, d_size)
        d_mu_sd_after = d_mu_sd[n] # (r_size, s_size, s_size, d_size)

        for s_idx in range(s_size):
            for d_idx in range(d_size):

                P_slice = P_sd_rs_n[:, :, s_idx, d_idx] # (r_size, s_size)

                Multiplied_sum = calc_Multiplied_sum(P_slice, d_mu_sd_before, d_idx)

                d_mu_sd_after[:, :, s_idx, d_idx] = -P_slice + Multiplied_sum


def calc_Multiplied_sum(P_slice, d_mu_sd_before, d_idx):
    
    # Multiplied_sum = np.zeros((r_size, s_size))
    # for r_idx2 in range(r_size):
    #     for s_idx2 in range(s_size):
    #         Multiplied_sum[r_idx2, s_idx2] = P_slice[r_idx2, s_idx2] * d_mu_sd_before[r_idx2, s_idx2, s_idx, d_idx]

    P_sum_r = P_slice.sum(axis=0)

    Multiplied_sum = np.einsum('x, rsx -> rs', P_sum_r, d_mu_sd_before[:, :, :, d_idx] )
    
    return Multiplied_sum


def calc_value_functions(rng, r_size, s_size, K, c_rs, v, theta_c):
    
    epsilon = rng.gumbel(0, theta_c, size=(r_size,s_size,K))*0.01
    C_rs = c_rs + epsilon
    sum_exp = np.zeros((r_size, s_size))

    sum_0 = np.exp(-theta_c * C_rs[:, :, 0])
    sum_1 = np.exp(-theta_c * (C_rs[:, :, 1] + v))

    sum_exp = sum_0 + sum_1

    V_rs = (-1/theta_c) * np.log(sum_exp)

    return C_rs, sum_exp, V_rs


def calc_mu_od_P_od_rs(n_size, o_nodes, r_nodes, s_nodes, d_size, node_to_index, t, mu_sd, mu_od, Z_od, P_od_rs, theta_p, v):
    mu_sd_last = mu_sd[n_size-2] # (s_size, d_size)

    for o_idx, o_node in enumerate(o_nodes):
        o_index = node_to_index[o_node]

        exp_direct_od = np.exp( -theta_p * t[o_index, :d_size] )

        for d_idx in range(d_size):
            sum_val_od, val_matrix_od = calc_exp_od(r_nodes, s_nodes, o_node, d_idx, t, node_to_index, v, theta_p, mu_sd_last)

            Z_val_od = exp_direct_od[d_idx] + sum_val_od
            Z_od[o_idx, d_idx] = Z_val_od
            mu_od[o_idx, d_idx] = (-1/theta_p) * np.log(Z_val_od)

            calc_P_od_rs(r_nodes, s_nodes, t, node_to_index, P_od_rs, o_idx, d_idx, val_matrix_od, Z_val_od)


def calc_exp_od(r_nodes, s_nodes, o_node, d_idx, t, node_to_index, v, theta_p, mu_sd_last):
    val_matrix_od = np.zeros((len(r_nodes), len(s_nodes)))
    for r_idx, r_node in enumerate(r_nodes):
        for s_idx, s_node in enumerate(s_nodes):
            t_or = t[node_to_index[o_node], node_to_index[r_node]]
            t_rs = t[node_to_index[r_node], node_to_index[s_node]]
            mask = (t_rs > 0) 
            val = v[r_idx, s_idx]
            val_matrix_od[r_idx][s_idx] = np.exp( -theta_p * ( t_or + t_rs - val + mu_sd_last[s_idx][d_idx] ))
    sum_val_od = (val_matrix_od[mask]).sum()

    return sum_val_od, val_matrix_od


def calc_P_od_rs(r_nodes, s_nodes, t, node_to_index, P_od_rs, o_idx, d_idx, val_matrix_od, Z_val_od):

    for r_idx, r_node in enumerate(r_nodes):
        for s_idx, s_node in enumerate(s_nodes):
            t_rs = t[node_to_index[r_node], node_to_index[s_node]]
            mask = (t_rs > 0)
            P_od_rs[r_idx, s_idx, o_idx, d_idx] = (val_matrix_od[r_idx, s_idx] * mask) / Z_val_od


def calc_d_mu_od(n_size, d_mu_sd, P_od_rs, d_mu_od):
    d_mu_sd_last = d_mu_sd[n_size-2] # (r_size, s_size, s_size, d_size)

    P_sum_r_in = P_od_rs.sum(axis=0) # (s_size, o_size, d_size)

    sum = np.einsum('r s a d, a o d -> r s o d', d_mu_sd_last, P_sum_r_in)

    d_mu_od[:] = -P_od_rs + sum


### アルゴリズム

def MCA(o_dict, d_dict, r_dict, s_dict, n_size, z_rs_bar, y_od_bar, theta_p, theta_c, K, c_rs, v):

    rng = np.random.default_rng(42)
    # 全ノードのインデックスリストと全ネットワークのコスト行列の生成
    t, node_to_index, d_nodes, o_nodes, r_nodes, s_nodes, r_node_index, s_node_index = get_node_order_and_t(o_dict, r_dict, s_dict, d_dict)

    # ノードサイズの取得
    o_size = len(o_nodes)
    r_size = len(r_nodes)
    s_size = len(s_nodes)
    d_size = len(d_nodes)


    ## SDループ
    # データセット
    Z_sd = np.zeros((n_size, s_size, d_size))
    mu_sd = np.zeros((n_size, s_size, d_size))
    P_sd_rs = np.zeros((n_size, r_size, s_size, s_size, d_size))
    d_mu_sd = np.zeros((n_size, r_size, s_size, s_size, d_size))

    # 初期値生成
    for s_idx, s_node in enumerate(s_nodes):
        mu_sd[0, s_idx, :] = t[node_to_index[s_node], :d_size]

    # mu_sd と P_sd_rs の計算
    calc_mu_sd_P_sd_rs(n_size, d_size, mu_sd, Z_sd, P_sd_rs, node_to_index, t, s_nodes, r_nodes, d_nodes, theta_p, v)
    
    # d_mu_sd の計算
    s_time = time.time()
    calc_d_mu_sd(n_size, P_sd_rs, d_mu_sd, s_size, d_size)
    e_time = time.time()
    # print("time:", e_time - s_time)
    
    
    # C_rs, sum_exp, V_rs の計算
    C_rs, sum_exp, V_rs = calc_value_functions(rng, r_size, s_size, K, c_rs, v, theta_c)
    

    # z_rs_1 の計算
    z_rs_1 = z_rs_bar * (np.exp(-theta_c*(C_rs[:,:,1]+v))) / sum_exp
    

    ## ODループ
    # データセット
    Z_od = np.zeros((o_size, d_size))
    mu_od = np.zeros((o_size, d_size))
    P_od_rs = np.zeros((r_size, s_size, o_size, d_size))
    d_mu_od = np.zeros((r_size, s_size, o_size, d_size))

    # mu_od, P_od_rs の計算
    calc_mu_od_P_od_rs(n_size, o_nodes, r_nodes, s_nodes, d_size, node_to_index, t, mu_sd, mu_od, Z_od, P_od_rs, theta_p, v)
    
    # d_mu_od の計算
    calc_d_mu_od(n_size, d_mu_sd, P_od_rs, d_mu_od)
    # print("d_mu_od:", d_mu_od)

    # MC の計算
    MC = np.sum(y_od_bar*mu_od) + np.sum(z_rs_bar*V_rs)
    # print("MC:", MC)

    # grad_MC の計算
    grad_MC = np.zeros((r_size, s_size))
    sum_d_mu = np.sum(y_od_bar*d_mu_od, axis=(2,3))
    grad_MC[:,:] = sum_d_mu + z_rs_1
    # print("grad_MC:", grad_MC)

    return MC, grad_MC

      