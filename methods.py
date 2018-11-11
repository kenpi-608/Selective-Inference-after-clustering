# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:26:04 2018

@author: suzuki.k
"""

import numpy as np
import copy
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


class Cluster:
    """
    クラスタを定義するクラス
    """
    def __init__(self, element, num, index, size, table, centroid):
        self.element = element  # クラスタ内の要素
        self.num = num  # どのデータが含まているか(リスト)
        self.index = index  # クラスタの番号
        self.size = size  # クラスタの大きさ
        self.table = table  # どのデータが含まれているか(SIの計算に使用, numpy.ndarray)
        self.centroid = centroid  # クラスタの重心

    def merge(self, c, index):
        """
        クラスタを統合する
        """
        new_num = np.append(self.num, c.num)
        new_element = np.vstack((self.element, c.element))
        new_centroid = np.vstack((self.element, c.element)).mean(axis=0)
        new_cluster = Cluster(new_element, new_num, index, self.size + c.size, self.table + c.table, new_centroid)
        return new_cluster


def make_data(n, d, mulist, nlist, qlist, Xi, Sigma):
    """
    人工データ生成
    
    引数
    ----------
    arg1 : n
        サンプルサイズ
    arg2 : d
        特徴数
    arg3, 4, 5 : mulist, nlist, qlist
        ex) mulist = [1, 2], nlist = [[0, 10], [10, 15]], qlist = [[0, 5],[5, 10]]]
            n = 0 ~ 10, d = 0 ~ 5には平均1を, n = 10 ~ 15, d = 5 ~ 10には平均2を載せる
    arg6, 7 : Xi, Sigma
        Xi : 特徴間の分散共分散行列
        Sigma : 標本間の分散共分散行列
    返り値
    -------
    ndarray型
    """
    M = np.zeros((n, d))
    for i in range(len(nlist)):
        M[nlist[i][0]:nlist[i][1], qlist[i][0]:qlist[i][1]] += mulist[i]
    
    V = np.random.normal(0, 1, (n, d))
    # cholesky decomposition
    Ls = np.linalg.cholesky(Sigma)
    Lx = np.linalg.cholesky(Xi)
    X = M + np.dot(np.dot(Ls.T, V), Lx)
    return X   


def intersection(forL, forU, interval):
    """
    区間の共通部分を計算する
    Created on Thu Oct 19 11:59:46 2017
    @author: Inoue.S
    """
    intersect = []
    lower = forL
    using = interval
    while 1:
        if len(using) == 0:
            break
        low = using[:, 0]
        up = using[:, 1]
        base = using[np.argsort(low)][0]
        intersect.append([lower, base[0]])
        inner_fg = (low <= base[1])
        while 1:
            lower = np.max(up[inner_fg])
            if np.array_equal(inner_fg, (low <= lower)):
                break
            else:
                inner_fg = (low <= lower)
        outer_fg = (inner_fg == False)
        using = using[outer_fg]
    intersect.append([lower, forU])
    return intersect


def calc_multisec_p(tau, sig, interval):
    """複数の区間がある場合のSelective-p値を計算する"""
    denominator = []
    numerator = []
    for j in range(len(tau)):
        t_j = tau[j] / sig[j]
        interval_j = interval[j] / sig[j]
        l_j = interval_j[:, 0]
        u_j = interval_j[:, 1]
        denominator.append(np.sum(stats.norm.cdf(-l_j) - stats.norm.cdf(-u_j)))
        # どの区間に統計量が属するか
        p = np.argmax((l_j <= t_j) * (t_j <= u_j))
        if p + 1 < len(l_j):
            upper_section = np.sum(stats.norm.cdf(-l_j[p + 1:]) - stats.norm.cdf(-u_j[p + 1:]))
        else:
            upper_section = 0
        numerator.append(stats.norm.cdf(-t_j) - stats.norm.cdf(-u_j[p]) + upper_section)
    denominator = np.array(denominator)
    numerator = np.array(numerator)
    selective_p = numerator / denominator
    assert len(np.where(selective_p < 0)[0]) <= 0, "selective_p < 0"
    return numerator / denominator

def kstest(d, result_p, step):
    """
    ks-検定を行う
    return 一様分布であることを棄却されなかった数
    """
    cnt = 0
    for i in range(d):
#         if stats.kstest(result_p[:, step, i], "uniform")[1] > 0.05:
#             cnt += 1
        print(stats.kstest(result_p[:, step, i], "uniform"))
    return cnt


def pv_dendrogram(sp, nap, dim, *args, **kwargs):
    """
    一つの次元におけるselective-p, naive-p値を付与した樹形図を表示する
    引数
    ----------
    arg1 : sp
        selective-p: np.array型
    arg2 : nap
        naive-p: np.array型
    arg3 : dim
        どの次元におけるp-valueを表示するか
    arg4, 5 : *args, **kwargs
        scipy.cluster.hierarchy.dendrogramの*args, **kwargs
        
    返り値
    -------
    ndarray型
        scipy.cluster.hierarchy.dendrogramの出力
    """    
    
    ddata = dendrogram(*args, **kwargs)

#    plt.xlabel('sample index', fontsize=20)
#    plt.ylabel('distance',fontsize = 15)
    
    xarray = np.array(ddata['icoord'])
    yarray = np.array(ddata['dcoord'])
    
    xarray = xarray[np.argsort(yarray, axis=0)[:, 2]]
    yarray = yarray[np.argsort(yarray, axis=0)[:, 2]]
    for sp, nap, i, d, c in zip(sp, nap, xarray, yarray, ddata['color_list']):
        
        xm = 0.5 * (i[1] + i[2])

        xdiv = abs(i[1] + i[2])
        # x1, x2の幅は文字の大きさや図の大きさによって適切に変更する必要がある
        x1 = xm - 20 * (xm/xdiv)
    
        x2 = xm + 20 * (xm/xdiv)
                
        y1 = d[1]
        
        y2 = d[1]
        
        plt.annotate("%.3f" % sp[dim], (x1, y1), color="fuchsia", xytext=(0, -5), textcoords='offset points', va='top', ha='center', label="s", fontsize=30)
        plt.annotate("%.3f" % nap[dim], (x2, y2), color="darkcyan", xytext=(0, -5), textcoords='offset points', va='top', ha='center', label="n", fontsize=30)
    return ddata


def ward(data):
    n = data.shape[0]
#    d = data.shape[1]
    # クラスタ初期化
    c_list = []
    index = 0
    table = np.zeros(n)
    for i1 in range(n):
        num = i1
        table[i1] = 1
        c_list.append(Cluster(data[i1], num, index, 1, table, data[i1]))
        index = index + 1
        table = np.zeros(n)
     # 距離行列を計算する
    dmat = np.inf * np.ones((n, n))
    for i2 in range(n - 1):
        for j1 in range(i2 + 1, n):
            c_ij = c_list[i2].merge(c_list[j1], index)
            e_cij = np.sum((c_ij.element - c_ij.centroid) ** 2)
            e_ci = np.sum((c_list[i2].element - c_list[i2].centroid) ** 2)
            e_cj = np.sum((c_list[j1].element - c_list[j1].centroid) ** 2)
            dmat[i2, j1] = e_cij - (e_ci + e_cj)
    # ward法
    output = []
    c_list_list = []
    c_ab_list = []
    a_list = []
    b_list = []
    for t in range(n - 1):
#            print("--------------------%d step --------------------" % t)
        # 距離行列の最小値とインデックスを取得する
        [minvalue, [a, b]] = [dmat.min(), list(np.argwhere(dmat == dmat.min())[0])]
        c_id = [c_list[a].index, c_list[b].index]
        # クラスタを統合する
        c_ab = c_list[a].merge(c_list[b], index)
        c_list.append(c_ab)
        index = index + 1
        # 新しくできたクラスタと残りのクラスタの距離を計算
        new_dis = []
        for k in range(len(c_list) - 1):
            if k != a and k != b:
                e_ck = np.sum((c_list[k].element - c_list[k].centroid) ** 2)
                e_cab = np.sum((c_ab.element - c_ab.centroid) ** 2)
                c_abk = c_ab.merge(c_list[k], 0)
                e_cabk = np.sum((c_abk.element - c_abk.centroid) ** 2)
                new_dis.append(e_cabk - (e_cab + e_ck))
        new_dis = np.array(new_dis)
        
        c_list_list.append(copy.deepcopy(c_list))
        c_ab_list.append(copy.deepcopy(c_ab))
        a_list.append(a)
        b_list.append(b)
        
         # 統合されたクラスタを削除, 距離行列からも該当箇所を削除
        del (c_list[a])
        del (c_list[b - 1])
        # 行
        dmat = np.delete(dmat, [a, b], 0)
        # 列
        dmat = np.delete(dmat, [a, b], 1)
        # 新しい距離を挿入
        dmat = np.hstack((dmat, np.c_[new_dis]))
        dmat = np.vstack((dmat, np.inf * np.ones(dmat.shape[1])))
        # 結果のフォーマットを整える
        result = [c_id[0], c_id[1], minvalue, c_list[len(c_list) - 1].size]
        output.append(result)
    return output, c_list_list, c_ab_list, a_list, b_list


def pci_ward(data, sigma, xi, stop, c_list_list, c_ab_list, a_list, b_list, tol=10**(-12)):
    """
    Selective Inference at each step of hierarchical clustering
    each feature
    """
    n = data.shape[0]
    d = data.shape[1]        
    naive_p_step = []
    selective_p_step = []
    for t in range(n - 1):
        c_list = copy.deepcopy(c_list_list[t])
        c_ab = copy.deepcopy(c_ab_list[t])
        a = a_list[t]
        b = b_list[t]
        if t < stop:
#            print("--------------------%d step --------------------" % t)
            """Selective Inferenceの計算"""
            tau = np.abs(c_list[a].centroid - c_list[b].centroid)
            s = np.sign(c_list[a].centroid - c_list[b].centroid)
            delta = c_list[a].table / c_list[a].size - c_list[b].table / c_list[b].size
            sigma_tilde2 = np.diag(xi) * np.dot(np.dot(delta, sigma), delta)
            # 切断点の計算に必要なものの準備
            vbv_list = []
            cbc_list = []
            vbc_list = []
            e_cab = np.sum((c_ab.element - c_ab.centroid) ** 2)
            e_ca = np.sum((c_list[a].element - c_list[a].centroid) ** 2)
            e_cb = np.sum((c_list[b].element - c_list[b].centroid) ** 2)
            abab = np.dot(np.dot(c_ab.table, sigma), delta) ** 2 / c_ab.size
            aa = np.dot(np.dot(c_list[a].table, sigma), delta) ** 2 / c_list[a].size
            bb = np.dot(np.dot(c_list[b].table, sigma), delta) ** 2 / c_list[b].size
            abx = np.dot(np.dot(c_ab.table, sigma), delta) * c_ab.centroid
            ax = np.dot(np.dot(c_list[a].table, sigma), delta) * c_list[a].centroid
            bx = np.dot(np.dot(c_list[b].table, sigma), delta) * c_list[b].centroid
            for k1 in range(n - t - 1):
                for k2 in range(k1 + 1, n - t):
                    if (k1, k2) != (a, b):
                        c_k12 = c_list[k1].merge(c_list[k2], 0)            
                        e_ck12 = np.sum((c_k12.element - c_k12.centroid) ** 2)
                        e_ck1 = np.sum((c_list[k1].element - c_list[k1].centroid) ** 2)
                        e_ck2 = np.sum((c_list[k2].element - c_list[k2].centroid) ** 2)
                        # vbvは必ず0以下となる
                        vbv = e_cab - e_ck12 - e_ca + e_ck1 - e_cb + e_ck2
                        assert vbv <= 0, "vbv > 0 occured"
                        vbv_list.append(vbv)
                        scalar = (- abab + np.dot(np.dot(c_k12.table, sigma), delta) ** 2 / c_k12.size
                                  + aa   - np.dot(np.dot(c_list[k1].table, sigma), delta) ** 2 / c_list[k1].size
                                  + bb   - np.dot(np.dot(c_list[k2].table, sigma), delta) ** 2 / c_list[k2].size)
                        cbc = (s**2 * scalar * np.diag(np.dot(xi.T, xi))) / sigma_tilde2 ** 2
                        cbc_list.append(cbc)
                        d_vec = (- abx + np.dot(np.dot(c_k12.table, sigma), delta) * c_k12.centroid
                                 + ax  - np.dot(np.dot(c_list[k1].table, sigma), delta) * c_list[k1].centroid
                                 + bx  - np.dot(np.dot(c_list[k2].table, sigma), delta) * c_list[k2].centroid)
                        vbc = (s * np.dot(d_vec, xi)) / sigma_tilde2
                        vbc_list.append(vbc)
            # {}_(n- t + 1) C_2個
            vbv_array = np.array(vbv_list)
            # {}_(n- t + 1) C_2個のd次元ベクトル shape : {}_(n - t + 1) C_2, d
            cbc_array = np.array(cbc_list)
            # {}_(n- t + 1) C_2個のd次元ベクトル shape : {}_(n - t + 1) C_2, d
            vbc_array = np.array(vbc_list)
#            # L, Uがともに0となることを防ぐために閾値をもうける
#            cbc_array[np.abs(cbc_array) < tol] = 0
#            vbc_array[np.abs(vbc_array) < tol] = 0
            
            # このステップまでのSelection Eventでの計算
            for i3 in range(t + 1):
                old_c_list = copy.deepcopy(c_list_list[i3])
                old_c_ab = copy.deepcopy(c_ab_list[i3])
                p_vbv_list = []
                p_cbc_list = []
                p_vbc_list = []
                e_cab = np.sum((old_c_ab.element - old_c_ab.centroid) ** 2)
                e_ca = np.sum((old_c_list[a_list[i3]].element - old_c_list[a_list[i3]].centroid) ** 2)
                e_cb = np.sum((old_c_list[b_list[i3]].element - old_c_list[b_list[i3]].centroid) ** 2)
                abab = np.dot(np.dot(old_c_ab.table, sigma), delta) ** 2 / old_c_ab.size
                aa = np.dot(np.dot(old_c_list[a_list[i3]].table, sigma), delta) ** 2 / old_c_list[a_list[i3]].size
                bb = np.dot(np.dot(old_c_list[b_list[i3]].table, sigma), delta) ** 2 / old_c_list[b_list[i3]].size
                abx = np.dot(np.dot(old_c_ab.table, sigma), delta) * old_c_ab.centroid
                ax = np.dot(np.dot(old_c_list[a_list[i3]].table, sigma), delta) * old_c_list[a_list[i3]].centroid
                bx = np.dot(np.dot(old_c_list[b_list[i3]].table, sigma), delta) * old_c_list[b_list[i3]].centroid
                for k1 in range(len(old_c_list) - 2):
                    for k2 in range(k1 + 1, len(old_c_list) - 1):
                        if k1 != k2:
                            c_k12 = old_c_list[k1].merge(old_c_list[k2], 0)            
                            e_ck12 = np.sum((c_k12.element - c_k12.centroid) ** 2)
                            e_ck1 = np.sum((old_c_list[k1].element - old_c_list[k1].centroid) ** 2)
                            e_ck2 = np.sum((old_c_list[k2].element - old_c_list[k2].centroid) ** 2)
                            # vbvは必ず0以下となる
                            p_vbv = e_cab - e_ck12 - e_ca + e_ck1 - e_cb + e_ck2
                            assert p_vbv <= 0, "p_vbv > 0 occured"
                            p_vbv_list.append(p_vbv)
                            scalar = (- abab + np.dot(np.dot(c_k12.table, sigma), delta) ** 2 / c_k12.size
                                      + aa   - np.dot(np.dot(old_c_list[k1].table, sigma), delta) ** 2 / old_c_list[k1].size
                                      + bb   - np.dot(np.dot(old_c_list[k2].table, sigma), delta) ** 2 / old_c_list[k2].size)
                            p_cbc = (s**2 * scalar * np.diag(np.dot(xi.T, xi))) / sigma_tilde2 ** 2
                            p_cbc_list.append(p_cbc)
                            d_vec = (- abx + np.dot(np.dot(c_k12.table, sigma), delta) * c_k12.centroid
                                     + ax  - np.dot(np.dot(old_c_list[k1].table, sigma), delta) * old_c_list[k1].centroid
                                     + bx  - np.dot(np.dot(old_c_list[k2].table, sigma), delta) * old_c_list[k2].centroid)
                            p_vbc = (s * np.dot(d_vec, xi)) / sigma_tilde2
                            p_vbc_list.append(p_vbc)
                p_vbv_array = np.array(p_vbv_list)
                p_vbc_array = np.array(p_vbc_list)
                p_cbc_array = np.array(p_cbc_list)
                if vbv_array.shape[0] == 0:
                    vbv_array = p_vbv_array
                    vbc_array = p_vbc_array
                    cbc_array = p_cbc_array
                else:
                    vbv_array = np.r_[vbv_array, p_vbv_array]
#                    print("vbc_array:", vbc_array.shape)
#                    print("p_vbc_array:", p_vbc_array.shape)
                    vbc_array = np.r_[vbc_array, p_vbc_array]
                    cbc_array = np.r_[cbc_array, p_cbc_array]
#            print("vbv_array:", len(vbv_array))
                
            
            # 切断点計算
            forL_list = []
            forU_list = []
            interval = []
            for j in range(d):
#               # 判別式
                """
                切断点計算
                """
                discriminant = vbc_array[:, j]**2 - cbc_array[:, j] * vbv_array
                cond = cbc_array[:, j] == 0
                cond_l = (cbc_array[:, j] == 0) & (vbc_array[:, j] < 0)
                cond_u = (cbc_array[:, j] == 0) & (vbc_array[:, j] > 0)
                if cbc_array[cond].shape[0] > 0:
                    if vbc_array[:, j][cond_l].shape[0] > 0:
                        forL = np.max(-vbv_array[cond_l] / (2 * vbc_array[:, j][cond_l]))
                        forL_list.append(forL)
                    elif vbc_array[:, j][cond_u].shape[0] > 0:
                        forU = np.min(-vbv_array[cond_u] / (2 * vbc_array[:, j][cond_u]))
                        forU_list.append(forU)
                cond2 = cbc_array[:, j] > 0
                if cbc_array[:, j][cond2].shape[0] > 0:
                    forL = np.max((-vbc_array[:, j][cond2] - np.sqrt(discriminant[cond2])) / cbc_array[:, j][cond2])
                    forL_list.append(forL)
                    forU = np.min((-vbc_array[:, j][cond2] + np.sqrt(discriminant[cond2])) / cbc_array[:, j][cond2])
                    forU_list.append(forU)
                    
                """
                分岐focus
                """
                if len(forL_list) > 0:
                    forL = max(forL_list)
                else:
                    forL = -np.inf
                if len(forU_list) > 0:
                    forU = min(forU_list)
                else:
                    forU = np.inf
                forL = max(forL, -tau[j])     
                forL_list = []
                forU_list = []
                cond3 = (cbc_array[:, j] < 0) & (discriminant > 0)    
                if cbc_array[:, j][cond3].shape[0] > 0:
                    x_small = (-vbc_array[:, j][cond3] + np.sqrt(discriminant[cond3])) / cbc_array[:, j][cond3]
                    x_large = (-vbc_array[:, j][cond3] - np.sqrt(discriminant[cond3])) / cbc_array[:, j][cond3]
                    # x_small < forL < x_large
                    flag1 = (x_small < forL) & (forL < x_large)
                    while np.sum(flag1) > 0:
                        if x_large[flag1].shape[0] > 0: 
                            x_large_max = np.max(x_large[flag1])
                            forL = x_large_max
                        flag1 = (x_small < forL) & (forL < x_large)
                    # x_small < forU < x_large
                    flag2 = (x_small < forU) & (forU < x_large)
                    while np.sum(flag2) > 0:
                        if x_small[flag2].shape[0] > 0:
                            x_small_min = np.min(x_small[flag2])
                            forU = x_small_min
                        flag2 = (x_small < forU) & (forU < x_large)
                    # forL < x_small, x_large < forU
                    flag3 = (forL < x_small) & (x_large < forU)
                    if np.sum(flag3) > 0:
                        interval_j = np.array(intersection(forL, forU, np.c_[x_small, x_large][flag3]))
                        interval.append(interval_j + tau[j])
                    else:
                        interval.append(np.array([[forL, forU]]) + tau[j])                
                else:
                    L = forL + tau[j]
                    if L < 0:
                        L = 0
                    U = forU + tau[j]
                    interval.append(np.array([[L, U]]))
                    
            interval = np.array(interval)
            # 検定
            sig = np.sqrt(sigma_tilde2)
#             print(sig)
            sub1 = stats.norm.cdf(tau / sig)
            sub2 = stats.norm.cdf(-tau / sig)
            naive_p = 2 * np.min(np.c_[sub1, sub2], axis=1)
            selective_p = calc_multisec_p(tau, sig, interval)  
            naive_p_step.append(naive_p)
            selective_p_step.append(selective_p)

    return naive_p_step, selective_p_step



