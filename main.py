# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:48:40 2018

@author: suzuki.k
"""

import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from methods import ward, pci_ward
import time

if __name__ == '__main__':
    n = 5
    d = 5
    stop = n
    repeat = 1000
    start = time.time()
    naive_list = []
    selective_list = []
    for i in range(repeat):
#        print("----------the %d-th----------" % i)
        np.random.seed(i)
        data = np.random.normal(0, 1, (n, d))
        sigma = np.identity(n)
        xi = np.identity(d)
        result, c_list_list, c_ab_list, a_list, b_list = ward(data)
        naive_p, selective_p = pci_ward(data, sigma, xi, stop, c_list_list, c_ab_list, a_list, b_list)
        del c_list_list, c_ab_list, a_list, b_list
        naive_list.append(naive_p)
        selective_list.append(selective_p)
    end = time.time()
    naive_array = np.array(naive_list)
    selective_array = np.array(selective_list)
    print("time[sec]: ", end - start)    