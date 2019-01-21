# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 11:48:40 2018

@author: suzuki.k
"""

import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from methods import ward, pci_ward, pci_ward_improve2
import time

if __name__ == '__main__':
    n = 5
    d = 3
    stop = n
    repeat = 1
    naive_list = []
    selective_list = []
    start = time.time()
    for i in range(repeat):
#        print("----------the %d-th----------" % i)
        np.random.seed(i)
        data = np.random.normal(0, 1, (n, d))
#        data = np.random.randint(0, 10, (n, d))
        sigma = np.identity(n)
        xi = np.identity(d)
        result, c_list_list, c_ab_list, a_list, b_list = ward(data)
        start = time.time()
        pci_ward_improve2(data, sigma, xi, stop, c_list_list, c_ab_list, a_list, b_list)
        end = time.time()
        print("time[sec]: ", end - start)
        del c_list_list, c_ab_list, a_list, b_list
#        naive_list.append(naive_p)
#        selective_list.append(selective_p)
#    naive_array = np.array(naive_list)
#    selective_array = np.array(selective_list)
