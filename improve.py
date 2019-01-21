# -*- coding: utf-8 -*-
"""
Created on Sun Jan 07 09:57:40 2019

@author: suzuki.k
"""

import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import kstest
import matplotlib.pyplot as plt
from methods import ward, pci_ward, pci_ward_improve, pci_ward_improve2, pci_ward_improve2_2
import time


if __name__ == '__main__':
    n = 20
    d = 2
    print("n:", n, end= ", ")
    print("d:", d)
    stop = n
    sigma = np.identity(n)
    xi = np.identity(d)
    naive_p_list = []
    selective_p_list = []
#    repeat = 10000
    start1 = time.time()
    np.random.seed(0)
    data = np.random.normal(0, 1, (n, d))
    start = time.time()
    output, c_list_list, c_ab_list, a_list, b_list, delta_list, tau_list = ward(data)
    end = time.time()
    print("ward time:", end - start)
#    start = time.time()
#    naive_p, selective_p = pci_ward_improve2_2(data, sigma, xi, 0, c_list_list, c_ab_list, a_list, b_list, delta_list, tau_list)
#    end = time.time()
#    print("first layer:", end - start)
    start = time.time()
    naive_p, selective_p = pci_ward_improve2_2(data, sigma, xi, n - 2, c_list_list, c_ab_list, a_list, b_list, delta_list, tau_list)
    end = time.time()
    print("last layer:", end - start)
#    naive_p, selective_p = pci_ward_improve(data, sigma, xi, stop, c_list_list, c_ab_list, a_list, b_list)
#    naive_p, selective_p = pci_ward_improve2(data, sigma, xi, stop, c_list_list, c_ab_list, a_list, b_list, delta_list, tau_list)

#    naive_p_list.append(naive_p)
#    selective_p_list.append(selective_p)
#    print("time[sec]:", end - start1)
#    selective_array = np.array(selective_p_list)
#    j = 1
#    for i in range(selective_array.shape[1]):
#        plt.hist(selective_array[:, i, j])
#        plt.show()
#        print(kstest(selective_array[:, i, j], "uniform"))

#import seaborn as sns
#sns.set()
##x = [5	, 10, 20, 50, 100, 200, 500]   
#x = [5	, 10, 20, 50, 100, 200]
#y1 = [2.18*10**(-3), 7.64*10**(-3), 26.9*10**(-3), 155*10**(-3), 601*10**(-3), 2.49]
#std1 = [43.2*10**(-6), 201*10**(-6), 361*10**(-6), 1.94*10**(-3), 4.81*10**(-3), 77.9*10**(-3)]
#y2 = [2.23*10**(-3), 5.29*10**(-3), 17.7*10**(-3), 104*10**(-3), 424*10**(-3), 1.68]
#std2 = [56.9*10**(-6), 105*10**(-6), 698*10**(-6), 1.88*10**(-3), 3.58*10**(-3), 14.4*10**(-3)]
#y3 = [4.32*10**(-3), 18.8*10**(-3), 124*10**(-3), 1.79, 15.1, 117]
#std3 = [54.3*10**(-6), 436*10**(-6), 2.75*10**(-3),	 36.3*10**(-3), 103*10**(-3), 2.44]
#
#plt.ylabel("time[sec]", fontsize=15)
#plt.xlabel("$n$", fontsize=15)
##plt.plot(x, y1, label="ward")
##plt.plot(x, y2, label="first")
##plt.plot(x, y3, label="last")
#plt.errorbar(x,y1,yerr=std1, label="ward",fmt='-o')
#plt.errorbar(x,y2,yerr=std2, label="first", fmt='-o')
#plt.errorbar(x,y3,yerr=std3, label="last", fmt='-o')
#plt.legend(fontsize=15)
#plt.tick_params(labelsize=15)
#plt.savefig("time_wfl.pdf")
#plt.show()
#x = [2, 10, 50, 100, 200, 500, 1000]
#y1 = [7.62*10**(-3), 7.94*10**(-3), 7.78*10**(-3), 8.13*10**(-3), 8.27*10**(-3), 8.86*10**(-3), 10.1*10**(-3)]   
#std1 = [201*10**(-6), 590*10**(-6), 197*10**(-6), 521*10**(-6), 258*10**(-6), 47*10**(-6), 238*10**(-6)]
#y2 = [5.21*10**(-3), 8.49*10**(-3), 25.7*10**(-3), 48.6*10**(-3), 98.3*10**(-3), 397*10**(-3), 1.47]
#std2 = [80.5*10**(-6), 299*10**(-6), 639*10**(-6), 796*10**(-6), 5.1*10**(-3), 9.49*10**(-3), 44.7*10**(-3)]
#y3 = [18.8*10**(-3), 25*10**(-3), 54.4*10**(-3), 98.1*10**(-3), 179*10**(-3), 1.08, 4.58]
#std3 = [321*10**(-6), 212*10**(-6), 3.04*10**(-3), 8.75*10**(-3), 5.69*10**(-3), 19.4*10**(-3), 74.8*10**(-3)]
#plt.ylabel("time[sec]", fontsize=15)
#plt.xlabel("$d$", fontsize=15)
##plt.plot(x, y1, label="before")
##plt.plot(x, y2, label="after")
#plt.errorbar(x,y1,yerr=std1, label="ward",fmt='-o')
#plt.errorbar(x,y2,yerr=std2, label="first", fmt='-o')
#plt.errorbar(x,y3,yerr=std3, label="last", fmt='-o')
#plt.legend(fontsize=15)
#plt.tick_params(labelsize=15)
#plt.savefig("time_d.pdf")
#plt.show()
