import os
import sys
import ast
import numpy as np
from xml.dom import minidom
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from csv import reader

timeloop_energy = []
salsa_energy = []

with open('./Alexnet_L2_timeloop_energy.csv', 'r') as read_obj:
    csv_reader = reader(read_obj, delimiter=';')
    for row in csv_reader:
        timeloop_energy = row
        timeloop_energy.pop()

with open('./Alexnet_L2_salsa_value_list.csv', 'r') as read_obj:
    csv_reader = reader(read_obj, delimiter=',')
    for row in csv_reader:
        salsa_energy = row

timeloop_energy = [float(x) for x in timeloop_energy]
salsa_energy = [float(x) for x in salsa_energy]

bins = np.arange(min(timeloop_energy), max(timeloop_energy), 5*10**7)

print(len(bins))

#plt.xlim([min(data)-5, max(data)+5])

(n_timeloop, bins, patches) = plt.hist(timeloop_energy, bins=bins, histtype="stepfilled", alpha=0.5)
(n_salsa, bins, patches) = plt.hist(salsa_energy, bins=bins, histtype="stepfilled", alpha=0.5)

#plt.xscale('log')
#plt.yscale('log')

plt.figure()

x_timeloop = np.arange(0, len(n_timeloop), 1)
x_salsa = np.arange(0, len(n_salsa), 1)

plt.plot(x_timeloop, n_timeloop, '-o')
plt.plot(x_salsa, n_salsa, '-o')

plt.show()