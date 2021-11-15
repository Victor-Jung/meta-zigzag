import os
import sys
import ast
import csv
import numpy as np
from xml.dom import minidom
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

barchart500_data_path = "../barchart_data/500_iterations/"
barchart1000_data_path = "../barchart_data/1000_iterations/"
barchart3000_data_path = "../barchart_data/3000_iterations/"

filname1 = "ResNet34_eyeriss_tl_L"
filname_en = "_en_prob.csv"
filname_lat = "_lat_prob.csv"

number_of_layer = 8
layer_idx_list = [1,2,9,10,18,19,31,32]

en_go = [681682329.8, 648525476.9, 329125093.7, 645732580.1, 344624170.0, 682989536.3, 455943427.8, 908757453.8]
lat_go = [1347584, 774144, 516096, 1032192, 516096, 1032192, 566272, 1132544]

# Extract data 

it500_list_en = []
it1000_list_en = []
it3000_list_en = []

it500_list_lat = []
it1000_list_lat = []
it3000_list_lat = []

for layer_idx in layer_idx_list:
    with open(barchart500_data_path + filname1 + str(layer_idx) + filname_en, "r") as f:
        for row in csv.reader(f):
            it500_list_en.append([float(x) for x in row])
    f.close()
    with open(barchart500_data_path + filname1 + str(layer_idx) + filname_lat, "r") as f:
        for row in csv.reader(f):
            it500_list_lat.append([float(x) for x in row])
    f.close()

# Divide by the Global Optimum to get a percentage

for i in range(number_of_layer):
    for j in range(len(it500_list_en[i])):
        it500_list_en[i][j] = en_go[i]/it500_list_en[i][j]
    for j in range(len(it500_list_lat[i])):
        it500_list_lat[i][j] = lat_go[i]/it500_list_lat[i][j]

# Plotting

fig, ax = plt.subplots(1, 2)

ax[0].set_title('Energy')
ax[0].boxplot(it500_list_en, boxprops=dict(color='#426A8C'))

ax[1].set_title('Latency')
ax[1].boxplot(it500_list_lat)

plt.show()