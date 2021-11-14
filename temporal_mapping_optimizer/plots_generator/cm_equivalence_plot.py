import os
import sys
import ast
import numpy as np
from xml.dom import minidom
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

tl_to_zz_data_path = "../../tl_to_zz_benchmark/resnet34_zigzag_output/"
tl_to_tl_data_path = "../../tl_to_zz_benchmark/resnet34_timeloop_output/"
zz_to_zz_data_path = "../../zz_to_tl_benchmark/"
zz_to_tl_data_path = "../../zz_to_tl_benchmark/"

zz_file_name1 = "ResNet34_eyeriss_tl_L"
zz_file_name2 = "ResNet34_zz_to_tl_L"
zz_file_name3 = "_M1_SU1_min_en.xml"

timeloop_file_name1 = "resnet34_"
extension1 = ".stats.txt"
extension2 = "_zz.stats.txt"

tl_to_zz_data_dict_list = []
zz_to_zz_data_dict_list = []

tl_to_tl_energy_list = []
tl_to_tl_memory_energy_list = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
zz_to_tl_energy_list = []
zz_to_tl_memory_energy_list = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

number_of_layer = 8
layer_idx_list = [1,2,9,10,18,19,31,32]

### Extract data from Timeloop and ZigZag file ###

for i in range(1,number_of_layer + 1):
    tl_to_zz_output_doc = minidom.parse(tl_to_zz_data_path + zz_file_name1 + str(layer_idx_list[i-1]) + zz_file_name3)
    tl_to_zz_data_dict_list.append(tl_to_zz_output_doc)

    zz_to_zz_output_doc = minidom.parse(zz_to_zz_data_path + zz_file_name2 + str(layer_idx_list[i-1]) + zz_file_name3)
    zz_to_zz_data_dict_list.append(zz_to_zz_output_doc)

    tl_to_tl_file = open(tl_to_tl_data_path + timeloop_file_name1 + str(layer_idx_list[i-1]) + extension1, 'r')
    tl_to_tl_lines = tl_to_tl_file.readlines()

    zz_to_tl_file = open(zz_to_tl_data_path + timeloop_file_name1 + str(layer_idx_list[i-1]) + extension2, 'r')
    zz_to_tl_lines = zz_to_tl_file.readlines()

    for line in tl_to_tl_lines:
        if line[0:21] == "Total topology energy":
            tl_to_tl_energy_list.append(float(line[23:-3]))

        if line[0:8] == "MACCs = ":
            mac_cost = float(line[8:])
            print(mac_cost)
        if line[4:15] == "psum_spad  ":
            tl_to_tl_memory_energy_list[i-1][2] = float(line[-5:])*mac_cost
        if line[4:18] == "weights_spad  ":
            tl_to_tl_memory_energy_list[i-1][0] = float(line[-5:])*mac_cost
        if line[4:16] == "ifmap_spad  ":
            tl_to_tl_memory_energy_list[i-1][1] = float(line[-5:])*mac_cost
        if line[4:16] == "shared_glb  ":
            tl_to_tl_memory_energy_list[i-1][3] = float(line[-5:])*mac_cost
        if line[4:10] == "DRAM  ":
            tl_to_tl_memory_energy_list[i-1][4] = float(line[-5:])*mac_cost
    
    for line in zz_to_tl_lines:
        if line[0:21] == "Total topology energy":
            zz_to_tl_energy_list.append(float(line[23:-3]))

        if line[0:8] == "MACCs = ":
            mac_cost = float(line[8:])
            print(mac_cost)
        if line[4:15] == "psum_spad  ":
            zz_to_tl_memory_energy_list[i-1][2] = float(line[-5:])*mac_cost
        if line[4:18] == "weights_spad  ":
            zz_to_tl_memory_energy_list[i-1][0] = float(line[-5:])*mac_cost
        if line[4:16] == "ifmap_spad  ":
            zz_to_tl_memory_energy_list[i-1][1] = float(line[-5:])*mac_cost
        if line[4:16] == "shared_glb  ":
            zz_to_tl_memory_energy_list[i-1][3] = float(line[-5:])*mac_cost
        if line[4:10] == "DRAM  ":
            zz_to_tl_memory_energy_list[i-1][4] = float(line[-5:])*mac_cost

### Post-Processing of ZigZag Data ###

tl_to_zz_energy_list = []
tl_to_zz_memory_energy_list = []
tl_to_zz_memory_energy_list2 = [[],[],[],[],[],[],[],[]]
tl_to_zz_time_list = []

zz_to_zz_energy_list = []
zz_to_zz_memory_energy_list = []
zz_to_zz_memory_energy_list2 = [[],[],[],[],[],[],[],[]]
zz_to_zz_time_list = []

for i in range(number_of_layer):
    tl_to_zz_energy_list.append(float(tl_to_zz_data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    tl_to_zz_memory_energy_list.append([ast.literal_eval(tl_to_zz_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(tl_to_zz_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(tl_to_zz_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])
    tl_to_zz_time_list.append(float(tl_to_zz_data_dict_list[i].getElementsByTagName('elapsed_time_second')[0].nextSibling.data))

for i in range(number_of_layer):
    zz_to_zz_energy_list.append(float(zz_to_zz_data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    zz_to_zz_memory_energy_list.append([ast.literal_eval(zz_to_zz_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(zz_to_zz_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(zz_to_zz_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])
    zz_to_zz_time_list.append(float(zz_to_zz_data_dict_list[i].getElementsByTagName('elapsed_time_second')[0].nextSibling.data))

for i in range(number_of_layer):
    for j in range(3):
        tl_to_zz_memory_energy_list2[i].append(tl_to_zz_memory_energy_list[i][j][0]) # add spad
    tl_to_zz_memory_energy_list2[i].append(tl_to_zz_memory_energy_list[i][1][1] + tl_to_zz_memory_energy_list[i][2][1]) # add shared buffer
    tl_to_zz_memory_energy_list2[i].append(tl_to_zz_memory_energy_list[i][0][1] + tl_to_zz_memory_energy_list[i][1][2] + tl_to_zz_memory_energy_list[i][2][2]) # add DRAM

for i in range(number_of_layer):
    for j in range(3):
        zz_to_zz_memory_energy_list2[i].append(zz_to_zz_memory_energy_list[i][j][0]) # add spad
    zz_to_zz_memory_energy_list2[i].append(zz_to_zz_memory_energy_list[i][1][1] + zz_to_zz_memory_energy_list[i][2][1]) # add shared buffer
    zz_to_zz_memory_energy_list2[i].append(zz_to_zz_memory_energy_list[i][0][1] + zz_to_zz_memory_energy_list[i][1][2] + zz_to_zz_memory_energy_list[i][2][2]) # add DRAM

# Extract the sum of Memory access energy for all layers per memory type

tl_to_zz_w_cost = []
tl_to_zz_i_cost = []
tl_to_zz_o_cost = []
tl_to_zz_buff_cost = []
tl_to_zz_dram_cost = []

tl_to_tl_w_cost = []
tl_to_tl_i_cost = []
tl_to_tl_o_cost = []
tl_to_tl_buff_cost = []
tl_to_tl_dram_cost = []

zz_to_zz_w_cost = []
zz_to_zz_i_cost = []
zz_to_zz_o_cost = []
zz_to_zz_buff_cost = []
zz_to_zz_dram_cost = []

zz_to_tl_w_cost = []
zz_to_tl_i_cost = []
zz_to_tl_o_cost = []
zz_to_tl_buff_cost = []
zz_to_tl_dram_cost = []

for i in range(number_of_layer):
    
    tl_to_zz_w_cost.append(tl_to_zz_memory_energy_list2[i][0])
    tl_to_zz_i_cost.append(tl_to_zz_memory_energy_list2[i][1])
    tl_to_zz_o_cost.append(tl_to_zz_memory_energy_list2[i][2])
    tl_to_zz_buff_cost.append(tl_to_zz_memory_energy_list2[i][3])
    tl_to_zz_dram_cost.append(tl_to_zz_memory_energy_list2[i][4])

    tl_to_tl_w_cost.append(tl_to_tl_memory_energy_list[i][0])
    tl_to_tl_i_cost.append(tl_to_tl_memory_energy_list[i][1])
    tl_to_tl_o_cost.append(tl_to_tl_memory_energy_list[i][2])
    tl_to_tl_buff_cost.append(tl_to_tl_memory_energy_list[i][3])
    tl_to_tl_dram_cost.append(tl_to_tl_memory_energy_list[i][4])

    zz_to_zz_w_cost.append(zz_to_zz_memory_energy_list2[i][0])
    zz_to_zz_i_cost.append(zz_to_zz_memory_energy_list2[i][1])
    zz_to_zz_o_cost.append(zz_to_zz_memory_energy_list2[i][2])
    zz_to_zz_buff_cost.append(zz_to_zz_memory_energy_list2[i][3])
    zz_to_zz_dram_cost.append(zz_to_zz_memory_energy_list2[i][4])
    
    zz_to_tl_w_cost.append(zz_to_tl_memory_energy_list[i][0])
    zz_to_tl_i_cost.append(zz_to_tl_memory_energy_list[i][1])
    zz_to_tl_o_cost.append(zz_to_tl_memory_energy_list[i][2])
    zz_to_tl_buff_cost.append(zz_to_tl_memory_energy_list[i][3])
    zz_to_tl_dram_cost.append(zz_to_tl_memory_energy_list[i][4])


### Plotting ###

fig, ax = plt.subplots(2)
labels = ['Layer 1', 'Layer 2', 'Layer 9', 'Layer 10', "Layer 18", "Layer 19", "Layer 31", "Layer 32"]

w = 0.4
X = np.arange(1,number_of_layer+1)
plt.suptitle("     Cost Model Equivalence", fontsize=20, y=0.95)
fig.tight_layout()
fig.subplots_adjust(hspace=0.4)

ax[0].set_title('Timeloop Mapping', fontsize=17, pad=12)
ax[1].set_title('X-XXXX Mapping', fontsize=17, pad=12)

ax[0].bar(X-(0.2), [sum(x) for x in zip(tl_to_tl_w_cost, tl_to_tl_i_cost, tl_to_tl_o_cost, tl_to_tl_buff_cost, tl_to_tl_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.2), [sum(x) for x in zip(tl_to_tl_w_cost, tl_to_tl_i_cost, tl_to_tl_o_cost, tl_to_tl_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.2), [sum(x) for x in zip(tl_to_tl_w_cost, tl_to_tl_i_cost, tl_to_tl_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.2), [sum(x) for x in zip(tl_to_tl_w_cost, tl_to_tl_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.2), tl_to_tl_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

ax[0].bar(X+(0.2), [sum(x) for x in zip(zz_to_tl_w_cost, zz_to_tl_i_cost, zz_to_tl_o_cost, zz_to_tl_buff_cost, zz_to_tl_dram_cost)], label='DRAM', width = w, color="#e76f51", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.2), [sum(x) for x in zip(zz_to_tl_w_cost, zz_to_tl_i_cost, zz_to_tl_o_cost, zz_to_tl_buff_cost)], label='shared buff', width = w, color="#f4a261", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.2), [sum(x) for x in zip(zz_to_tl_w_cost, zz_to_tl_i_cost, tl_to_tl_o_cost)], label='O-reg', width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.2), [sum(x) for x in zip(zz_to_tl_w_cost, zz_to_tl_i_cost)], label='I-reg', width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.2), zz_to_tl_w_cost, label='W-reg', width = w, color="#264653", linewidth=1, edgecolor='grey')

ax[1].bar(X-(0.2), [sum(x) for x in zip(tl_to_zz_w_cost, tl_to_zz_i_cost, tl_to_zz_o_cost, tl_to_zz_buff_cost, tl_to_zz_dram_cost)], label='DRAM', width = w, color="#e76f51", linewidth=1, edgecolor='grey')
ax[1].bar(X-(0.2), [sum(x) for x in zip(tl_to_zz_w_cost, tl_to_zz_i_cost, tl_to_zz_o_cost, tl_to_zz_buff_cost)], label='shared buff', width = w, color="#f4a261", linewidth=1, edgecolor='grey')
ax[1].bar(X-(0.2), [sum(x) for x in zip(tl_to_zz_w_cost, tl_to_zz_i_cost, tl_to_zz_o_cost)], label='O-reg', width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
ax[1].bar(X-(0.2), [sum(x) for x in zip(tl_to_zz_w_cost, tl_to_zz_i_cost)], label='I-reg', width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
ax[1].bar(X-(0.2), tl_to_zz_w_cost, label='W-reg', width = w, color="#264653", linewidth=1, edgecolor='grey')

ax[1].bar(X+(0.2), [sum(x) for x in zip(zz_to_zz_w_cost, zz_to_zz_i_cost, zz_to_zz_o_cost, zz_to_zz_buff_cost, zz_to_zz_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
ax[1].bar(X+(0.2), [sum(x) for x in zip(zz_to_zz_w_cost, zz_to_zz_i_cost, zz_to_zz_o_cost, zz_to_zz_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
ax[1].bar(X+(0.2), [sum(x) for x in zip(zz_to_zz_w_cost, zz_to_zz_i_cost, zz_to_zz_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
ax[1].bar(X+(0.2), [sum(x) for x in zip(zz_to_zz_w_cost, zz_to_zz_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
ax[1].bar(X+(0.2), zz_to_zz_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

plt.xticks(fontsize=13)
ax[0].set_xticks(np.arange(1, len(labels)+1))
ax[0].set_xticklabels(labels, fontsize=10)
ax[0].tick_params(pad=28)

ax[0].set_ylabel("Energy (pJ)", fontsize=15, labelpad=10)
ax[0].legend(loc='upper left', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)
ax[0].set_facecolor('#F2F2F2')

ax[0].tick_params(axis='y', which='major', pad=5)

ax[0].yaxis.set_major_locator(MultipleLocator(0.5e9))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.1e9))
ax[0].grid(which='major', color='#CCCCCC', linestyle='-')
ax[0].grid(which='minor', color='#CCCCCC', linestyle='--')
ax[0].set_axisbelow(True)

offset = 0.116
for i in range(number_of_layer):
    ax[0].annotate('Timeloop', xy=(i*offset + 0.06, -0.2), xycoords='axes fraction', fontsize=10, rotation=-75)
    ax[0].annotate('Zigzag', xy=(i*offset + 0.11, -0.16), xycoords='axes fraction', fontsize=10, rotation=-75)

ax[1].set_xticks(np.arange(1, len(labels)+1))
ax[1].set_xticklabels(labels, fontsize=11)
ax[1].tick_params(pad=25)

ax[1].set_ylabel("Energy (pJ)", fontsize=15, labelpad=10)
ax[1].legend(loc='upper left', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)
ax[1].set_facecolor('#F2F2F2')

ax[1].tick_params(axis='y', which='major', pad=5)

ax[1].yaxis.set_major_locator(MultipleLocator(0.5e9))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.1e9))
ax[1].grid(which='major', color='#CCCCCC', linestyle='-')
ax[1].grid(which='minor', color='#CCCCCC', linestyle='--')
ax[1].set_axisbelow(True)

offset = 0.116
for i in range(number_of_layer):
    ax[1].annotate('Timeloop', xy=(i*offset + 0.058, -0.2), xycoords='axes fraction', fontsize=10, rotation=-75)
    ax[1].annotate('Zigzag', xy=(i*offset + 0.115, -0.16), xycoords='axes fraction', fontsize=10, rotation=-75)

plt.show()
