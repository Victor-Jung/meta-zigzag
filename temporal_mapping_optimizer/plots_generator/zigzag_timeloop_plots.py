import os
import sys
import ast
import numpy as np
from xml.dom import minidom
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

zigzag_data_path = "../../results_test/zigzag_output/"
meta_zigzag_data_path = "../../results_test/meta_zigzag_output/"
loma_zigzag_data_path = "../../results_test/loma_zigzag_output/"
timeloop_data_path = "../../results_test/timeloop_output/"

file_name1 = "AlexNet_eyeriss_tl_L"
file_name2 = "_M1_SU1_min_en.xml"

timeloop_file_name = "alexnet_"
extension = ".stats.txt"

data_dict_list = []
meta_data_dict_list = []
loma_data_dict_list = []

timeloop_energy_list = []
timeloop_memory_energy_list = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

number_of_layer = 4

### Timeloop ###

for i in range(1,number_of_layer + 1):
    output_doc = minidom.parse(zigzag_data_path + file_name1 + str(i) + file_name2)
    data_dict_list.append(output_doc)

    meta_output_doc = minidom.parse(meta_zigzag_data_path + file_name1 + str(i) + file_name2)
    meta_data_dict_list.append(meta_output_doc)

    loma_output_doc = minidom.parse(loma_zigzag_data_path + file_name1 + str(i) + file_name2)
    loma_data_dict_list.append(loma_output_doc)

    timeloop_file = open(timeloop_data_path + timeloop_file_name + str(i) + extension, 'r')
    lines = timeloop_file.readlines()

    for line in lines:
        if line[0:21] == "Total topology energy":
            timeloop_energy_list.append(float(line[23:-3]))

        if line[0:8] == "MACCs = ":
            mac_cost = float(line[8:])
            print(mac_cost)
        if line[4:15] == "psum_spad  ":
            timeloop_memory_energy_list[i-1][2] = float(line[-5:])*mac_cost
        if line[4:18] == "weights_spad  ":
            timeloop_memory_energy_list[i-1][0] = float(line[-5:])*mac_cost
        if line[4:16] == "ifmap_spad  ":
            timeloop_memory_energy_list[i-1][1] = float(line[-5:])*mac_cost
        if line[4:16] == "shared_glb  ":
            timeloop_memory_energy_list[i-1][3] = float(line[-5:])*mac_cost
        if line[4:10] == "DRAM  ":
            timeloop_memory_energy_list[i-1][4] = float(line[-5:])*mac_cost


### Timeloop SU + TM evaluated in ZigZag ###

zigzag_energy_list = []
zigzag_memory_energy_list = []
zigzag_memory_energy_list2 = [[],[],[],[]]

for i in range(number_of_layer):
    zigzag_energy_list.append(float(data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    zigzag_memory_energy_list.append([ast.literal_eval(data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])

for i in range(number_of_layer):
    for j in range(3):
        zigzag_memory_energy_list2[i].append(zigzag_memory_energy_list[i][j][0]) # add spad
    zigzag_memory_energy_list2[i].append(zigzag_memory_energy_list[i][1][1] + zigzag_memory_energy_list[i][2][1]) # add shared buffer
    zigzag_memory_energy_list2[i].append(zigzag_memory_energy_list[i][0][1] + zigzag_memory_energy_list[i][1][2] + zigzag_memory_energy_list[i][2][2]) # add DRAM


### Meta-Loma with ZigZag ###

meta_zigzag_energy_list = []
meta_zigzag_memory_energy_list = []
meta_zigzag_memory_energy_list2 = [[],[],[],[]]

for i in range(number_of_layer):
    meta_zigzag_energy_list.append(float(meta_data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    meta_zigzag_memory_energy_list.append([ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])

for i in range(number_of_layer):
    for j in range(3):
        meta_zigzag_memory_energy_list2[i].append(meta_zigzag_memory_energy_list[i][j][0]) # add spad
    meta_zigzag_memory_energy_list2[i].append(meta_zigzag_memory_energy_list[i][1][1] + meta_zigzag_memory_energy_list[i][2][1]) # add shared buffer
    meta_zigzag_memory_energy_list2[i].append(meta_zigzag_memory_energy_list[i][0][1] + meta_zigzag_memory_energy_list[i][1][2] + meta_zigzag_memory_energy_list[i][2][2]) # add DRAM

### Loma with ZigZag ###

loma_zigzag_energy_list = []
loma_zigzag_memory_energy_list = []
loma_zigzag_memory_energy_list2 = [[],[],[],[]]

for i in range(number_of_layer):
    loma_zigzag_energy_list.append(float(loma_data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    loma_zigzag_memory_energy_list.append([ast.literal_eval(loma_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(loma_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(loma_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])

for i in range(number_of_layer):
    for j in range(3):
        loma_zigzag_memory_energy_list2[i].append(loma_zigzag_memory_energy_list[i][j][0]) # add spad
    loma_zigzag_memory_energy_list2[i].append(loma_zigzag_memory_energy_list[i][1][1] + loma_zigzag_memory_energy_list[i][2][1]) # add shared buffer
    loma_zigzag_memory_energy_list2[i].append(loma_zigzag_memory_energy_list[i][0][1] + loma_zigzag_memory_energy_list[i][1][2] + loma_zigzag_memory_energy_list[i][2][2]) # add DRAM


# Extract the sum of Memory access energy for all layers per memory type
zigzag_w_cost = []
zigzag_i_cost = []
zigzag_o_cost = []
zigzag_buff_cost = []
zigzag_dram_cost = []

timeloop_w_cost = []
timeloop_i_cost = []
timeloop_o_cost = []
timeloop_buff_cost = []
timeloop_dram_cost = []

meta_w_cost = []
meta_i_cost = []
meta_o_cost = []
meta_buff_cost = []
meta_dram_cost = []

loma_w_cost = []
loma_i_cost = []
loma_o_cost = []
loma_buff_cost = []
loma_dram_cost = []

for i in range(number_of_layer):
    
    zigzag_w_cost.append(zigzag_memory_energy_list2[i][0])
    zigzag_i_cost.append(zigzag_memory_energy_list2[i][1])
    zigzag_o_cost.append(zigzag_memory_energy_list2[i][2])
    zigzag_buff_cost.append(zigzag_memory_energy_list2[i][3])
    zigzag_dram_cost.append(zigzag_memory_energy_list2[i][4])

    timeloop_w_cost.append(timeloop_memory_energy_list[i][0])
    timeloop_i_cost.append(timeloop_memory_energy_list[i][1])
    timeloop_o_cost.append(timeloop_memory_energy_list[i][2])
    timeloop_buff_cost.append(timeloop_memory_energy_list[i][3])
    timeloop_dram_cost.append(timeloop_memory_energy_list[i][4])

    meta_w_cost.append(meta_zigzag_memory_energy_list2[i][0])
    meta_i_cost.append(meta_zigzag_memory_energy_list2[i][1])
    meta_o_cost.append(meta_zigzag_memory_energy_list2[i][2])
    meta_buff_cost.append(meta_zigzag_memory_energy_list2[i][3])
    meta_dram_cost.append(meta_zigzag_memory_energy_list2[i][4])

    loma_w_cost.append(loma_zigzag_memory_energy_list2[i][0])
    loma_i_cost.append(loma_zigzag_memory_energy_list2[i][1])
    loma_o_cost.append(loma_zigzag_memory_energy_list2[i][2])
    loma_buff_cost.append(loma_zigzag_memory_energy_list2[i][3])
    loma_dram_cost.append(loma_zigzag_memory_energy_list2[i][4])



### Plotting ###

fig, ax = plt.subplots()
labels = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']

w = 0.18

plt.bar([0.7,1.7,2.7,3.7], [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost, timeloop_dram_cost)], label='DRAM', width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar([0.7,1.7,2.7,3.7], [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost)], label='shared buff', width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar([0.7,1.7,2.7,3.7], [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost)], label='O', width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar([0.7,1.7,2.7,3.7], [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost)], label='I', width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar([0.7,1.7,2.7,3.7], timeloop_w_cost, label='W', width = w, color="#264653", linewidth=1, edgecolor='grey')

plt.bar([0.9,1.9,2.9,3.9], [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost, zigzag_o_cost, zigzag_buff_cost, zigzag_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar([0.9,1.9,2.9,3.9], [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost, zigzag_o_cost, zigzag_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar([0.9,1.9,2.9,3.9], [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost, zigzag_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar([0.9,1.9,2.9,3.9], [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar([0.9,1.9,2.9,3.9], zigzag_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

plt.bar([1.1,2.1,3.1,4.1], [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost, loma_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar([1.1,2.1,3.1,4.1], [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar([1.1,2.1,3.1,4.1], [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar([1.1,2.1,3.1,4.1], [sum(x) for x in zip(loma_w_cost, loma_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar([1.1,2.1,3.1,4.1], loma_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

plt.bar([1.3,2.3,3.3,4.3], [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar([1.3,2.3,3.3,4.3], [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar([1.3,2.3,3.3,4.3], [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar([1.3,2.3,3.3,4.3], [sum(x) for x in zip(meta_w_cost, meta_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar([1.3,2.3,3.3,4.3], meta_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

ax.annotate('TL', xy=(0.175, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('TL/ZZ', xy=(0.175 + 0.027, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('LOMA', xy=(0.175 + 0.065, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('META-LOMA', xy=(0.175 + 0.093, 0.08), xycoords='figure fraction', fontsize=10)

ax.annotate('TL', xy=(0.360, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('TL/ZZ', xy=(0.360 + 0.027, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('LOMA', xy=(0.360 + 0.065, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('META-LOMA', xy=(0.360 + 0.093, 0.08), xycoords='figure fraction', fontsize=10)

ax.annotate('TL', xy=(0.547, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('TL/ZZ', xy=(0.547 + 0.027, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('LOMA', xy=(0.547 + 0.065, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('META-LOMA', xy=(0.547 + 0.093, 0.08), xycoords='figure fraction', fontsize=10)

ax.annotate('TL', xy=(0.735, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('TL/ZZ', xy=(0.735 + 0.027, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('LOMA', xy=(0.735 + 0.065, 0.08), xycoords='figure fraction', fontsize=10)
ax.annotate('META-LOMA', xy=(0.735 + 0.093, 0.08), xycoords='figure fraction', fontsize=10)

plt.xticks(fontsize=13)
ax.set_xticks(np.arange(1, len(labels)+1))
ax.set_xticklabels(labels)
ax.tick_params(pad=25)

plt.title("AlexNet", fontsize=20, pad=15)
plt.ylabel("Energy (pJ)", fontsize=15, labelpad=10)
plt.legend(loc='upper right', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)
ax.set_facecolor('#F2F2F2')

ax.tick_params(axis='y', which='major', pad=5)

ax.yaxis.set_major_locator(MultipleLocator(0.5e9))
ax.yaxis.set_minor_locator(MultipleLocator(0.1e9))
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='--')
ax.set_axisbelow(True)

plt.show()

# for i in range(4):
#     fig, ax = plt.subplots()
#     labels = ['rf_W', 'rf_I', 'rf_O', 'shared_buff', 'DRAM']

    
#     ax.bar([-0.3,0.7,1.7,2.7,3.7], timeloop_memory_energy_list[i], label='TimeLoop', width = 0.2)
#     ax.bar([-0.1,0.9,1.9,2.9,3.9], zigzag_memory_energy_list2[i], label='ZigZag', width = 0.2)

#     plt.bar([0.3,1.3,2.3,3.3,4.3], meta_zigzag_memory_energy_list2[i], label='Meta Zigzag', width = 0.2)

#     ax.set_xticks(np.arange(len(labels)))
#     ax.set_xticklabels(labels)

#     ax.set_title("AlexNet Layer "+ str(i + 1) + " memory access benchmark")
#     ax.set_xlabel("Memory")
#     ax.set_ylabel("Energy")
#     ax.legend(loc='upper right')

# plt.show()