import os
import sys
import ast
import numpy as np
from xml.dom import minidom
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

meta_zigzag_data_path = "../../tl_to_zz_benchmark/resnet34_meta_zigzag_output/"
loma_exh_zigzag_data_path = "../../tl_to_zz_benchmark/resnet34_loma_exh_zigzag_output/"
loma_zigzag_data_path = "../../tl_to_zz_benchmark/resnet34_loma_zigzag_output/"
timeloop_data_path = "../../tl_to_zz_benchmark/resnet34_timeloop_output/"

file_name1 = "ResNet34_eyeriss_tl_L"
file_name2 = "_M1_SU1_min_en.xml"
file_name_ut = "_M1_SU1_max_ut.xml"

timeloop_file_name = "resnet34_"
extension = ".stats.txt"

data_dict_list = []
meta_data_dict_list = []
loma_data_dict_list = []
loma_exh_data_dict_list = []

timeloop_energy_list = []
timeloop_memory_energy_list = []

meta_data_dict_list_lat = []
loma_exh_data_dict_list_lat = []

timeloop_time_list = [86,468,224,343,254,304,181,195]

number_of_layer = 8
layer_idx_list = [1,2,9,10,18,19,31,32]

for i in range(number_of_layer):
    timeloop_memory_energy_list.append([0,0,0,0,0,0,0,0])

### Timeloop ###
counter = 1
for i in layer_idx_list:

    meta_output_doc = minidom.parse(meta_zigzag_data_path + file_name1 + str(i) + file_name2)
    meta_data_dict_list.append(meta_output_doc)

    meta_output_doc_lat = minidom.parse(meta_zigzag_data_path + file_name1 + str(i) + file_name_ut)
    meta_data_dict_list_lat.append(meta_output_doc_lat)

    loma_output_doc = minidom.parse(loma_zigzag_data_path + file_name1 + str(i) + file_name2)
    loma_data_dict_list.append(loma_output_doc)

    loma_exh_output_doc = minidom.parse(loma_exh_zigzag_data_path + file_name1 + str(i) + file_name2)
    loma_exh_data_dict_list.append(loma_exh_output_doc)

    loma_exh_output_doc_lat = minidom.parse(loma_exh_zigzag_data_path + file_name1 + str(i) + file_name_ut)
    loma_exh_data_dict_list_lat.append(loma_exh_output_doc_lat)

    timeloop_file = open(timeloop_data_path + timeloop_file_name + str(i) + extension, 'r')
    lines = timeloop_file.readlines()

    for line in lines:
        if line[0:21] == "Total topology energy":
            timeloop_energy_list.append(float(line[23:-3]))

        if line[0:8] == "MACCs = ":
            mac_cost = float(line[8:])
        if line[4:15] == "psum_spad  ":
            timeloop_memory_energy_list[counter-1][2] = float(line[-5:])*mac_cost
        if line[4:18] == "weights_spad  ":
            timeloop_memory_energy_list[counter-1][0] = float(line[-5:])*mac_cost
        if line[4:16] == "ifmap_spad  ":
            timeloop_memory_energy_list[counter-1][1] = float(line[-5:])*mac_cost
        if line[4:16] == "shared_glb  ":
            timeloop_memory_energy_list[counter-1][3] = float(line[-5:])*mac_cost
        if line[4:10] == "DRAM  ":
            timeloop_memory_energy_list[counter-1][4] = float(line[-5:])*mac_cost
    counter += 1

### Meta-Loma with ZigZag ###

meta_zigzag_energy_list = []
meta_zigzag_memory_energy_list = []
meta_zigzag_memory_energy_list2 = [[],[],[],[],[],[],[],[]]

meta_zigzag_time_list = []
meta_zigzag_latency_list = []

for i in range(number_of_layer):
    meta_zigzag_energy_list.append(float(meta_data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    meta_zigzag_memory_energy_list.append([ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])
    meta_zigzag_time_list.append(float(meta_data_dict_list[i].getElementsByTagName('elapsed_time_second')[0].nextSibling.data))
    meta_zigzag_latency_list.append(float(meta_data_dict_list_lat[i].getElementsByTagName('latency_cycle_without_data_loading')[0].nextSibling.data))

for i in range(number_of_layer):
    for j in range(3):
        meta_zigzag_memory_energy_list2[i].append(meta_zigzag_memory_energy_list[i][j][0]) # add spad
    meta_zigzag_memory_energy_list2[i].append(meta_zigzag_memory_energy_list[i][1][1] + meta_zigzag_memory_energy_list[i][2][1]) # add shared buffer
    meta_zigzag_memory_energy_list2[i].append(meta_zigzag_memory_energy_list[i][0][1] + meta_zigzag_memory_energy_list[i][1][2] + meta_zigzag_memory_energy_list[i][2][2]) # add DRAM

### Loma + Loma exh with ZigZag ###

loma_zigzag_energy_list = []
loma_zigzag_memory_energy_list = []
loma_zigzag_memory_energy_list2 = [[],[],[],[],[],[],[],[]]

loma_exh_zigzag_energy_list = []
loma_exh_zigzag_memory_energy_list = []
loma_exh_zigzag_memory_energy_list2 = [[],[],[],[],[],[],[],[]]

loma_zigzag_time_list = []
loma_exh_zigzag_time_list = []

loma_exh_latency_list = []

for i in range(number_of_layer):
    loma_zigzag_energy_list.append(float(loma_data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    loma_zigzag_memory_energy_list.append([ast.literal_eval(loma_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(loma_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(loma_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])
    loma_zigzag_time_list.append(float(loma_data_dict_list[i].getElementsByTagName('elapsed_time_second')[0].nextSibling.data))

    loma_exh_zigzag_energy_list.append(float(loma_exh_data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    loma_exh_zigzag_memory_energy_list.append([ast.literal_eval(loma_exh_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(loma_exh_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(loma_exh_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])
    loma_exh_zigzag_time_list.append(float(loma_exh_data_dict_list[i].getElementsByTagName('elapsed_time_second')[0].nextSibling.data))
    loma_exh_latency_list.append(float(loma_exh_data_dict_list_lat[i].getElementsByTagName('latency_cycle_without_data_loading')[0].nextSibling.data))

for i in range(number_of_layer):
    for j in range(3):
        loma_zigzag_memory_energy_list2[i].append(loma_zigzag_memory_energy_list[i][j][0]) # add spad
    loma_zigzag_memory_energy_list2[i].append(loma_zigzag_memory_energy_list[i][1][1] + loma_zigzag_memory_energy_list[i][2][1]) # add shared buffer
    loma_zigzag_memory_energy_list2[i].append(loma_zigzag_memory_energy_list[i][0][1] + loma_zigzag_memory_energy_list[i][1][2] + loma_zigzag_memory_energy_list[i][2][2]) # add DRAM

    for j in range(3):
        loma_exh_zigzag_memory_energy_list2[i].append(loma_exh_zigzag_memory_energy_list[i][j][0]) # add spad
    loma_exh_zigzag_memory_energy_list2[i].append(loma_exh_zigzag_memory_energy_list[i][1][1] + loma_exh_zigzag_memory_energy_list[i][2][1]) # add shared buffer
    loma_exh_zigzag_memory_energy_list2[i].append(loma_exh_zigzag_memory_energy_list[i][0][1] + loma_exh_zigzag_memory_energy_list[i][1][2] + loma_exh_zigzag_memory_energy_list[i][2][2]) # add DRAM


# Extract the sum of Memory access energy for all layers per memory type

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

loma_exh_w_cost = []
loma_exh_i_cost = []
loma_exh_o_cost = []
loma_exh_buff_cost = []
loma_exh_dram_cost = []

for i in range(number_of_layer):

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

    loma_exh_w_cost.append(loma_exh_zigzag_memory_energy_list2[i][0])
    loma_exh_i_cost.append(loma_exh_zigzag_memory_energy_list2[i][1])
    loma_exh_o_cost.append(loma_exh_zigzag_memory_energy_list2[i][2])
    loma_exh_buff_cost.append(loma_exh_zigzag_memory_energy_list2[i][3])
    loma_exh_dram_cost.append(loma_exh_zigzag_memory_energy_list2[i][4]) 

# Arithmetic mean
tl_energy_sum = [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost, timeloop_dram_cost)]
meta_energy_sum = [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)]

tl_energy_mean = sum(tl_energy_sum) / number_of_layer
meta_energy_mean = sum(meta_energy_sum) / number_of_layer

print(meta_energy_mean/tl_energy_mean)

tl_time_mean = sum(timeloop_time_list)/number_of_layer
meta_time_mean = sum(meta_zigzag_time_list)/number_of_layer

loma_exh_total_en = sum([sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost, loma_exh_buff_cost, loma_exh_dram_cost)])
salsa_total_en = sum([sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)])

loma_exh_total_lat = sum(loma_exh_latency_list)
salsa_total_lat = sum(meta_zigzag_latency_list)

print([sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost, loma_exh_buff_cost, loma_exh_dram_cost)])
print([sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)])

print(loma_exh_total_en)
print(salsa_total_en)
print(salsa_total_en/loma_exh_total_en - 1)

print(loma_exh_latency_list)
print(meta_zigzag_latency_list)

print(loma_exh_total_lat)
print(salsa_total_lat)
print(salsa_total_lat/loma_exh_total_lat - 1)

### Plotting ###
# We plot the energy of the mapping found by A-LOMA and evaluated with TimeLoop cost model to avoid bias

fig, ax = plt.subplots(2)
labels = ['Layer 1', 'Layer 2', 'Layer 9', 'Layer 10', "Layer 18", "Layer 19", "Layer 31", "Layer 32"]
fig.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0.2, right=0.9, top=0.9, wspace=0.2, hspace=0.025)

w = 0.2
X = np.arange(1,number_of_layer+1)

ax[0].bar(X-(0.30), [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost, timeloop_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.30), [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.30), [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.30), [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.30), timeloop_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

ax[0].bar(X-(0.10), [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost, loma_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.10), [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.10), [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.10), [sum(x) for x in zip(loma_w_cost, loma_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
ax[0].bar(X-(0.10), loma_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

ax[0].bar(X+(0.10), [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost, loma_exh_buff_cost, loma_exh_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.10), [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost, loma_exh_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.10), [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.10), [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.10), loma_exh_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

ax[0].bar(X+(0.30), [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)], width = w, label='DRAM', color="#e76f51", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.30), [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost)], width = w, label='shared_buff', color="#f4a261", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.30), [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost)], width = w, label='O-reg', color="#e9c46a", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.30), [sum(x) for x in zip(meta_w_cost, meta_i_cost)], width = w, label='I-reg', color="#2a9d8f", linewidth=1, edgecolor='grey')
ax[0].bar(X+(0.30), meta_w_cost, width = w, label='W-reg', color="#264653", linewidth=1, edgecolor='grey')

# plt.xticks(fontsize=13)
ax[0].set_xticks([])
# ax[0].set_xticks(np.arange(1, len(labels)+1))
# ax[0].set_xticklabels(labels, fontsize=10)
# ax[0].tick_params(pad=40)

plt.suptitle("ResNet34", fontsize=20, y=0.94)
ax[0].set_ylabel("Energy (pJ)", fontsize=17, labelpad=10)
ax[0].legend(loc='upper left', framealpha=1, ncol=5, edgecolor='grey', fontsize=17, bbox_to_anchor = (0, 1.02))
ax[0].set_facecolor('#F2F2F2')

ax[0].tick_params(axis='y', which='major', pad=5)

ax[0].yaxis.set_major_locator(MultipleLocator(0.5e9))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.1e9))
ax[0].grid(which='major', color='#CCCCCC', linestyle='-')
ax[0].grid(which='minor', color='#CCCCCC', linestyle='--')
ax[0].set_axisbelow(True)

offset = 0.116
for i in range(number_of_layer):
    # ax[0].annotate('Timeloop', xy=(i*offset + 0.0515, -0.18), xycoords='axes fraction', fontsize=10, rotation=-75)
    # ax[0].annotate('LOMA 7', xy=(i*offset + 0.078, -0.15), xycoords='axes fraction', fontsize=10, rotation=-75)
    # ax[0].annotate('LOMA Exh', xy=(i*offset + 0.1035, -0.19), xycoords='axes fraction', fontsize=10, rotation=-75)
    # ax[0].annotate('SALSA', xy=(i*offset + 0.125, -0.15), xycoords='axes fraction', fontsize=10, rotation=-75)

    ax[1].annotate('Timeloop', xy=(i*offset + 0.05, -0.35), xycoords='axes fraction', fontsize=17, rotation=-75)
    ax[1].annotate('LOMA 7', xy=(i*offset + 0.074, -0.295), xycoords='axes fraction', fontsize=17, rotation=-75)
    ax[1].annotate('LOMA Exh', xy=(i*offset + 0.0975, -0.385), xycoords='axes fraction', fontsize=17, rotation=-75)
    ax[1].annotate('SALSA', xy=(i*offset + 0.120, -0.25), xycoords='axes fraction', fontsize=17, rotation=-75)

# Time Plot

w = 0.2
X = np.arange(1, number_of_layer+1)

ax[1].bar(X-0.3, timeloop_time_list, width = w, color="#e76f51", linewidth=1, edgecolor='grey', label='Timeloop')
ax[1].bar(X-0.1, loma_zigzag_time_list, width = w, color="#e9c46a", linewidth=1, edgecolor='grey', label='Loma 7')
ax[1].bar(X+0.1, loma_exh_zigzag_time_list, width = w, color="#e9c46a", linewidth=1, edgecolor='grey', label='Loma Exh')
ax[1].bar(X+0.3, meta_zigzag_time_list, width = w, color="#e9c46a", linewidth=1, edgecolor='grey', label='SALSA')

plt.xticks(fontsize=11)
ax[1].set_xticks(np.arange(1, len(labels)+1))
ax[1].set_xticklabels(labels, fontsize=16)
ax[1].tick_params(pad=100)

ax[1].set_ylabel("Time (s)", fontsize=19, labelpad=0)
#ax[1].legend(loc='upper right', framealpha=1, ncol=4, edgecolor='grey', fontsize=13)
ax[1].set_facecolor('#F2F2F2')

ax[1].tick_params(axis='y', which='major', pad=5)

ax[1].yaxis.set_major_locator(MultipleLocator(0.5e9))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.1e9))
ax[1].grid(which='major', color='#CCCCCC', linestyle='-')
ax[1].grid(which='minor', color='#CCCCCC', linestyle='--')
ax[1].set_axisbelow(True)

ax[1].set_yscale('log')

plt.show()