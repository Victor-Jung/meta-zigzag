import os
import sys
import ast
import numpy as np
from xml.dom import minidom
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

zigzag_data_path = "../../tl_zz_benchmark/resnet34_zigzag_output/"
meta_zigzag_data_path = "../../tl_zz_benchmark/resnet34_meta_zigzag_output/"
loma_exh_zigzag_data_path = "../../tl_zz_benchmark/resnet34_meta_zigzag_output_even/"
# loma_exh_zigzag_data_path = "../../tl_zz_benchmark/resnet34_loma_exh_zigzag_output/"
loma_zigzag_data_path = "../../tl_zz_benchmark/resnet34_loma_zigzag_output/"
timeloop_data_path = "../../tl_zz_benchmark/resnet34_timeloop_output/"

file_name1 = "ResNet34_eyeriss_tl_L"
file_name2 = "_M1_SU1_min_en.xml"

timeloop_file_name = "resnet34_"
extension = ".stats.txt"

data_dict_list = []
meta_data_dict_list = []
loma_data_dict_list = []
loma_exh_data_dict_list = []

timeloop_energy_list = []
timeloop_memory_energy_list = []

timeloop_time_list = [86,468,224,343,254,304,181,195]

number_of_layer = 8
layer_idx_list = [1,2,9,10,18,19,31,32]

for i in range(number_of_layer):
    timeloop_memory_energy_list.append([0,0,0,0,0,0,0,0])

### Timeloop ###
counter = 1
for i in layer_idx_list:
    output_doc = minidom.parse(zigzag_data_path + file_name1 + str(i) + file_name2)
    data_dict_list.append(output_doc)

    meta_output_doc = minidom.parse(meta_zigzag_data_path + file_name1 + str(i) + file_name2)
    meta_data_dict_list.append(meta_output_doc)

    loma_output_doc = minidom.parse(loma_zigzag_data_path + file_name1 + str(i) + file_name2)
    loma_data_dict_list.append(loma_output_doc)

    loma_exh_output_doc = minidom.parse(loma_exh_zigzag_data_path + file_name1 + str(i) + file_name2)
    loma_exh_data_dict_list.append(loma_exh_output_doc)

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

### Timeloop SU + TM evaluated in ZigZag ###

zigzag_energy_list = []
zigzag_memory_energy_list = []
zigzag_memory_energy_list2 = [[],[],[],[],[],[],[],[]]

zigzag_time_list = []

for i in range(number_of_layer):
    zigzag_energy_list.append(float(data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    zigzag_memory_energy_list.append([ast.literal_eval(data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])
    zigzag_time_list.append(float(data_dict_list[i].getElementsByTagName('elapsed_time_second')[0].nextSibling.data))

for i in range(number_of_layer):
    for j in range(3):
        zigzag_memory_energy_list2[i].append(zigzag_memory_energy_list[i][j][0]) # add spad
    zigzag_memory_energy_list2[i].append(zigzag_memory_energy_list[i][1][1] + zigzag_memory_energy_list[i][2][1]) # add shared buffer
    zigzag_memory_energy_list2[i].append(zigzag_memory_energy_list[i][0][1] + zigzag_memory_energy_list[i][1][2] + zigzag_memory_energy_list[i][2][2]) # add DRAM


### Meta-Loma with ZigZag ###

meta_zigzag_energy_list = []
meta_zigzag_memory_energy_list = []
meta_zigzag_memory_energy_list2 = [[],[],[],[],[],[],[],[]]

meta_zigzag_time_list = []

for i in range(number_of_layer):
    meta_zigzag_energy_list.append(float(meta_data_dict_list[i].getElementsByTagName('total_energy')[0].nextSibling.data))
    meta_zigzag_memory_energy_list.append([ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[1].data),
                                      ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[3].data),
                                      ast.literal_eval(meta_data_dict_list[i].getElementsByTagName('mem_energy_breakdown')[0].childNodes[5].data),
                                    ])
    meta_zigzag_time_list.append(float(meta_data_dict_list[i].getElementsByTagName('elapsed_time_second')[0].nextSibling.data))

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

loma_exh_w_cost = []
loma_exh_i_cost = []
loma_exh_o_cost = []
loma_exh_buff_cost = []
loma_exh_dram_cost = []

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

    loma_exh_w_cost.append(loma_exh_zigzag_memory_energy_list2[i][0])
    loma_exh_i_cost.append(loma_exh_zigzag_memory_energy_list2[i][1])
    loma_exh_o_cost.append(loma_exh_zigzag_memory_energy_list2[i][2])
    loma_exh_buff_cost.append(loma_exh_zigzag_memory_energy_list2[i][3])
    loma_exh_dram_cost.append(loma_exh_zigzag_memory_energy_list2[i][4])    

tl_sum = [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost, timeloop_dram_cost)]
zz_sum = [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost, zigzag_o_cost, zigzag_buff_cost, zigzag_dram_cost)]
loma_sum = [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost, loma_dram_cost)]
meta_sum = [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)]

print("Energy improvement compared to ZigZag Baseline:")
print("Timeloop Baseline  -     Loma 7          -      Meta Loma")
for i in range(len(tl_sum)):
    print(tl_sum[i]/zz_sum[i], " - ", 1 - loma_sum[i]/zz_sum[i], " - ", 1 - meta_sum[i]/zz_sum[i])

print("Search time improvement compared to ZigZag Baseline:")
print("Loma 7          -      Meta Loma")
for i in range(len(timeloop_time_list)):
    print(1 - loma_zigzag_time_list[i]/timeloop_time_list[i], " - ", 1 - meta_zigzag_time_list[i]/timeloop_time_list[i])

### Plotting ###

fig, ax = plt.subplots()
labels = ['Layer 1', 'Layer 2', 'Layer 9', 'Layer 10', "Layer 18", "Layer 19", "Layer 31", "Layer 32"]

w = 0.15
X = np.arange(number_of_layer)

plt.bar(X+(4/6), [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost, timeloop_dram_cost)], label='DRAM', width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar(X+(4/6), [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost)], label='shared buff', width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar(X+(4/6), [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost)], label='O-reg', width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar(X+(4/6), [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost)], label='I-reg', width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar(X+(4/6), timeloop_w_cost, label='W-reg', width = w, color="#264653", linewidth=1, edgecolor='grey')

plt.bar(X+(5/6), [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost, zigzag_o_cost, zigzag_buff_cost, zigzag_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar(X+(5/6), [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost, zigzag_o_cost, zigzag_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar(X+(5/6), [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost, zigzag_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar(X+(5/6), [sum(x) for x in zip(zigzag_w_cost, zigzag_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar(X+(5/6), zigzag_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

plt.bar(X+1, [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost, loma_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar(X+1, [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar(X+1, [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar(X+1, [sum(x) for x in zip(loma_w_cost, loma_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar(X+1, loma_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

plt.bar(X+(7/6), [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost, loma_exh_buff_cost, loma_exh_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar(X+(7/6), [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost, loma_exh_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar(X+(7/6), [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar(X+(7/6), [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar(X+(7/6), loma_exh_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

plt.bar(X+(8/6), [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)], width = w, color="#e76f51", linewidth=1, edgecolor='grey')
plt.bar(X+(8/6), [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost)], width = w, color="#f4a261", linewidth=1, edgecolor='grey')
plt.bar(X+(8/6), [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost)], width = w, color="#e9c46a", linewidth=1, edgecolor='grey')
plt.bar(X+(8/6), [sum(x) for x in zip(meta_w_cost, meta_i_cost)], width = w, color="#2a9d8f", linewidth=1, edgecolor='grey')
plt.bar(X+(8/6), meta_w_cost, width = w, color="#264653", linewidth=1, edgecolor='grey')

offset = 0.1168
for i in range(number_of_layer):
    ax.annotate('TL', xy=(i*offset + 0.045, -0.0375), xycoords='axes fraction', fontsize=10, rotation=-75)
    ax.annotate('TL/ZZ', xy=(i*offset + 0.0650, -0.0625), xycoords='axes fraction', fontsize=10, rotation=-75)
    ax.annotate('LOMA 7 Lpf', xy=(i*offset + 0.085, -0.1075), xycoords='axes fraction', fontsize=10, rotation=-75)
    ax.annotate('LOMA Exh', xy=(i*offset + 0.105, -0.0965), xycoords='axes fraction', fontsize=10, rotation=-75)
    ax.annotate('META-LOMA', xy=(i*offset + 0.122, -0.11), xycoords='axes fraction', fontsize=10, rotation=-75)

plt.xticks(fontsize=13)
ax.set_xticks(np.arange(1, len(labels)+1))
ax.set_xticklabels(labels)
ax.tick_params(pad=65)

plt.title("ResNet34", fontsize=20, pad=15)
plt.ylabel("Energy (pJ)", fontsize=15, labelpad=10)
plt.legend(loc='upper right', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)
ax.set_facecolor('#F2F2F2')

ax.tick_params(axis='y', which='major', pad=5)

ax.yaxis.set_major_locator(MultipleLocator(0.5e9))
ax.yaxis.set_minor_locator(MultipleLocator(0.1e9))
ax.grid(which='major', color='#CCCCCC', linestyle='-')
ax.grid(which='minor', color='#CCCCCC', linestyle='--')
ax.set_axisbelow(True)

fig2, ax2 = plt.subplots()
labels = ['Layer 1', 'Layer 2', 'Layer 9', 'Layer 10', "Layer 18", "Layer 19", "Layer 31", "Layer 32"]

w = 0.18
X = np.arange(number_of_layer)

plt.bar(X+0.7, timeloop_time_list, width = w, color="#e76f51", linewidth=1, edgecolor='grey', label='Timeloop')
plt.bar(X+0.9, loma_exh_zigzag_time_list, width = w, color="#e9c46a", linewidth=1, edgecolor='grey', label='Loma Exhaustive')
plt.bar(X+1.1, loma_zigzag_time_list, width = w, color="#2a9d8f", linewidth=1, edgecolor='grey', label='Loma')
plt.bar(X+1.3, meta_zigzag_time_list, width = w, color="#264653", linewidth=1, edgecolor='grey', label='Meta-Loma')

plt.xticks(fontsize=13)
ax2.set_xticks(np.arange(1, len(labels)+1))
ax2.set_xticklabels(labels)
ax2.tick_params(pad=25)

plt.title("ResNet34", fontsize=20, pad=15)
plt.ylabel("Time (s)", fontsize=15, labelpad=10)
plt.legend(loc='upper right', framealpha=1, ncol=4, edgecolor='grey', fontsize=13)
ax2.set_facecolor('#F2F2F2')

ax2.tick_params(axis='y', which='major', pad=5)

ax2.yaxis.set_major_locator(MultipleLocator(0.5e9))
ax2.yaxis.set_minor_locator(MultipleLocator(0.1e9))
ax2.grid(which='major', color='#CCCCCC', linestyle='-')
ax2.grid(which='minor', color='#CCCCCC', linestyle='--')
ax2.set_axisbelow(True)

plt.yscale('log')

plt.show()