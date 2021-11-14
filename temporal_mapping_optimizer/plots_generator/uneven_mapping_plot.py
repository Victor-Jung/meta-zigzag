import os
import sys
import ast
import numpy as np
from xml.dom import minidom
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


##### ALEXNET ######

meta_zigzag_data_path = "../../tl_to_zz_benchmark/alexnet_meta_zigzag_output/"
loma_exh_zigzag_data_path = "../../tl_to_zz_benchmark/alexnet_loma_exh_zigzag_output/"
loma_zigzag_data_path = "../../tl_to_zz_benchmark/alexnet_loma_zigzag_output/"
timeloop_data_path = "../../tl_to_zz_benchmark/alexnet_timeloop_output/"

file_name1 = "AlexNet_eyeriss_tl_L"
file_name2 = "_M1_SU1_min_en.xml"

timeloop_file_name = "alexnet_"
extension = ".stats.txt"

data_dict_list = []
meta_data_dict_list = []
loma_data_dict_list = []
loma_exh_data_dict_list = []

timeloop_energy_list = []
timeloop_memory_energy_list = []

timeloop_time_list = [89,230,453,522,395]

number_of_layer = 5
layer_idx_list = [1,2,3,4,5]

for i in range(number_of_layer):
    timeloop_memory_energy_list.append([0,0,0,0,0,0,0,0])

### Timeloop ###
counter = 1
for i in layer_idx_list:

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

alexnet_tl_layer_energy = [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost, timeloop_dram_cost)]
alexnet_loma_layer_energy = [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost, loma_dram_cost)]
alexnet_loma_exh_layer_energy = [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost, loma_exh_buff_cost, loma_exh_dram_cost)]
alexnet_salsa_layer_energy = [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)]

alexnet_tl_layer_time = timeloop_time_list
alexnet_loma_layer_time = loma_zigzag_time_list
alexnet_loma_exh_layer_time = loma_exh_zigzag_time_list
alexnet_salsa_layer_time = meta_zigzag_time_list

alexnet_tl_total_energy = sum(alexnet_tl_layer_energy)
alexnet_loma_total_energy = sum(alexnet_loma_layer_energy)
alexnet_loma_exh_total_energy = sum(alexnet_loma_exh_layer_energy)
alexnet_salsa_total_energy = sum(alexnet_salsa_layer_energy)

alexnet_tl_total_time = sum(alexnet_tl_layer_time)
alexnet_loma_total_time = sum(alexnet_loma_layer_time)
alexnet_loma_exh_total_time = sum(alexnet_loma_exh_layer_time)
alexnet_salsa_total_time = sum(alexnet_salsa_layer_time)

meta_zigzag_data_path = "../../tl_to_zz_benchmark/resnet34_meta_zigzag_output/"
loma_exh_zigzag_data_path = "../../tl_to_zz_benchmark/resnet34_loma_exh_zigzag_output/"
loma_zigzag_data_path = "../../tl_to_zz_benchmark/resnet34_loma_zigzag_output/"
timeloop_data_path = "../../tl_to_zz_benchmark/resnet34_timeloop_output/"

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


resnet_layer_multiplicity = [1, 7, 2, 7, 2, 11, 2, 5]

resnet_tl_layer_energy = [sum(x) for x in zip(timeloop_w_cost, timeloop_i_cost, timeloop_o_cost, timeloop_buff_cost, timeloop_dram_cost)]
resnet_loma_layer_energy = [sum(x) for x in zip(loma_w_cost, loma_i_cost, loma_o_cost, loma_buff_cost, loma_dram_cost)]
resnet_loma_exh_layer_energy = [sum(x) for x in zip(loma_exh_w_cost, loma_exh_i_cost, loma_exh_o_cost, loma_exh_buff_cost, loma_exh_dram_cost)]
resnet_salsa_layer_energy = [sum(x) for x in zip(meta_w_cost, meta_i_cost, meta_o_cost, meta_buff_cost, meta_dram_cost)]

resnet_tl_layer_time = timeloop_time_list
resnet_loma_layer_time = loma_zigzag_time_list
resnet_loma_exh_layer_time = loma_exh_zigzag_time_list
resnet_salsa_layer_time = meta_zigzag_time_list

resnet_tl_layer_energy =  [a*b for a,b in zip(resnet_tl_layer_energy,resnet_layer_multiplicity)]
resnet_loma_layer_energy =  [a*b for a,b in zip(resnet_loma_layer_energy,resnet_layer_multiplicity)]
resnet_loma_exh_layer_energy =  [a*b for a,b in zip(resnet_loma_exh_layer_energy,resnet_layer_multiplicity)]
resnet_salsa_layer_energy =  [a*b for a,b in zip(resnet_salsa_layer_energy,resnet_layer_multiplicity)]

resnet_tl_layer_time =  [a*b for a,b in zip(resnet_tl_layer_time,resnet_layer_multiplicity)]
resnet_loma_layer_time =  [a*b for a,b in zip(resnet_loma_layer_time,resnet_layer_multiplicity)]
resnet_loma_exh_layer_time =  [a*b for a,b in zip(resnet_loma_exh_layer_time,resnet_layer_multiplicity)]
resnet_salsa_layer_time =  [a*b for a,b in zip(resnet_salsa_layer_time,resnet_layer_multiplicity)]

resnet_tl_total_energy = sum(resnet_tl_layer_energy)
resnet_loma_total_energy = sum(resnet_loma_layer_energy)
resnet_loma_exh_total_energy = sum(resnet_loma_exh_layer_energy)
resnet_salsa_total_energy = sum(resnet_salsa_layer_energy)

resnet_tl_total_time = sum(resnet_tl_layer_time)
resnet_loma_total_time = sum(resnet_loma_layer_time)
resnet_loma_exh_total_time = sum(resnet_loma_exh_layer_time)
resnet_salsa_total_time = sum(resnet_salsa_layer_time)

w = 0.9
X = np.arange(1, number_of_layer+1)

fig, ax = plt.subplots(2, 2)
fig.tight_layout()
plt.subplots_adjust(left=0.125, bottom=0.075, right=0.9, top=0.9, wspace=0.2, hspace=0.05)
#plt.suptitle("ResNet34", fontsize=20, y=0.94)
ax[0,0].set_title('AlexNet', fontsize=20, y=1.05)
ax[0,1].set_title('ResNet34', fontsize=20, y=1.05)

ax[0,0].bar(1, alexnet_tl_total_energy, width = w, color="#e76f51", linewidth=1, edgecolor='grey', label='Timeloop')
#ax[0,0].bar(2, alexnet_loma_total_energy, width = w, color="#f4a261", linewidth=1, edgecolor='grey', label='Loma 7')
#ax[0,0].bar(3, alexnet_loma_exh_total_energy, width = w, color="#e9c46a", linewidth=1, edgecolor='grey', label='Loma Exh')
ax[0,0].bar(2, alexnet_salsa_total_energy, width = w, color="#2a9d8f", linewidth=1, edgecolor='grey', label='SALSA')

#ax[0,0].legend(loc='upper left', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)
ax[0,0].set_ylabel("Energy (pJ)", fontsize=15, labelpad=10)
ax[0,0].set_facecolor('#F2F2F2')
ax[0,0].set_ylim([0,3e9])

ax[0,0].tick_params(axis='y', which='major', pad=5)
ax[0,0].set_xticks([])

ax[0,0].yaxis.set_major_locator(MultipleLocator(0.5e9))
ax[0,0].yaxis.set_minor_locator(MultipleLocator(0.1e9))
ax[0,0].grid(which='major', color='#CCCCCC', linestyle='-')
ax[0,0].grid(which='minor', color='#CCCCCC', linestyle='--')
ax[0,0].set_axisbelow(True)

ax[0,0].text(1.70, 2.46e9, '11.7% improvement', fontsize = 13)
ax[0,0].arrow(1.65, 2.65e9, 0, -2.2e8, width=0.015, head_width=0.05, head_length=0.6e8, facecolor="black")

ax[1,0].bar(1, alexnet_tl_total_time, width = w, color="#e76f51", linewidth=1, edgecolor='grey', label='Timeloop')
#ax[1,0].bar(2, alexnet_loma_total_time, width = w, color="#f4a261", linewidth=1, edgecolor='grey', label='Loma 7')
#ax[1,0].bar(3, alexnet_loma_exh_total_time, width = w, color="#e9c46a", linewidth=1, edgecolor='grey', label='Loma Exh')
ax[1,0].bar(2, alexnet_salsa_total_time, width = w, color="#2a9d8f", linewidth=1, edgecolor='grey', label='SALSA')

#ax[0,1].legend(loc='upper left', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)
ax[1,0].set_ylabel("Time (s)", fontsize=15, labelpad=5)
ax[1,0].set_facecolor('#F2F2F2')

ax[1,0].tick_params(axis='y', which='major', pad=5)
ax[1,0].set_xticks([])

ax[1,0].yaxis.set_major_locator(MultipleLocator(0.5e10))
ax[1,0].yaxis.set_minor_locator(MultipleLocator(0.1e10))
ax[1,0].grid(which='major', color='#CCCCCC', linestyle='-')
ax[1,0].grid(which='minor', color='#CCCCCC', linestyle='--')
ax[1,0].set_axisbelow(True)

ax[1,0].text(1.75, 1150, '35x\nspeedup', fontsize = 15)
ax[1,0].arrow(1.65, 54, 0, 1525 - 54, width=0.015, head_width=0.05, head_length=175, facecolor="black")

ax[1,0].set_yscale('log')

ax[0,1].bar(1, resnet_tl_total_energy, width = w, color="#e76f51", linewidth=1, edgecolor='grey', label='Timeloop')
#ax[0,1].bar(2, resnet_loma_total_energy, width = w, color="#f4a261", linewidth=1, edgecolor='grey', label='Loma 7')
#ax[0,1].bar(3, resnet_loma_exh_total_energy, width = w, color="#e9c46a", linewidth=1, edgecolor='grey', label='Loma Exh')
ax[0,1].bar(2, resnet_salsa_total_energy, width = w, color="#2a9d8f", linewidth=1, edgecolor='grey', label='SALSA')

#ax[1,0].legend(loc='upper left', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)
ax[0,1].set_ylabel("Energy (pJ)", fontsize=15, labelpad=10)
ax[0,1].set_facecolor('#F2F2F2')
ax[0,1].set_ylim([0,1.8e10])

ax[0,1].tick_params(axis='y', which='major', pad=5)
ax[0,1].set_xticks([])

ax[0,1].yaxis.set_major_locator(MultipleLocator(0.5e10))
ax[0,1].yaxis.set_minor_locator(MultipleLocator(0.1e10))
ax[0,1].grid(which='major', color='#CCCCCC', linestyle='-')
ax[0,1].grid(which='minor', color='#CCCCCC', linestyle='--')
ax[0,1].set_axisbelow(True)

ax[0,1].text(1.70, 1.57e10, '6.5% improvement', fontsize = 13)
ax[0,1].arrow(1.65, 1.65217e10, 0, -0.7e9, width=0.015, head_width=0.05, head_length=0.3e9, facecolor="black")

ax[1,1].bar(1, resnet_tl_total_time, width = w, color="#e76f51", linewidth=1, edgecolor='grey', label='Timeloop')
#ax[1,1].bar(2, resnet_loma_total_time, width = w, color="#f4a261", linewidth=1, edgecolor='grey', label='Loma 7')
#ax[1,1].bar(3, resnet_loma_exh_total_time, width = w, color="#e9c46a", linewidth=1, edgecolor='grey', label='Loma Exh')
ax[1,1].bar(2, resnet_salsa_total_time, width = w, color="#2a9d8f", linewidth=1, edgecolor='grey', label='SALSA')

#ax[1,1].legend(loc='upper left', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)
ax[1,1].set_ylabel("Time (s)", fontsize=15, labelpad=5)
ax[1,1].set_facecolor('#F2F2F2')

ax[1,1].tick_params(axis='y', which='major', pad=5)
ax[1,1].set_xticks([])

ax[1,1].yaxis.set_major_locator(MultipleLocator(0.5e10))
ax[1,1].yaxis.set_minor_locator(MultipleLocator(0.1e10))
ax[1,1].grid(which='major', color='#CCCCCC', linestyle='-')
ax[1,1].grid(which='minor', color='#CCCCCC', linestyle='--')
ax[1,1].set_axisbelow(True)

ax[1,1].text(1.75, 7960, '17.9x\nspeedup', fontsize = 15)
ax[1,1].arrow(1.65, 420, 0, 10250 - 420, width=0.015, head_width=0.05, head_length=1000, facecolor="black")

ax[1,1].set_yscale('log')

handles, labels = ax[0,0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', framealpha=1, ncol=5, edgecolor='grey', fontsize=13)

plt.show()

