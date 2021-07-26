import os
import sys
import yaml
import json
import shlex
import subprocess
import statistics
import numpy as np
from pprint import pprint
from copy import deepcopy
import importlib.machinery
import matplotlib.pyplot as plt

sys.path.append(r'/users/micasgst/vjung/Documents/zigzag')
os.chdir('../')

def get_perf_dict_list(result_path):

    perf_dict_list = []

    filename_list = os.listdir(result_path)
    filename_list.sort()

    for filename in filename_list:

        with open(result_path + filename) as f:
            data_doc = yaml.safe_load(f)

        su_dict = dict()

        for layer_idx in data_doc.keys():
            su_dict[layer_idx] = {"en" : data_doc[layer_idx]["mcmc"]["en"],
                                  "su" : data_doc[layer_idx]["mcmc"]["en_su"]}
            
        perf_dict_list.append(su_dict)
    
    return perf_dict_list

def extract_data(perf_dict_list, nn_depth):

    best_energy_list = [float('inf')] * nn_depth
    best_su_list = [0] * nn_depth

    for su_dict in perf_dict_list:
        for layer_idx in su_dict.keys():
            if su_dict[layer_idx]["en"] < best_energy_list[layer_idx - 1]:
                best_energy_list[layer_idx - 1] = su_dict[layer_idx]["en"]
                best_su_list[layer_idx - 1] = su_dict[layer_idx]["su"]
    
    return best_energy_list, best_su_list

def merge_su_loop(su):

    for su_idx, loop_list in enumerate(su):
        merged_loop_list = []
        if loop_list != 0:
            for loop_idx, loop in enumerate(loop_list):
                if any([loop[0] == merged_loop[0] for merged_loop in merged_loop_list]):
                    for merged_idx, merged_loop in enumerate(merged_loop_list):
                        if loop[0] == merged_loop[0]:
                            merged_loop_list[merged_idx] = [merged_loop[0], merged_loop[1]*loop[1]]
                else:
                    merged_loop_list.append(list(loop))
            su[su_idx] = merged_loop_list

    return su


nn_name = sys.argv[1] #"NASNet_small" "ResNet18" MobileNet_v3_small" "Inception_v3" 
plot_path = "temporal_mapping_optimizer/"

result_suopt_path = "results_benchmark/" + nn_name + "_suopt/"
result_suopt2_path = "results_benchmark/" + nn_name + "_suopt_v2/"
result_suopt3_path = "results_benchmark/" + nn_name + "_suopt_v3/"
result_exh_path = "results_benchmark/" + nn_name + "_exh/"
result_hint_path = "results_benchmark/" + nn_name + "_KC/"

# List all energy per layer for all SU
exh_energy = get_perf_dict_list(result_exh_path)
hint_energy = get_perf_dict_list(result_hint_path)
suopt_energy = get_perf_dict_list(result_suopt_path)
suopt2_energy = get_perf_dict_list(result_suopt2_path)
suopt3_energy = get_perf_dict_list(result_suopt3_path)

# Then extract the best energy found by suopt and exh
nn_depth = 8
best_energy_exh, best_su_exh = extract_data(exh_energy, nn_depth)
best_energy_hint, best_su_hint = extract_data(hint_energy, nn_depth)
best_energy_suopt, best_su_suopt = extract_data(suopt_energy, nn_depth)
best_energy_suopt2, best_su_suopt2 = extract_data(suopt2_energy, nn_depth)
best_energy_suopt3, best_su_suopt3 = extract_data(suopt3_energy, nn_depth)

#best_su_exh = merge_su_loop(best_su_exh)
#best_su_hint = merge_su_loop(best_su_hint)
#best_su_suopt = merge_su_loop(best_su_suopt)

# print("Best Energy per Layer Exhaustive :", best_energy_exh)
# print("Best Energy per Layer SUopt :", best_energy_suopt)
for i in range(len(best_energy_exh)):
    if best_energy_exh[i] == float('inf'):
        best_energy_exh[i] = 0
        best_energy_suopt[i] = 0
        best_energy_hint[i] = 0

print("Total Best Energy Exhaustive :", np.sum(best_energy_exh))
print("Total Best Energy Hint :", np.sum(best_energy_hint))
print("Total Best Energy SUopt :", np.sum(best_energy_suopt))
print("Total Best Energy SUopt v2 :", np.sum(best_energy_suopt2))
print("Total Best Energy SUopt v3 :", np.sum(best_energy_suopt3))

# print("Best SU per Layer Exhaustive :")
# for idx, su in enumerate(best_su_exh):
#     print(su, " : ", best_energy_exh[idx])
# print("Best SU per Layer SUopt :")
# for idx, su in enumerate(best_su_suopt):
#     print(su, " : ", best_energy_suopt[idx])

plt.title(nn_name + "Spatial Unrolling Benchmark")

w = 0.20

plt.bar([*np.arange(0, nn_depth, 1)], best_energy_hint, label="hint driven K/C", width=w)
plt.bar([*np.arange(w, nn_depth, 1)], best_energy_exh, label="exhausitive", width=w)
plt.bar([*np.arange(w*2, nn_depth, 1)], best_energy_suopt, label="SUopt", width=w)
plt.bar([*np.arange(w*3, nn_depth, 1)], best_energy_suopt2, label="SUopt v2", width=w)
plt.bar([*np.arange(w*4, nn_depth, 1)], best_energy_suopt3, label="SUopt v3", width=w)

plt.legend(loc="upper right")
plt.xlabel("Layer Idx")
plt.ylabel("Energy")

plt.show()
plt.savefig("./su_plot_" + nn_name + ".png")

# su_counter = 0
# suopt_energy_list = np.zeros((5,nn_depth))
# suopt_su_list = []
# for i in range(5):
#     new = []
#     for j in range(nn_depth):
#         new.append(0)
#     suopt_su_list.append(new)

# for su_dict in suopt_energy:
#     for layer_idx in su_dict.keys():
#         suopt_energy_list[su_counter][layer_idx - 1] = su_dict[layer_idx]["en"]
#         suopt_su_list[su_counter][layer_idx - 1] = su_dict[layer_idx]["su"]
#     su_counter += 1

# print("SU per Layer :")
# for su_idx, layer_list in enumerate(suopt_su_list):
#     print(merge_su_loop([suopt_su_list[su_idx][0]]))
# print("")
# for su_idx, layer_list in enumerate(suopt_su_list):
#     print(merge_su_loop([suopt_su_list[su_idx][1]]))
# print("")
# for su_idx, layer_list in enumerate(suopt_su_list):
#     print(merge_su_loop([suopt_su_list[su_idx][2]]))
# print("")
# for su_idx, layer_list in enumerate(suopt_su_list):
#     print(merge_su_loop([suopt_su_list[su_idx][3]]))
# print("")
# for su_idx, layer_list in enumerate(suopt_su_list):
#     print(merge_su_loop([suopt_su_list[su_idx][4]]))
# print("")

# plt.figure(2)
# plt.title(nn_name + " SUopt perf distrib")

# w = 0.12

# plt.bar([*np.arange(w*0, nn_depth, 1)], best_energy_hint, label="hint driven K/C", width=w)
# plt.bar([*np.arange(w*1, nn_depth, 1)], best_energy_exh, label="exhausitive", width=w)
# plt.bar([*np.arange(w*2, nn_depth, 1)], suopt_energy_list[0], label="su 1", width=w)
# plt.bar([*np.arange(w*3, nn_depth, 1)], suopt_energy_list[1], label="su 2", width=w)
# plt.bar([*np.arange(w*4, nn_depth, 1)], suopt_energy_list[2], label="su 3", width=w)
# plt.bar([*np.arange(w*5, nn_depth, 1)], suopt_energy_list[3], label="su 4", width=w)
# plt.bar([*np.arange(w*6, nn_depth, 1)], suopt_energy_list[4], label="su 5", width=w)


# plt.legend(loc="upper right")
# plt.xlabel("Layer Idx")
# plt.ylabel("Energy")

# plt.show()