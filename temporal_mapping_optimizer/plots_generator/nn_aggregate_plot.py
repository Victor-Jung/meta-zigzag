import os
import sys
import yaml
import shlex
import subprocess
import statistics
from pprint import pprint
from copy import deepcopy
import importlib.machinery
import matplotlib.pyplot as plt

sys.path.append(r'/users/micasgst/vjung/Documents/zigzag')
os.chdir('../../')

import loma
import classes as cls

opt = "utilization"
loma_exhaustive = True
load_only = False
nn_name = sys.argv[1] #"NASNet_small" "ResNet18" MobileNet_v3_small" "Inception_v3" 
nn_path = "NN_layers/" + nn_name + ".py"
result_path = "results_36/"

### Load Neural Network layers info ###
layer_spec = importlib.machinery.SourceFileLoader(nn_name, nn_path).load_module()
layers = [cls.Layer.extract_layer_info(layer_spec.layer_info[layer_number+1]) for layer_number in range(len(layer_spec.layer_info.items()))]
network_depth = len(layer_spec.layer_info.items())

### Make sure we don't run Mcmc and Loma twice for similar Layer ###
duplicate_layer_idx_dict = dict()
layer_range = [*range(1, len(layer_spec.layer_info.items()) + 1)]
for idx, layer in enumerate(layers):
    layers_seen = layers[:idx]
    for idx_other, other in enumerate(layers_seen):
        if layer == other:
            duplicate_layer_idx_dict[idx + 1] = idx_other + 1
            layer_range.remove(idx + 1)
            break

mcmc_arch_perf_list = []
mcmc_arch_en_avg_list = []
mcmc_arch_ut_avg_list = []
mcmc_arch_lat_avg_list = []
mcmc_arch_time_avg_list = []

loma_arch_perf_list = []
loma_arch_en_avg_list = []
loma_arch_lat_avg_list = []
loma_arch_time_avg_list = []

meta_loma_arch_perf_list = []
meta_loma_arch_en_avg_list = []
meta_loma_arch_lat_avg_list = []
meta_loma_arch_time_avg_list = []

for i in range(0, 360, 10):

    mcmc_net_perf_list = []
    loma_net_perf_list = []
    meta_loma_net_perf_list = []

    mcmc_en_sum = 0
    mcmc_lat_sum = 0
    mcmc_time_sum = 0

    loma_en_sum = 0
    loma_lat_sum = 0
    loma_time_sum = 0

    meta_loma_en_sum = 0
    meta_loma_lat_sum = 0
    meta_loma_time_sum = 0

    with open(result_path + nn_name + "/" + nn_name + "_Arch" + str(i) + ".yaml") as f:
        data_doc = yaml.safe_load(f)

    for layer_idx in layer_range:
        if layer_idx in duplicate_layer_idx_dict.keys():
            mcmc_net_perf_list.append(mcmc_net_perf_list[duplicate_layer_idx_dict[layer_idx]])
            loma_net_perf_list.append(loma_net_perf_list[duplicate_layer_idx_dict[layer_idx]])
            meta_loma_net_perf_list.append(meta_loma_net_perf_list[duplicate_layer_idx_dict[layer_idx]])

            mcmc_en_sum += mcmc_net_perf_list[duplicate_layer_idx_dict[layer_idx]][0]
            mcmc_lat_sum += mcmc_net_perf_list[duplicate_layer_idx_dict[layer_idx]][1]
            mcmc_time_sum += mcmc_net_perf_list[duplicate_layer_idx_dict[layer_idx]][2]

            loma_en_sum += loma_net_perf_list[duplicate_layer_idx_dict[layer_idx]][0]
            loma_lat_sum += loma_net_perf_list[duplicate_layer_idx_dict[layer_idx]][1]
            loma_time_sum += loma_net_perf_list[duplicate_layer_idx_dict[layer_idx]][2]

            meta_loma_en_sum += meta_loma_net_perf_list[duplicate_layer_idx_dict[layer_idx]][0]
            meta_loma_lat_sum += meta_loma_net_perf_list[duplicate_layer_idx_dict[layer_idx]][1]
            meta_loma_time_sum += meta_loma_net_perf_list[duplicate_layer_idx_dict[layer_idx]][2]
        else:
            mcmc_net_perf_list.append([data_doc[layer_idx]['mcmc']['en'], data_doc[layer_idx]['mcmc']['lat'], data_doc[layer_idx]['mcmc']['exec_time']])
            loma_net_perf_list.append([data_doc[layer_idx]['loma']['en'], data_doc[layer_idx]['loma']['lat'], data_doc[layer_idx]['loma']['exec_time']])
            meta_loma_net_perf_list.append([data_doc[layer_idx]['meta-loma']['en'], data_doc[layer_idx]['meta-loma']['lat'], data_doc[layer_idx]['meta-loma']['exec_time']])
            
            mcmc_en_sum += data_doc[layer_idx]['mcmc']['en']
            mcmc_lat_sum += data_doc[layer_idx]['mcmc']['lat']
            mcmc_time_sum += data_doc[layer_idx]['mcmc']['exec_time']
            
            loma_en_sum += data_doc[layer_idx]['loma']['en']
            loma_lat_sum += data_doc[layer_idx]['loma']['lat']
            loma_time_sum += data_doc[layer_idx]['loma']['exec_time']

            meta_loma_en_sum += data_doc[layer_idx]['meta-loma']['en']
            meta_loma_lat_sum += data_doc[layer_idx]['meta-loma']['lat']
            meta_loma_time_sum += data_doc[layer_idx]['meta-loma']['exec_time']

    mcmc_arch_en_avg_list.append(mcmc_en_sum / network_depth)
    mcmc_arch_lat_avg_list.append(mcmc_lat_sum / network_depth)
    mcmc_arch_time_avg_list.append(mcmc_time_sum / network_depth)

    loma_arch_en_avg_list.append(loma_en_sum / network_depth)
    loma_arch_lat_avg_list.append(loma_lat_sum / network_depth)
    loma_arch_time_avg_list.append(loma_time_sum / network_depth)

    meta_loma_arch_en_avg_list.append(meta_loma_en_sum / network_depth)
    meta_loma_arch_lat_avg_list.append(meta_loma_lat_sum / network_depth)
    meta_loma_arch_time_avg_list.append(meta_loma_time_sum / network_depth)

    mcmc_arch_perf_list.append(mcmc_net_perf_list)
    loma_arch_perf_list.append(loma_net_perf_list)
    meta_loma_arch_perf_list.append(meta_loma_net_perf_list)

arch_range = [*range(1, 37, 1)]
arch_range_meta_loma = [x - 0.5 + (0.33/2) for x in arch_range]
arch_range_mcmc = [x - 0.5 + (0.33 + 0.33/2) for x in arch_range]
arch_range_loma = [x - 0.5 + (0.66 + 0.33/2) for x in arch_range]

### Plotting ###
fig, axs = plt.subplots(2, 3)

axs[0, 0].set_title(nn_name + " Energy")
axs[0, 1].set_title(nn_name + " Latency")
axs[0, 2].set_title(nn_name + " Time")

axs[0, 0].bar([0], statistics.mean(meta_loma_arch_en_avg_list), label='meta_loma', color='tab:green', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 0].bar([1], statistics.mean(mcmc_arch_en_avg_list), label='mcmc', color='tab:orange', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 0].bar([2], statistics.mean(loma_arch_en_avg_list), label='loma', color='tab:blue', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 0].set_ylabel("Energy")

axs[0, 1].bar([0], statistics.mean(meta_loma_arch_lat_avg_list), label='meta_loma', color='tab:green', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 1].bar([1], statistics.mean(mcmc_arch_lat_avg_list), label='mcmc', color='tab:orange', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 1].bar([2], statistics.mean(loma_arch_lat_avg_list), label='loma', color='tab:blue', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 1].set_ylabel("Latency (cycles)")

axs[0, 2].bar([0], statistics.mean(meta_loma_arch_time_avg_list), label='meta_loma', color='tab:green', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 2].bar([1], statistics.mean(mcmc_arch_time_avg_list), label='mcmc', color='tab:orange', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 2].bar([2], statistics.mean(loma_arch_time_avg_list), label='loma', color='tab:blue', width = 0.33, alpha=0.66, linewidth=2)
axs[0, 2].set_ylabel("Time (s)")

axs[1, 0].bar(arch_range_mcmc, mcmc_arch_en_avg_list, label='mcmc', color='tab:orange', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 0].bar(arch_range_loma, loma_arch_en_avg_list, label='loma', color='tab:blue', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 0].bar(arch_range_meta_loma, meta_loma_arch_en_avg_list, label='meta_loma', color='tab:green', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 0].set_xlabel("Arch")
axs[1, 0].set_ylabel("Energy")

axs[1, 1].bar(arch_range_mcmc, mcmc_arch_lat_avg_list, label='mcmc', color='tab:orange', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 1].bar(arch_range_loma, loma_arch_lat_avg_list, label='loma', color='tab:blue', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 1].bar(arch_range_meta_loma, meta_loma_arch_lat_avg_list, label='meta_loma', color='tab:green', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 1].set_xlabel("Arch")
axs[1, 1].set_ylabel("Latency")

axs[1, 2].bar(arch_range_mcmc, mcmc_arch_time_avg_list, label='mcmc', color='tab:orange', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 2].bar(arch_range_loma, loma_arch_time_avg_list, label='loma', color='tab:blue', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 0].bar(arch_range_meta_loma, meta_loma_arch_time_avg_list, label='meta_loma', color='tab:green', width = 0.33, alpha=0.66, linewidth=2)
axs[1, 2].set_xlabel("Arch")
axs[1, 2].set_ylabel("Latency")

fig.legend(loc='upper right')
fig.set_size_inches(14, 7)

print(nn_name)
print("Energy improvement", 1 - statistics.mean(mcmc_arch_en_avg_list) / statistics.mean(loma_arch_en_avg_list))
print("Latency improvement", 1 - statistics.mean(mcmc_arch_lat_avg_list) / statistics.mean(loma_arch_lat_avg_list))

plt.savefig(result_path + nn_name + ".png")
