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

############ A VERSION OF PLOT PER LAYER TO COMPARE META LOMA AND LOMA ############

nn_name = sys.argv[1] #"NASNet_small" "ResNet18" MobileNet_v3_small" "Inception_v3" 
nn_path = "NN_layers/" + nn_name + ".py"
arch_id = 0
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

loma_en_list = []
loma_lat_list = []
loma_time_list = []

meta_loma_en_list = []
meta_loma_lat_list = []
meta_loma_time_list = []

layer_range = [*range(1, network_depth + 1, 1)]

with open(result_path + nn_name + "/" + nn_name + "_Arch" + str(arch_id) + ".yaml") as f:
        data_doc = yaml.safe_load(f)

for layer_idx in layer_range:

    if layer_idx in duplicate_layer_idx_dict.keys():
        
        dupl_idx = duplicate_layer_idx_dict[layer_idx] - 1
        
        loma_en_list.append(loma_en_list[dupl_idx])
        loma_lat_list.append(loma_lat_list[dupl_idx])
        loma_time_list.append(loma_time_list[dupl_idx])

        meta_loma_en_list.append(meta_loma_en_list[dupl_idx])
        meta_loma_lat_list.append(meta_loma_lat_list[dupl_idx])
        meta_loma_time_list.append(meta_loma_time_list[dupl_idx])

    else:
        
        loma_en_list.append(data_doc[layer_idx]['loma']['en'])
        loma_lat_list.append(data_doc[layer_idx]['loma']['lat'])
        loma_time_list.append(data_doc[layer_idx]['loma']['exec_time'])

        meta_loma_en_list.append(data_doc[layer_idx]['meta-loma']['en'])
        meta_loma_lat_list.append(data_doc[layer_idx]['meta-loma']['lat'])
        meta_loma_time_list.append(data_doc[layer_idx]['meta-loma']['exec_time'])

arch_range = [*range(1, network_depth + 1, 1)]
arch_range_meta_loma = [x - 0.25 for x in arch_range]
arch_range_loma = [x + 0.25 for x in arch_range]

### Plotting ###
fig, axs = plt.subplots(2, 3)

fig.suptitle(nn_name + " : Meta Loma vs Loma Lpf 7 With Hint C/K")

axs[0, 0].set_title("Energy")
axs[0, 1].set_title("Latency")
axs[0, 2].set_title("Time")

axs[0, 0].bar(["Meta Loma"], statistics.mean(meta_loma_en_list), label='Meta Loma', color='tab:green', width = 0.5, alpha=0.66, linewidth=2)
axs[0, 0].bar(["Loma"], statistics.mean(loma_en_list), label='Loma', color='tab:blue', width = 0.5, alpha=0.66, linewidth=2)
axs[0, 0].set_ylabel("Energy")

axs[0, 1].bar(["Meta Loma"], statistics.mean(meta_loma_lat_list), color='tab:green', width = 0.5, alpha=0.66, linewidth=2)
axs[0, 1].bar(["Loma"], statistics.mean(loma_lat_list), color='tab:blue', width = 0.5, alpha=0.66, linewidth=2)
axs[0, 1].set_ylabel("Latency (cycles)")

axs[0, 2].bar(["Meta Loma"], statistics.mean(meta_loma_time_list), color='tab:green', width = 0.5, alpha=0.66, linewidth=2)
axs[0, 2].bar(["Loma"], statistics.mean(loma_time_list), color='tab:blue', width = 0.5, alpha=0.66, linewidth=2)
axs[0, 2].set_ylabel("Time (s)")

axs[1, 0].bar(arch_range_loma, loma_en_list, color='tab:blue', width = 0.5, alpha=0.66, linewidth=2)
axs[1, 0].bar(arch_range_meta_loma, meta_loma_en_list, color='tab:green', width = 0.5, alpha=0.66, linewidth=2)
axs[1, 0].set_xlabel("Layer")
axs[1, 0].set_ylabel("Energy")

axs[1, 1].bar(arch_range_loma, loma_lat_list, color='tab:blue', width = 0.5, alpha=0.66, linewidth=2)
axs[1, 1].bar(arch_range_meta_loma, meta_loma_lat_list, color='tab:green', width = 0.5, alpha=0.66, linewidth=2)
axs[1, 1].set_xlabel("Layer")
axs[1, 1].set_ylabel("Latency")

axs[1, 2].bar(arch_range_loma, loma_time_list, color='tab:blue', width = 0.5, alpha=0.66, linewidth=2)
axs[1, 2].bar(arch_range_meta_loma, meta_loma_time_list, color='tab:green', width = 0.5, alpha=0.66, linewidth=2)
axs[1, 2].set_xlabel("Layer")
axs[1, 2].set_ylabel("Time (s)")

fig.legend(loc='upper right')
fig.set_size_inches(17, 7)

plt.savefig(result_path + nn_name + "_Layers.png")