import os
import sys
import yaml
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

nn_name = sys.argv[1] #"NASNet_small" "ResNet18" MobileNet_v3_small" "Inception_v3" 
nn_path = "NN_layers/" + nn_name + ".py"
arch_id = 0
result_path = "results_36/"
baseline_result_path = "baseline_results_36/"
float_result_path = "float_results_36/"
ceiling_result_path = "ceiling_results_36/"

baseline_en_avg_list = []
float_en_avg_list = []
ceiling_en_avg_list = []

range36 = range(0, 360, 10)
range10 = range(0, 10, 1)

for arch_id in range10:

    with open(result_path + nn_name + "_baseline" + "/" + nn_name + "_baseline" + "_Arch" + str(arch_id) + ".yaml") as f:
             baseline_data_doc = yaml.safe_load(f)
    with open(result_path + nn_name + "_float" + "/" + nn_name + "_float" + "_Arch" + str(arch_id) + ".yaml") as f:
            float_data_doc = yaml.safe_load(f)
    with open(result_path + nn_name + "_ceiling" + "/" + nn_name + "_ceiling" + "_Arch" + str(arch_id) + ".yaml") as f:
            ceiling_data_doc = yaml.safe_load(f)

    baseline_en_list = []
    float_en_list = []
    ceiling_en_list = []
    layer_idx_list = float_data_doc.keys()

    for layer_idx in layer_idx_list:

        baseline_en_list.append(baseline_data_doc[layer_idx]["meta-loma"]["en"])
        float_en_list.append(float_data_doc[layer_idx]["meta-loma"]["en"])
        ceiling_en_list.append(ceiling_data_doc[layer_idx]["meta-loma"]["en"])

    baseline_en_avg_list.append(np.sum(baseline_en_list))
    float_en_avg_list.append(np.sum(float_en_list))
    ceiling_en_avg_list.append(np.sum(ceiling_en_list))

plt.figure(1)
plt.title("Loop breaking (ceiling and float) evaluation for " + nn_name + "\n prime factor threshold = 7")

plt.bar([*np.arange(0, 10, 1)], baseline_en_avg_list, label="baseline", width=0.3)
plt.bar([*np.arange(0.3, 10, 1)], ceiling_en_avg_list, label="ceiling", width=0.3)
plt.bar([*np.arange(0.6, 10, 1)], float_en_avg_list, label="float", width=0.3)

plt.legend(loc="upper right")
plt.xlabel("HW Idx")
plt.ylabel("Energy")

plt.savefig("./loop_breaking_plot_" + nn_name + ".png")