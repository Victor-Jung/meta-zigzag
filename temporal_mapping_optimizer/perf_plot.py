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
baseline_result_path = "baseline_results_36/"
float_result_path = "float_results_36/"
ceiling_result_path = "ceiling_results_36/"

baseline_en_avg_list = []
float_en_avg_list = []
ceiling_en_avg_list = []

for arch_id in range(0, 360, 10):

    with open(baseline_result_path + nn_name + "/" + nn_name + "_Arch" + str(arch_id) + ".yaml") as f:
             baseline_data_doc = yaml.safe_load(f)
    with open(float_result_path + nn_name + "/" + nn_name + "_Arch" + str(arch_id) + ".yaml") as f:
            float_data_doc = yaml.safe_load(f)
    with open(ceiling_result_path + nn_name + "/" + nn_name + "_Arch" + str(arch_id) + ".yaml") as f:
            ceiling_data_doc = yaml.safe_load(f)

    baseline_en_list = []
    float_en_list = []
    ceiling_en_list = []
    layer_idx_list = float_data_doc.keys()

    for layer_idx in layer_idx_list:

        baseline_en_list.append(baseline_data_doc[layer_idx]["meta-loma"]["en"])
        float_en_list.append(float_data_doc[layer_idx]["meta-loma"]["en"])
        ceiling_en_list.append(ceiling_data_doc[layer_idx]["meta-loma"]["en"])

    baseline_en_avg_list.append(np.mean(baseline_en_list))
    float_en_avg_list.append(np.mean(float_en_list))
    ceiling_en_avg_list.append(np.mean(ceiling_en_list))

plt.figure(1)
plt.title("Loop breaking (ceiling and float) evaluation for " + nn_name + "\n prime factor threshold = 7")

plt.bar([*range(0, 36)], baseline_en_avg_list, label="baseline")
plt.bar([*range(0, 36)], ceiling_en_avg_list, label="ceiling")
plt.bar([*range(0, 36)], float_en_avg_list, label="float")

plt.legend(loc="upper right")
plt.xlabel("HW Idx")
plt.ylabel("Energy")

plt.savefig("./loop_breaking_plot_" + nn_name + ".png")