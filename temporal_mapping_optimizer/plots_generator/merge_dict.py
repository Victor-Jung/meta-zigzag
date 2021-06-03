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

nn_name = sys.argv[1] #"NASNet_small" "ResNet18" MobileNet_v3_small" "Inception_v3" 
nn_path = "NN_layers/" + nn_name + ".py"
result_path = "results_36/"

if not os.path.exists(result_path + nn_name + "_merged"):
    os.makedirs(result_path + nn_name + "_merged")

for i in range(0, 360, 10):

    with open(result_path + nn_name + "/" + nn_name + "_Arch" + str(i) + ".yaml") as f:
        data_doc = yaml.safe_load(f)
    
    with open(result_path + nn_name + "_meta/" + nn_name + "_Arch" + str(i) + ".yaml") as f:
        data_doc_meta = yaml.safe_load(f)

    for layer_idx in data_doc.keys():
        data_doc[layer_idx]["meta-loma"] = data_doc_meta[layer_idx]["meta-loma"]
        #data_doc[layer_idx] = { "mcmc":data_doc[layer_idx]["mcmc"], "loma":data_doc[layer_idx]["loma"], "meta-loma":data_doc[layer_idx]["meta-loma"]}

    with open(result_path + nn_name + "_merged/" + nn_name + "_Arch" + str(i) + ".yaml", "w") as f:
        yaml.dump(data_doc, f)