import os
import sys
import yaml
import shlex
import subprocess
sys.path.append(r'/users/micasgst/vjung/Documents/zigzag')
import classes as cls
import loma
from copy import deepcopy
import importlib.machinery
import matplotlib.pyplot as plt

opt = "utilization"
loma_exhaustive = True
load_only = False
nn_name = "MobileNet_v3_small"
nn_path = "NN_layers/" + nn_name + ".py"
plot_data_path = "temporal_mapping_optimizer/plots_data"
plot_path = "temporal_mapping_optimizer/plots"

def get_layer_perf(nn_name, layer_idx, loma_exhaustive, plot_data_path, layer_post):

    ######### Fixed Parameters #########
    settings_file_path = "inputs/settings.yaml"
    data_file_path = "temporal_mapping_optimizer/plots_data/visualisation_data.yaml"
    run_zigzag_command = "python3 top_module.py --arch ./inputs/architecture.yaml --map ./inputs/mapping.yaml --set ./inputs/settings.yaml --mempool ./inputs/memory_pool.yaml"

    ######### Prepare MCMC settings and run it #########
    with open(settings_file_path) as f:
        settings_doc = yaml.safe_load(f)

    settings_doc["temporal_mapping_search_method"] = "RL"
    settings_doc["layer_filename"] = "./NN_layers/" + nn_name
    settings_doc["layer_indices"] = [layer_idx]

    with open(settings_file_path, "w") as f:
        yaml.dump(settings_doc, f)

    process = subprocess.Popen(shlex.split(run_zigzag_command), stdout=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    #output, error = process.communicate()
    #print(output)

    ######### Prepare settings and Exec Loma with the given lpf limit #########
    with open(data_file_path) as f:
        data_doc = yaml.safe_load(f)

    data_doc["loma_ut_list"] = []
    data_doc["loma_en_list"] = []
    data_doc["loma_pareto_score_list"] = []
    data_doc["loma_pareto_en_list"] = []
    data_doc["loma_pareto_ut_list"] = []
    data_doc["loma_exec_time_list"] = []

    with open(data_file_path, "w") as f:
        yaml.dump(data_doc, f)

    # Get Mcmc exec time and su
    with open(data_file_path) as f:
        data_doc = yaml.safe_load(f)

    mcmc_exec_time = data_doc["mcmc_exec_time_list"][0]
    su = data_doc["su"]

    # Find the lpf_limit giving the closest loma execution time to mcmc
    # For 907,200 combinations => 359 sec, we assume linear relation
    if loma_exhaustive:
        lpf_limit = 99
    else:
        lpf_limit = 5
        delta_t_list = []

        for i in range(0, 10):
            lpf_limit += 1
            tl_list, nonmerged_count_dict, loop_type_order, tl_combinations = loma.og(layer_post, su, lpf_limit)
            estimated_loma_exec_time = (359/907200)*tl_combinations
            delta_t_list.append(abs(mcmc_exec_time - estimated_loma_exec_time))

        lpf_limit = delta_t_list.index(min(delta_t_list)) + 5
        print("TD with Mcmc :", min(delta_t_list))

    print("LPF Limit :", lpf_limit)
    settings_doc["temporal_mapping_search_method"] = "loma"
    settings_doc["max_nb_lpf_layer"] = lpf_limit

    with open(settings_file_path, "w") as f:
        yaml.dump(settings_doc, f)

    process = subprocess.Popen(run_zigzag_command.split(), stdout=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()

    ######### Load and return MCMC and LOMA result for this layer #########
    with open(data_file_path) as f:
        data_doc = yaml.safe_load(f)

    if opt == "energy":
        mcmc_en = data_doc["mcmc_en_list"][0]
    elif opt == "utilization":
        mcmc_ut = data_doc["mcmc_ut_list"][0]
    elif opt == "pareto":
        mcmc_en = data_doc["mcmc_en_list"][0]
        mcmc_ut = data_doc["mcmc_ut_list"][0]
        mcmc_pareto_en = data_doc["mcmc_pareto_en_list"][0]
        mcmc_pareto_ut = data_doc["mcmc_pareto_ut_list"][0]
        mcmc_pareto_score = data_doc["mcmc_pareto_list"][0]

    loma_en = data_doc["loma_en_list"][0]
    loma_ut = data_doc["loma_ut_list"][0]
    loma_pareto_score = data_doc["loma_pareto_score_list"][0]
    loma_pareto_en = data_doc["loma_pareto_en_list"][0]
    loma_pareto_ut = data_doc["loma_pareto_ut_list"][0]

    loma_exec_time = data_doc["loma_exec_time_list"][0]

    if opt == "energy":
        return [mcmc_en, loma_en, su]
    elif opt == "utilization":
        return [mcmc_ut, loma_ut, su]
    elif opt == "pareto":
        return [mcmc_pareto_en, mcmc_pareto_ut, mcmc_pareto_score, loma_pareto_en, loma_pareto_ut, loma_pareto_score, su]

### Load Neural Network layers info ###
layer_spec = importlib.machinery.SourceFileLoader(nn_name, nn_path).load_module()
layers = [cls.Layer.extract_layer_info(layer_spec.layer_info[layer_number+1]) for layer_number in range(len(layer_spec.layer_info.items()))]
layer_post_list = []

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
    layer_post_list.append({'B': layer.B, 'K': layer.K, 'C': layer.C, 'OY': layer.OY, 'OX': layer.OX,
    'FY': layer.FY, 'FX': layer.FX, 'SY': layer.SY, 'SX': layer.SX, 'SFY': layer.SFY, 'SFX': layer.SFX, 'G': layer.G})

### Run and collect data for plotting ###
full_layer_range = [*range(1, len(layer_spec.layer_info.items()) + 1)]
mcmc_perf = []
loma_perf = []
su_list = []

for layer_idx in full_layer_range:

    print(nn_name + " Layer " + str(layer_idx), " :")

    if layer_idx in duplicate_layer_idx_dict:
        result = [mcmc_perf[duplicate_layer_idx_dict[layer_idx] - 1], loma_perf[duplicate_layer_idx_dict[layer_idx] - 1]]
    else:
        result = get_layer_perf(nn_name, layer_idx, loma_exhaustive, plot_data_path, layer_post_list[layer_idx - 1])

    if opt == "energy" or opt == "utilization":
        mcmc_perf.append(result[0])
        loma_perf.append(result[1])
        su_list.append(result[2])
    elif opt == "pareto":
        mcmc_perf.append(result[0:3])
        loma_perf.append(result[3:6])
        su_list.append(result[6])

### Save Data ###
data_save = dict()
data_save['mcmc_perf'] = mcmc_perf
data_save['loma_perf'] = loma_perf
data_save['su_list'] = su_list

with open(plot_data_path + "/" + nn_name + "_" + opt + ".yaml", "w") as f:
    yaml.dump(data_save, f)

### Plotting ###
plt.title("Loma And MCMC performance for " + nn_name)

if opt == "energy":
    plt.ylabel('Energy')
elif opt == "utilization":
    plt.ylabel('Utilization')
elif opt == "pareto":
    plt.ylabel('Pareto Score')
plt.xlabel('Layer idx')

mcmc_layer_range = []
loma_layer_range = []

for i in range(len(full_layer_range)):
    loma_layer_range.append(full_layer_range[i] + 0.5/2)
    mcmc_layer_range.append(full_layer_range[i] - 0.5/2)


plt.bar(mcmc_layer_range, mcmc_perf, label='mcmc', color='tab:orange', width = 0.5, alpha=0.66, linewidth=2)
plt.bar(loma_layer_range, loma_perf, label='loma', color='tab:blue', width = 0.5, alpha=0.66, linewidth=2)
plt.legend(loc='lower right')

plt.savefig(plot_path + "/" + nn_name + "_" + opt + ".png")
plt.show()
