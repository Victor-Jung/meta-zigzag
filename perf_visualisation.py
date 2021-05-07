import os
import yaml
import subprocess
from copy import deepcopy
import matplotlib.pyplot as plt

loma_lpf_limit = 8
load_only = False
nn_name = "ResNet18" # ResNet18 layer 2 to 22
layer_idxs = [6, 7]
result_path = "test"

def performance_plot(nn_name, layer_idx, loma_lpf_limit, result_path, load_only):

    ######### Fixed Parameters #########
    settings_file_path = "inputs/settings.yaml"
    data_file_path = "visualisation_data.yaml"
    run_zigzag_command = "python3 top_module.py --arch ./inputs/architecture.yaml --map ./inputs/mapping2.yaml --set ./inputs/settings.yaml --mempool ./inputs/memory_pool.yaml"


    ######### Prepare MCMC settings run it #########
    if not load_only:
        with open(settings_file_path) as f:
            settings_doc = yaml.safe_load(f)

        settings_doc["temporal_mapping_search_method"] = "RL"
        settings_doc["layer_filename"] = "./NN_layers/" + nn_name
        settings_doc["layer_indices"] = [layer_idx]

        with open(settings_file_path, "w") as f:
            yaml.dump(settings_doc, f)

        process = subprocess.Popen(run_zigzag_command.split(), stdout=subprocess.PIPE, text=True)
        output, error = process.communicate()
        print(output)


    ######### Load MCMC utilization list and lpf range then create loma lpf range #########
    with open(data_file_path) as f:
        data_doc = yaml.safe_load(f)

    mcmc_utilization_list = data_doc["mcmc_utilization_list"]
    mcmc_exec_time_list = data_doc["mcmc_exec_time_list"]
    lpf_range = data_doc["lpf_range"]

    if lpf_range[-1] >= loma_lpf_limit:
        loma_lpf_range = [*range(lpf_range[0], loma_lpf_limit + 1)]
    else:
        loma_lpf_range = lpf_range


    ######### Loma exec part until hyperparameter limit (loma_lpf_limit) #########
    if not load_only:
        # Reset loma utilization array
        data_doc["loma_utilization_list"] = []
        data_doc["loma_exec_time_list"] = []
        with open(data_file_path, "w") as f:
                yaml.dump(data_doc, f)

        # Execute loma for the lpf_range of MCMC until 10 lpf
        if lpf_range[0] <= loma_lpf_limit:

            settings_doc["temporal_mapping_search_method"] = "loma"

            with open(settings_file_path, "w") as f:
                yaml.dump(settings_doc, f)

            for lpf in loma_lpf_range:

                settings_doc["max_nb_lpf_layer"] = lpf
                with open(settings_file_path, "w") as f:
                    yaml.dump(settings_doc, f)

                process = subprocess.Popen(run_zigzag_command.split(), stdout=subprocess.PIPE, text=True)
                output, error = process.communicate()
                print(output, error)

    # Get loma results
    with open(data_file_path) as f:
        data_doc = yaml.safe_load(f)
    loma_utilization_list = data_doc["loma_utilization_list"]
    loma_exec_time_list = data_doc["loma_exec_time_list"]

    # Save yaml file
    with open(result_path + "/" + nn_name + "_L" + str(layer_idx) + ".yaml", "x") as f:
        yaml.dump(data_doc, f)

    ######### Result Plotting #########

    # Shift range to avoid bar chart overlapping
    loma_lpf_range_shifted = deepcopy(loma_lpf_range)
    lpf_range_shifted = deepcopy(lpf_range)
    for i in range(len(loma_lpf_range_shifted )):
        loma_lpf_range_shifted[i] += 0.25/2   
    for i in range(len(lpf_range_shifted)):
        lpf_range_shifted[i] -= 0.25/2


    fig, ax1 = plt.subplots()
    plt.title("Loma And MCMC performance for " + nn_name + " L" + str(layer_idx))

    ax1.set_xlabel("Temporal Mapping Size")
    ax1.set_ylabel("Best Utilization Found")
    ax1.plot(lpf_range, mcmc_utilization_list, 'D-', label='mcmc', color='tab:orange')
    ax1.plot(loma_lpf_range, loma_utilization_list, '.-', label='loma', color='tab:blue')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Execution Time (s)')
    ax2.bar(lpf_range_shifted, mcmc_exec_time_list, label='mcmc', color='tab:orange', width = 0.25, alpha=0.66, linewidth=2)
    ax2.bar(loma_lpf_range_shifted , loma_exec_time_list, label='loma', color='tab:blue', width = 0.25, alpha=0.66, linewidth=2)
    ax2.tick_params(axis='y')

    fig.tight_layout()
    plt.legend(loc='upper left')

    plt.savefig(result_path + "/" + nn_name + "_L" + str(layer_idx) + ".png")

for layer_idx in range(layer_idxs[0], layer_idxs[1]+1):
    performance_plot(nn_name, layer_idx, loma_lpf_limit, result_path, load_only)