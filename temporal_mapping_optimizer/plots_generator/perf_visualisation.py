import os
import yaml
import subprocess
from copy import deepcopy
import matplotlib.pyplot as plt

opt = "pareto"
loma_lpf_limit = 10
load_only = False
nn_name = "ResNet18"
layer_idxs = [2]
plot_data_path = "temporal_mapping_optimizer/plots_data"
plot_path = "temporal_mapping_optimizer/plots"

def performance_plot(nn_name, layer_idx, loma_lpf_limit, plot_path, plot_data_path, load_only):

    ######### Fixed Parameters #########
    settings_file_path = "inputs/settings.yaml"
    data_file_path = "temporal_mapping_optimizer/plots_data/visualisation_data.yaml"
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
        '''while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:

                print(output.strip())
        rc = process.poll()'''
        output, error = process.communicate()
        print(output)


    ######### Load MCMC utilization list and lpf range then create loma lpf range #########
    with open(data_file_path) as f:
        data_doc = yaml.safe_load(f)

    if opt == "energy":
        mcmc_en_list = data_doc["mcmc_en_list"]
    elif opt == "utilization":
        mcmc_ut_list = data_doc["mcmc_ut_list"]
    elif opt == "pareto":
        mcmc_en_list = data_doc["mcmc_en_list"]
        mcmc_ut_list = data_doc["mcmc_ut_list"]
        mcmc_pareto_en_list = data_doc["mcmc_pareto_en_list"]
        mcmc_pareto_ut_list = data_doc["mcmc_pareto_ut_list"]
        mcmc_pareto_list = data_doc["mcmc_pareto_list"]
    
    mcmc_exec_time_list = data_doc["mcmc_exec_time_list"]
    lpf_range = data_doc["lpf_range"]

    if lpf_range[-1] >= loma_lpf_limit:
        loma_lpf_range = [*range(lpf_range[0], loma_lpf_limit + 1)]
    else:
        loma_lpf_range = lpf_range


    ######### Loma exec part until hyperparameter limit (loma_lpf_limit) #########
    if not load_only:
        # Reset loma utilization array
        data_doc["loma_ut_list"] = []
        data_doc["loma_en_list"] = []
        data_doc["loma_pareto_score_list"] = []
        data_doc["loma_pareto_en_list"] = []
        data_doc["loma_pareto_ut_list"] = []
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
    loma_ut_list = data_doc["loma_ut_list"]
    loma_en_list = data_doc["loma_en_list"]
    loma_pareto_score_list = data_doc["loma_pareto_score_list"]
    loma_pareto_en_list = data_doc["loma_pareto_en_list"]
    loma_pareto_ut_list = data_doc["loma_pareto_ut_list"]
    loma_exec_time_list = data_doc["loma_exec_time_list"]

    # Save yaml file
    with open(plot_data_path + "/" + nn_name + "_L" + str(layer_idx) + "_" + opt + ".yaml", "w") as f:
        yaml.dump(data_doc, f)

    ######### Plot Preprocessing #########

    # Shift range to avoid bar chart overlapping
    loma_lpf_range_shifted = deepcopy(loma_lpf_range)
    lpf_range_shifted = deepcopy(lpf_range)
    for i in range(len(loma_lpf_range_shifted )):
        loma_lpf_range_shifted[i] += 0.25/2   
    for i in range(len(lpf_range_shifted)):
        lpf_range_shifted[i] -= 0.25/2

    pareto_en_ratio_list = []
    pareto_ut_ratio_list = []

    if opt == "pareto": 
        for i in range(len(lpf_range)):
            pareto_en_ratio_list.append(mcmc_en_list[i] / mcmc_pareto_en_list[i])
            pareto_ut_ratio_list.append(mcmc_pareto_ut_list[i] / mcmc_ut_list[i])

    print("pareto_en_ratio", pareto_en_ratio_list)
    print("pareto_ut_ratio", pareto_ut_ratio_list)

    ######### Result Plotting #########

    fig, ax1 = plt.subplots()
    plt.title("Loma And MCMC performance for " + nn_name + " L" + str(layer_idx))

    ax1.set_ylabel('Execution Time (s)')
    ax1.bar(lpf_range_shifted, mcmc_exec_time_list, label='mcmc', color='tab:orange', width = 0.25, alpha=0.66, linewidth=2)
    ax1.bar(loma_lpf_range_shifted , loma_exec_time_list, label='loma', color='tab:blue', width = 0.25, alpha=0.66, linewidth=2)
    ax1.tick_params(axis='y')
    
    ax2 = ax1.twinx()
    ax2.set_xlabel("Temporal Mapping Size")
    if opt == "energy":
        ax2.set_ylabel("Best Energy Found")
        ax2.plot(lpf_range, mcmc_en_list, 'D-', label='mcmc', color='tab:orange')
        ax2.plot(loma_lpf_range, loma_en_list, '.-', label='loma', color='tab:blue')
    elif opt == "utilization":
        ax2.set_ylabel("Best Utilization Found")
        ax2.plot(lpf_range, mcmc_ut_list, 'D-', label='mcmc', color='tab:orange')
        ax2.plot(loma_lpf_range, loma_ut_list, '.-', label='loma', color='tab:blue')
    elif opt == "pareto":
        ax2.set_ylabel("Best Pareto Score (Energy / Utilization)")
        ax2.plot(lpf_range, mcmc_pareto_list, 'D-', label='mcmc', color='tab:orange')
        ax2.plot(loma_lpf_range, loma_pareto_score_list, '.-', label='loma', color='tab:blue')
        for i in range(len(lpf_range)):
            ax2.annotate('{:.2f},\n{:.2f}'.format(pareto_en_ratio_list[i], pareto_ut_ratio_list[i]), (lpf_range[i] + 0.15, mcmc_pareto_list[i] + 0.25))
    ax2.tick_params(axis='y')

    fig.tight_layout()
    if opt == "energy" or opt == "pareto":
       plt.legend(loc='upper right')
    elif opt == "utilization" or opt == "pareto":
       plt.legend(loc='upper left')

    plt.savefig(plot_path + "/" + nn_name + "_L" + str(layer_idx) + "_" + opt + ".png")

if len(layer_idxs) > 1:
    for layer_idx in range(layer_idxs[0], layer_idxs[1]+1):
        performance_plot(nn_name, layer_idx, loma_lpf_limit, plot_path, plot_data_path, load_only)
else:
    performance_plot(nn_name, layer_idxs[0], loma_lpf_limit, plot_path, plot_data_path, load_only)