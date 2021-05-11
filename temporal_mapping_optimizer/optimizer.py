import matplotlib.pyplot as plt
import numpy as np
import time
import yaml

#from reinforcement_learning_algo.core.state import TemporalMappingState
from temporal_mapping_optimizer.MCMC import *
from temporal_mapping_optimizer.random_search import *
from temporal_mapping_optimizer import loop_type_to_ids, ids_to_loop_type


def mcmc_proba_plot(temporal_mapping_ordering, layer_post, layer, im2col_layer, layer_rounded,
                    spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):
    
    max_iter = 2000 + 1
    min_iter = 100
    iter_interval = 100
    number_of_sample = 1000
    plot_path = "temporal_mapping_optimizer/plots"
    plot_data_path = "temporal_mapping_optimizer/plots_data"
    nn_name = "AlexNet"
    
    max_lpf = get_max_lpf_size(layer.size_list_output_print, spatial_unrolling)
    tmo = get_lpf_limited_tmo(layer_post, spatial_unrolling, max_lpf)
    prob_list = []

    # YAML Dict
    data = dict(
        proba_list = [],
        number_of_iter = []
    )

    # Get the ""Global Optimum""
    global_optimum, tmo, exec_time = mcmc(tmo, 3000, layer, im2col_layer, layer_rounded, spatial_loop_comb, 
                                        input_settings, mem_scheme, ii_su, spatial_unrolling, plot=False)
    #global_optimum = 0.57843696520251 # For AlexNet L3 with default SU and max LPF
    #print("Global Optimum :", global_optimum)

    for i in range(min_iter, max_iter, iter_interval):
        go_counter = 0
        for s in range(number_of_sample):
            utilization, tmo, exec_time = mcmc(tmo, i, layer, im2col_layer, layer_rounded, spatial_loop_comb, 
                                            input_settings, mem_scheme, ii_su, spatial_unrolling, plot=False)
            if utilization == global_optimum:
                go_counter += 1
        prob_list.append(go_counter/number_of_sample)
        print("For", i, "iter of MCMC, proba to reach GO is", go_counter/number_of_sample)

    data["proba_list"] = prob_list
    data["number_of_iter"] = [*range(min_iter, max_iter, iter_interval)]
    plt.plot([*range(min_iter, max_iter, iter_interval)], prob_list, 'D--', label='mcmc')
    plt.title("Reliability of MCMC depending on the number of iteration")
    plt.xlabel("Number of iteration")
    plt.ylabel("Probability of reaching GO")
    plt.legend(loc='upper left')
    
    # Saving Data
    plt.savefig(plot_path + "/go_prob_" + nn_name + "_S" + str(number_of_sample) + "_I" + str(iter_interval) + "_R" + str(min_iter) + "-" + str(max_iter) + ".png")
    with open(plot_data_path + "/go_prob_" + nn_name + "_S" + str(number_of_sample) + "_I" + str(iter_interval) + "_R" + str(min_iter) + "-" + str(max_iter) + ".yaml", 'w') as f:
        yaml.dump(data, f)


def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer_post, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):

    print("--------- Monte Carlo Markov Chain (MCMC) Temporal Mapping Optimization ---------")

    #mcmc_proba_plot(temporal_mapping_ordering, layer_post, layer, im2col_layer, layer_rounded,
    #                spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
    #return

    # Evaluate the min lpf size and the max lpf for the current layer
    min_lpf = get_min_lpf_size(layer.size_list_output_print, spatial_unrolling)
    max_lpf = get_max_lpf_size(layer.size_list_output_print, spatial_unrolling)

    min_lpf = max_lpf - 1
    #max_lpf = 8

    number_of_run = 1
    curr_lpf = min_lpf

    exec_time_list = []
    best_utilization_list = []


    print("Generating", number_of_run, "run of MCMC with", max_lpf - min_lpf, 
    "differents lpf size, from", min_lpf, "to", max_lpf, "lpf. For a total of", number_of_run*(max_lpf - min_lpf), "runs.")
    
    for i in range(max_lpf - min_lpf):

        # Generate TMO with given LPF
        tmo = get_lpf_limited_tmo(layer_post, spatial_unrolling, curr_lpf)

        print("LPF size ", curr_lpf, ":", tmo)
        curr_lpf += 1

        best_utilization_sum = 0
        best_utilization = 0
        exec_time_sum = 0
        best_tmo = []
        best_tmo = []
        worst_utilization = 1

        for i in range(number_of_run):
            utilization, tmo, exec_time = mcmc(tmo, 2000, layer, im2col_layer, layer_rounded, 
                                spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling, plot=False)

            best_utilization_sum += utilization
            exec_time_sum += exec_time

            if utilization > best_utilization:
                best_utilization = utilization
                best_tmo = tmo
            if utilization < worst_utilization:
                worst_utilization = utilization
        
        best_utilization_list.append(best_utilization)
        print("append to t list", exec_time)
        exec_time_list.append(exec_time)
        
        #print("Average Utilization :", best_utilization_sum / number_of_run)
        #print("Worst Utilization :", worst_utilization)
        print("Best Utilization :", best_utilization)
        print("Best tmo :", best_tmo)
        print("Average Exec time on", number_of_run, "runs :", exec_time_sum/number_of_run)  

    # Store result in visualisation_data
    with open("temporal_mapping_optimizer/plots_data/visualisation_data.yaml") as f:
        data_doc = yaml.safe_load(f)

    data_doc["mcmc_utilization_list"] = best_utilization_list
    data_doc["mcmc_exec_time_list"] = exec_time_list
    data_doc["lpf_range"] = [*range(min_lpf, max_lpf)]
    
    with open("temporal_mapping_optimizer/plots_data/visualisation_data.yaml", "w") as f:
        yaml.dump(data_doc, f)


    