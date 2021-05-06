import numpy as np
import time
import yaml

from reinforcement_learning_algo.core.state import TemporalMappingState
from reinforcement_learning_algo.MCMC import *
from reinforcement_learning_algo.random_search import *
from reinforcement_learning_algo import loop_type_to_ids

def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer_post, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):

    print("--------- Monte Carlo Markov Chain (MCMC) Temporal Mapping Optimization ---------")

    # Evaluate the min lpf size and the max lpf for the current layer
    min_lpf = get_min_lpf_size(layer.size_list_output_print, spatial_unrolling)
    max_lpf = get_max_lpf_size(layer.size_list_output_print, spatial_unrolling)

    #min_lpf = 6
    #max_lpf = 8

    number_of_run = 2
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
            utilization, tmo, exec_time = mcmc(tmo, layer, im2col_layer, layer_rounded, 
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
    with open("visualisation_data.yaml") as f:
        data_doc = yaml.safe_load(f)

    print("exec t list ", exec_time_list)
    data_doc["mcmc_utilization_list"] = best_utilization_list
    data_doc["mcmc_exec_time_list"] = exec_time_list
    data_doc["lpf_range"] = [*range(min_lpf, max_lpf)]
    
    with open("visualisation_data.yaml", "w") as f:
        yaml.dump(data_doc, f)
