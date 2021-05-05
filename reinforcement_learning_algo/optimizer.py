import matplotlib.pyplot as plt
import numpy as np

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

    number_of_run = 5
    curr_lpf = min_lpf
    best_utilization_list = []

    print("Generating", number_of_run, "run of MCMC with", max_lpf - min_lpf, 
    "differents lpf size, from", min_lpf, "to", max_lpf, "lpf. For a total of", number_of_run*(max_lpf - min_lpf), "runs.")
    
    for i in range(max_lpf - min_lpf):

        # Generate TMO with given LPF
        tmo = get_lpf_limited_tmo(layer.size_list_output_print, spatial_unrolling, curr_lpf)

        print("LPF size ", curr_lpf, ":", tmo)
        curr_lpf += 1

        best_utilization_sum = 0
        best_utilization = 0
        best_tmo = []
        worst_utilization = 1

        for i in range(number_of_run):
            utilization, tmo = mcmc(tmo, layer, im2col_layer, layer_rounded, 
                                spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
            #utilization = random_search(temporal_mapping_pf_ordering.value, layer, im2col_layer, layer_rounded, 
            #                   spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
            best_utilization_sum += utilization

            if utilization > best_utilization:
                best_utilization = utilization
            if utilization < worst_utilization:
                worst_utilization = utilization
        
        best_utilization_list.append(best_utilization)
        
        #print("Average Utilization :", best_utilization_sum / number_of_run)
        print("Best Utilization :", best_utilization)
        #print("Worst Utilization :", worst_utilization)

    
    plt.plot([*range(min_lpf, max_lpf)], best_utilization_list)
    plt.show()
    
    