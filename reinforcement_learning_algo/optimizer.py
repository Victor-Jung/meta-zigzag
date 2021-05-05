import numpy as np

from reinforcement_learning_algo.core.state import TemporalMappingState
from reinforcement_learning_algo.MCMC import *
from reinforcement_learning_algo.random_search import *

def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):

    print("--------- Monte Carlo Markov Chain (MCMC) Temporal Mapping Optimization ---------")

    temporal_mapping_pf_ordering = TemporalMappingState(spatial_unrolling, temporal_mapping_ordering,
                                                        layer_architecture=layer.size_list_output_print)

    number_of_run = 10
    min_lpf = 7
    max_lpf = 16
    curr_lpf = min_lpf

    # Get Spatial Unrolling Compressed tmo
    starting_tmo = form_tmo(layer.size_list_output_print, spatial_unrolling)

    print("Generating", number_of_run, "run of MCMC with", max_lpf - min_lpf, 
    "differents lpf size, from", min_lpf, "to", max_lpf, "lpf. For a total of", number_of_run*(max_lpf - min_lpf), "runs.")
    
    for i in range(max_lpf - min_lpf):

        # Generate TMO with given LPF
        

        print("Ordering", i, "of LPF", curr_lpf, ":", )
        curr_lpf += 1
    
    

    best_utilization_sum = 0
    best_utilization = 0
    best_tmo = []
    worst_utilization = 1
    print(temporal_mapping_pf_ordering.value)

    for i in range(number_of_run):
        utilization, tmo = mcmc(temporal_mapping_pf_ordering.value, layer, im2col_layer, layer_rounded, 
                            spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
        #utilization = random_search(temporal_mapping_pf_ordering.value, layer, im2col_layer, layer_rounded, 
         #                   spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
        best_utilization_sum += utilization
        print("TMO :", tmo)

        if utilization > best_utilization:
            best_utilization = utilization
        if utilization < worst_utilization:
            worst_utilization = utilization
    
    print("On", number_of_run, "run :")
    print("Average Utilization :", best_utilization_sum / number_of_run)
    print("Best Utilization :", best_utilization)
    print("Worst Utilization :", worst_utilization)