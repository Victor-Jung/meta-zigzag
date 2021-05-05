from reinforcement_learning_algo.cost_esimator import *

from copy import deepcopy
import numpy as np
import random

def evaluate_tmo(tmo, input_settings, spatial_loop_comb, mem_scheme, layer, mac_costs):
     return get_temporal_loop_estimation(tmo, input_settings, spatial_loop_comb, mem_scheme, layer, mac_costs)


def random_search(tmo, layer, im2col_layer, layer_rounded, spatial_loop_comb, 
                  input_settings, mem_scheme, ii_su, spatial_unrolling):
    
    iter = 2000
    best_utilization = 0
    curr_utilization = 0
    curr_tmo = tmo

    # Initialize mac costs
    mac_costs = calculate_mac_level_costs(layer, layer_rounded, input_settings, mem_scheme, ii_su)

    for k in range(iter):
        random.shuffle(curr_tmo)
        curr_energy, curr_utilization = evaluate_tmo(curr_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)
               
        if(curr_utilization > best_utilization):
            best_utilization = curr_utilization

    print("Best utilization found :", best_utilization)
    return best_utilization

        