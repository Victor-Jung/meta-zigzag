import matplotlib.pyplot as plt
import numpy as np
import time
import yaml

#from reinforcement_learning_algo.core.state import TemporalMappingState
from temporal_mapping_optimizer.MCMC import *
from temporal_mapping_optimizer.random_search import *
from temporal_mapping_optimizer.cost_esimator import *
from temporal_mapping_optimizer import loop_type_to_ids, ids_to_loop_type


def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer_post, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):

    print("--------- Simulated Annealing (SA) Temporal Mapping Optimization ---------")

    opt = "energy"

    if opt == "energy":
        optimize("energy", temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
    elif opt == "utilization":
        optimize("utilization", temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
    elif opt == "pareto":
        optimize("energy", temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
        optimize("utilization", temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
        optimize("pareto", temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
    

def optimize(opt, temporal_mapping_ordering, layer_post, layer, im2col_layer, 
            layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):

    # Initialize mac costs
    mac_costs = calculate_mac_level_costs(layer, layer_rounded, input_settings, mem_scheme, ii_su)

    iter = 2000

    exec_time_list = []
    best_value_list = []
    pareto_en_list = []
    pareto_ut_list = []
    best_tmo = []
    best_su = []

    if opt == "energy" or opt == "pareto":
        best_value = float('inf')
    elif opt == "utilization":
        best_value = 0

    # Generate TMO with given LPF
    starting_tmo = get_lpf_tmo(layer_post, spatial_unrolling)

    value, tmo, su, exec_time = mcmc(starting_tmo, iter, layer, im2col_layer, layer_rounded, 
                                spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling, layer_post, opt, plot=False)
    
    if ((opt == "energy" or opt == "pareto") and value < best_value) or (opt == "utilization" and value > best_value):
        best_tmo = tmo
        best_su = su
        best_value = value
        best_exec_time = exec_time

    best_value_list.append(best_value)
    exec_time_list.append(best_exec_time)

    if opt == "pareto":
        pareto_en, pareto_ut = get_temporal_loop_estimation(best_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)
        pareto_en_list.append(pareto_en.item())
        pareto_ut_list.append(pareto_ut)
    
    if opt == "energy":
        print("Best Energy :", best_value)
    elif opt == "utilization":
        print("Best Utilization :", best_value)
    elif opt == "pareto":
        print("Best Pareto Score :", best_value)

    print("Best tmo :", best_tmo)
    print("Best su :", best_su.items)
    print("Exec time", exec_time)  

    # Store result in visualisation_data
    with open("temporal_mapping_optimizer/plots_data/visualisation_data.yaml") as f:
        data_doc = yaml.safe_load(f)

    if opt == "energy":
        data_doc["mcmc_en_list"] = best_value_list
    elif opt == "utilization":
        data_doc["mcmc_ut_list"] = best_value_list
    elif opt == "pareto":
        data_doc["mcmc_pareto_list"] = best_value_list
        data_doc["mcmc_pareto_en_list"] = pareto_en_list
        data_doc["mcmc_pareto_ut_list"] = pareto_ut_list

    data_doc["mcmc_exec_time_list"] = exec_time_list
    
    with open("temporal_mapping_optimizer/plots_data/visualisation_data.yaml", "w") as f:
        yaml.dump(data_doc, f)
    