from multiprocessing import Process, Queue, cpu_count
import matplotlib.pyplot as plt
import numpy as np
import time
import yaml
import math

#from reinforcement_learning_algo.core.state import TemporalMappingState
from temporal_mapping_optimizer.MCMC import *
from temporal_mapping_optimizer.random_search import *
from temporal_mapping_optimizer.cost_esimator import *
from temporal_mapping_optimizer import loop_type_to_ids, ids_to_loop_type

def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer_post, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):

    print("--------- Simulated Annealing Monte Carlo Markov Chain (SA-MCMC) Temporal Mapping Optimization ---------")

    opt = "mixed"
    number_of_thread = 8

    if opt == "energy":
        best_en, best_en_tmo, best_ut, best_ut_tmo, exec_time = optimize("energy", number_of_thread, temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)

    elif opt == "latency":
        best_en, best_en_tmo, best_ut, best_ut_tmo, exec_time = optimize("latency", number_of_thread, temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
    
    elif opt == "mixed":
        best_en, best_en_tmo, best_en_su, best_en_order, best_lat, best_ut, best_lat_tmo, best_lat_su, best_lat_order, exec_time = optimize("mixed", number_of_thread, temporal_mapping_ordering, layer_post, 
                                layer, im2col_layer, layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)

    elif opt == "pareto":
        optimize("energy", number_of_thread, temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
        optimize("utilization", number_of_thread, temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
        optimize("pareto", number_of_thread, temporal_mapping_ordering, layer_post, layer, im2col_layer, 
                layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)
    
    return best_en, best_en_tmo, best_en_su, best_en_order, best_lat, best_ut, best_lat_tmo, best_lat_su, best_lat_order, exec_time


def optimize(opt, number_of_thread, temporal_mapping_ordering, layer_post, layer, im2col_layer, 
            layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):

    # Initialize mac costs
    mac_costs = calculate_mac_level_costs(layer, layer_rounded, input_settings, mem_scheme, ii_su)

    iter_number = 3000

    exec_time_list = []
    best_value_list = []
    pareto_en_list = []
    pareto_ut_list = []
    best_tmo = []
    best_su = []

    if opt == "energy" or opt == "pareto":
        best_value = float('inf')
    elif opt == "latency":
        best_value = float('inf')
        best_ut = 0
    elif opt == "mixed":
        best_en = float('inf')
        best_lat = float('inf')
        best_ut = 0
        # Split the processes in two for energy and utilization optimization
        max_core = min(number_of_thread, cpu_count())
        en_core = math.ceil(max_core/2)
        ut_core = max_core - en_core

    starting_tmo = get_lpf_tmo(layer_post, spatial_unrolling)
    worker_list = []
    results_queue = Queue()

    # Launch threads
    if opt == "mixed":
        for i in range(0, en_core):
            p = Process(target=mcmc, args=(starting_tmo, iter_number, layer, im2col_layer, layer_rounded, 
                        spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling, layer_post, results_queue, 'energy',  False))
            worker_list.append(p)
            p.start()
        for i in range(0, ut_core):
            p = Process(target=mcmc, args=(starting_tmo, iter_number, layer, im2col_layer, layer_rounded, 
                        spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling, layer_post, results_queue, 'latency', False))
            worker_list.append(p)
            p.start()
    else:
        for i in range(0, min(number_of_thread, cpu_count())):
            p = Process(target=mcmc, args=(starting_tmo, iter_number, layer, im2col_layer, layer_rounded, 
                        spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling, layer_post, results_queue, opt, False))
            worker_list.append(p)
            p.start()

    # Block while all process are not finished
    for worker in worker_list:
        worker.join()

    # Collect results from the queue and extract the best
    if opt == "mixed":
        for i in range(0, max_core):
            result = results_queue.get()
            if (result[-1] == 'energy' and result[0] < best_en):
                best_en = result[0]
                best_en_tmo = result[2]
                best_en_su = result[3]
                exec_time = result[4]
                best_en_order = result[5]
            if (result[-1] == 'latency' and result[0] < best_lat):
                best_lat = result[0]
                best_ut = result[1]
                best_lat_tmo = result[2]
                best_lat_su = result[3]
                exec_time = result[4]
                best_lat_order = result[5]
        
        print("Best Energy :", best_en)
        print("Best SU Energy :", best_en_su.items)
        print("Best TMO Energy :", best_en_tmo)
        print("Best Utilization :", best_ut)
        print("Best Su Utilization :", best_lat_su.items)
        print("Best TMO Utilization :", best_lat_tmo)
        print("Best Latency :", best_lat)
        print("Exec time", exec_time)

        return best_en, best_en_tmo, best_en_su, best_en_order, best_lat, best_ut, best_lat_tmo, best_lat_su, best_lat_order, exec_time

    else:
        for i in range(0, max_core):
            result = results_queue.get()
            if ((result[3] == "energy" or result[3] == "pareto") and result[0] < best_value) or (result[3] == "latency" and result[0] < best_value):
                best_value = result[0]
                best_tmo = result[1]
                best_exec_time = result[2]

        if opt == "pareto":
            pareto_en, pareto_ut = get_temporal_loop_estimation(best_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)
            pareto_en_list.append(pareto_en.item())
            pareto_ut_list.append(pareto_ut)
        
        if opt == "energy":
            print("Best Energy :", best_value)
        elif opt == "latency":
            print("Best Utilization :", best_value)
        elif opt == "pareto":
            print("Best Pareto Score :", best_value)

        print("Best tmo :", best_tmo)
        print("Exec time", best_exec_time)  

        return best_value, best_tmo, None, None, best_exec_time
