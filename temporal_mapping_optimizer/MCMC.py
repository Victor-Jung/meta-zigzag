from temporal_mapping_optimizer.cost_esimator import *
from temporal_mapping_optimizer import loop_type_to_ids, ids_to_loop_type
from temporal_mapping_optimizer.queue import Queue

from bsgutils import utilization_rate_optimizer
from loma import limit_lpf, get_prime_factors
import cost_model_funcs as cmf
import classes as cls
import msg

from multiprocessing import Process, Queue
from sympy import factorint, isprime
import matplotlib.pyplot as plt
from copy import deepcopy
from pprint import pprint
import numpy as np
import random
import time
import yaml


def tmo_swap(tmo, i, j):
     # Operate a swap between i and j on a given tmo
     swapped_tmo = deepcopy(tmo)
     temp = swapped_tmo[i]
     swapped_tmo[i] = swapped_tmo[j]
     swapped_tmo[j] = temp

     return swapped_tmo

def evaluate_tmo(tmo, input_settings, spatial_loop_comb, mem_scheme, layer, mac_costs):
     return get_temporal_loop_estimation(tmo, input_settings, spatial_loop_comb, mem_scheme, layer, mac_costs)

def form_tmo(layer_post, spatial_unrolling):

     layer_spec_temporal = {}
     ids_to_loop_type = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}

     # Extract the naive TM from the layer architecture contained in layer_post
     for loop_type, loop_factor in layer_post.items():
        if (loop_factor != 0 or loop_factor != 1) and (loop_type in ['B','K','C','OY','OX','FY','FX']):
            layer_spec_temporal[loop_type] = loop_factor

     # Update the temporal layer spec to remove the already spatially unrolled dimensions.
     for level in range(0, len(spatial_unrolling['W'])):
          for [loop_type_number, su_factor] in spatial_unrolling['W'][level]:
               loop_type = ids_to_loop_type[loop_type_number]
               try:
                    pf = layer_spec_temporal[loop_type]
               except:
                    continue
                q, rem = divmod(pf, su_factor)
                assert rem == 0 # pf/su_factor should have remainder 0
                layer_spec_temporal[loop_type] = q
        
        # Then filter the 1-size loops
        for loop_type, loop_size in list(layer_spec_temporal.items()):
            if loop_size == 1:
                layer_spec_temporal.pop(loop_type)
        
        return layer_spec_temporal

def get_lpf_tmo(layer_architecture, spatial_unrolling):

     lpf_tmo = []
     layer_spec_temporal = form_tmo(layer_architecture, spatial_unrolling)
     layer_spec_pf, layer_spec_pf_count, total_lpf_count = get_prime_factors(layer_spec_temporal)

     for loop_type in list(layer_spec_pf.keys()):
          for i in range(len(layer_spec_pf[loop_type])):
               loop_size = layer_spec_pf[loop_type]
               for number_of_loop in range(layer_spec_pf_count[loop_type][i]):
                    lpf_tmo.append((loop_type_to_ids[loop_type], loop_size[i]))

     return lpf_tmo

def get_prime_factors(layer_spec):

    layer_spec_pf = {}
    layer_spec_pf_count = {}
    layer_spec_pf_count_sum = {}

    for loop_type, loop_dimension in layer_spec.items():
        if loop_dimension == 0 or loop_dimension == 1:
            continue
        factors = factorint(loop_dimension)
        pfs = []
        counts = []
        for pf, count in factors.items():
            pfs.append(pf)
            counts.append(count)
        layer_spec_pf[loop_type] = tuple(pfs)
        layer_spec_pf_count[loop_type] =  tuple(counts)
        layer_spec_pf_count_sum[loop_type] = sum(counts)
    
    total_lpf_count = sum(layer_spec_pf_count_sum.values())

    #layer_spec_pf, layer_spec_pf_count, total_lpf_count = limit_lpf(layer_spec_pf, layer_spec_pf_count, layer_spec_pf_count_sum, lpf_limit)

    return layer_spec_pf, layer_spec_pf_count, total_lpf_count
    

def mcmc(temporal_mapping_ordering, iter, layer, im2col_layer, layer_rounded,
         spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling, layer_post, opt, plot=False):

     start_time = time.time()

     ### Hyperparameters ###
     max_temperature = 0.05
     min_temperature = max_temperature*(0.999**2000)
     temperature_linspace = np.flip(np.linspace(min_temperature, max_temperature, iter)) # Our cooling schedule

     # Plot lists
     accepted_p_list = []
     accepted_value_list = []
     value_list = []
     explotation_counter = 0
     exploration_counter = 0
     rejection_counter = 0
     exploration_swap_array = np.zeros((len(temporal_mapping_ordering), len(temporal_mapping_ordering)), dtype=float)
     explotation_swap_array = np.zeros((len(temporal_mapping_ordering), len(temporal_mapping_ordering)), dtype=float)

     # Initialize mac costs
     mac_costs = calculate_mac_level_costs(layer, layer_rounded, input_settings, mem_scheme, ii_su)
     # Extract the list of tuple from the tmo
     start_tmo = temporal_mapping_ordering

     start_energy, start_utilization, start_latency = evaluate_tmo(start_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)

     if opt == "energy":
          best_value = start_energy.item()
          old_value = start_energy.item()
     elif opt == "latency":
          best_value = start_latency
          old_value = start_latency
          best_ut = start_utilization
     elif opt == "pareto":
          best_value = start_energy/start_utilization
          old_value = start_energy/start_utilization
     best_tmo = start_tmo
     old_tmo = start_tmo

     su_max_size = 256
     su_action_count = 0

     # Init the Su Queue with the starting SU
     old_su = Queue(su_max_size)
     for loop in input_settings.spatial_unrolling_single['W'][1]:
          old_su.enqueue(loop)

     new_input_settings = deepcopy(input_settings)
     new_spatial_loop_comb = deepcopy(spatial_loop_comb)
     new_mem_scheme = deepcopy(mem_scheme)
     new_mac_costs = mac_costs
     
     best_su = old_su
     best_input_settings = input_settings
     best_spatial_loop_comb = spatial_loop_comb
     best_mem_scheme = mem_scheme

     # Check if the starting tmo is empty (means that all loops were spatially unrolled and we evaluate the cost model as such)
     if start_tmo == []:
          
          energy, utilization, latency = evaluate_tmo(start_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)
          exec_time = time.time() - start_time

          if opt == "energy":
               best_value = energy.item()
               results_queue.put([best_value, best_tmo, exec_time, opt])
          elif opt == "latency":
               best_value = latency
               results_queue.put([best_value, best_ut, best_tmo, exec_time, opt])
          elif opt == "pareto":
               best_value = energy.item()/utilization
          
          return best_value, start_tmo, exec_time

     for temperature in temperature_linspace:
          
          # Note : the tmo size is dynamic here depending on how many loops are in the su queue
          su_idx = len(old_tmo)

          # Uniforme random sampling in the neighborhoods
          i = np.random.randint(0, len(old_tmo)) 
          j = np.random.randint(0, len(old_tmo) + 1)

          if j == su_idx:
               
               su_action_count += 1

               # Put the loop at pos i into the su queue and put queue output into the tmo
               if old_tmo[i][1] > su_max_size:
                    continue
               new_tmo = deepcopy(old_tmo)
               new_su = deepcopy(old_su)
               q_output = new_su.enqueue(new_tmo[i])
               new_tmo.pop(i)
               for loop in q_output:
                    new_tmo.insert(i, loop)

               # Split our Su in two list where the product of pf is <= 16 (Row and Col)
               row_list = []
               col_list = []
               row_pf_product = 1
               col_pf_product = 1
               for loop in new_su.items:
                    if (row_pf_product * loop[1]) <= 16:
                         row_pf_product *= loop[1]
                         row_list.append([ids_to_loop_type[loop[0]], loop[1]])
                    else:
                         col_pf_product *= loop[1]
                         col_list.append([ids_to_loop_type[loop[0]], loop[1]])

               # Flooring Generation
               sm_fixed = {'W': [], 'I': [], 'O': []}
               flooring_fixed = {'W': [], 'I': [], 'O': []}
               i2a = {'B': 7, 'K': 6, 'C': 5, 'OY': 4, 'OX': 3, 'FY': 2, 'FX': 1}
               with open("./inputs/mapping.yaml") as f:
                    fl = yaml.full_load(f)
               fl['spatial_mapping_fixed']['weight'][0]['Col'] = col_list
               fl['spatial_mapping_fixed']['weight'][0]['Row'] = row_list
               fl['spatial_mapping_fixed']['input'][0]['Col'] = col_list
               fl['spatial_mapping_fixed']['input'][0]['Row'] = row_list
               fl['spatial_mapping_fixed']['output'][0]['Col'] = col_list
               fl['spatial_mapping_fixed']['output'][0]['Row'] = row_list

               for op in fl['spatial_mapping_fixed']:
                    if op == 'weight': operand = 'W'
                    elif op == 'input': operand = 'I'
                    elif op == 'output': operand = 'O'
                    sm_fixed[operand] = [[] for x in fl['spatial_mapping_fixed'][op]]
                    flooring_fixed[operand] = [[] for x in fl['spatial_mapping_fixed'][op]]
                    for lev in fl['spatial_mapping_fixed'][op]:
                         ii_lev = 0
                         if lev == 'MAC' : ii_lev = 0
                         else : ii_lev = lev + 1
                         flooring_fixed[operand][ii_lev] = [[] for d in fl['spatial_mapping_fixed'][op][lev]]
                         for dim in fl['spatial_mapping_fixed'][op][lev]:
                              ii_dim = 0
                              if dim == 'Col': ii_dim = 0
                              elif dim == 'Row': ii_dim = 1
                              for pf in fl['spatial_mapping_fixed'][op][lev][dim]:
                                   sm_fixed[operand][ii_lev].append(tuple([i2a[pf[0]], pf[1]]))
                                   flooring_fixed[operand][ii_lev][ii_dim].append(i2a[pf[0]])
               
               mem_unroll_active, mem_unroll_total = cmf.get_mem_complete_unrolling_count(
                         sm_fixed, flooring_fixed, input_settings.mac_array_info['array_size'])

               # Init of New Obj
               new_mem_scheme = deepcopy(mem_scheme)
               new_input_settings = deepcopy(input_settings)

               new_input_settings.spatial_unrolling_single['W'][1] = [[loop[0], loop[1]] for loop in new_su.items]
               new_input_settings.spatial_unrolling_single['I'][1] = [[loop[0], loop[1]] for loop in new_su.items]
               new_input_settings.spatial_unrolling_single['O'][1] = [[loop[0], loop[1]] for loop in new_su.items]
               new_input_settings.flooring_single = flooring_fixed

               new_mem_scheme.spatial_unrolling = [new_input_settings.spatial_unrolling_single]
               new_mem_scheme.flooring = [new_input_settings.flooring_single]
               new_mem_scheme.mem_unroll_complete = {'mem_unroll_active': mem_unroll_active, 'mem_unroll_total': mem_unroll_total}

               new_spatial_unrolling = [new_input_settings.spatial_unrolling_single]

               new_spatial_loop = cls.SpatialLoop.extract_loop_info(new_mem_scheme.spatial_unrolling[ii_su], layer_post)
               new_spatial_loop_comb = [new_spatial_loop, new_spatial_loop]

               new_mac_costs = calculate_mac_level_costs(layer, layer_rounded, new_input_settings, new_mem_scheme, ii_su)

               new_mem_scheme.mem_utilization_rate, good_scheme = utilization_rate_optimizer(new_mem_scheme.mem_size,
                                                                                               new_mem_scheme.spatial_unrolling[ii_su],
                                                                                               layer_post,
                                                                                               new_input_settings.precision,
                                                                                               new_mem_scheme.mem_utilization_rate,
                                                                                               new_spatial_loop.unit_unique)
               

          else: 
               # Apply the selected swap
               new_su = old_su
               new_input_settings = input_settings
               new_mac_costs = mac_costs
               new_spatial_loop_comb = spatial_loop_comb
               new_mem_scheme = mem_scheme
               new_tmo = tmo_swap(old_tmo, i, j)

          # Evaluate the quality of the new tmo + su
          new_energy, new_utilization = evaluate_tmo(new_tmo, new_input_settings, new_spatial_loop_comb, new_mem_scheme, [im2col_layer, layer_rounded], new_mac_costs)

          # Compute the acceptance probability p of the new tmo
          if opt == "energy":
               new_value = new_energy.item()
               p = np.exp(((old_value / new_value) - 1) / temperature)
          elif opt == "latency":
               p = np.exp(((old_value / new_value) - 1) / temperature)
          elif opt == "pareto":
               new_value = new_energy.item()/new_utilization
               p = np.exp(((old_value / new_value) - 1) / temperature)

          # Sample x to make the choice and update temperature
          x = np.random.rand() # x belongs to [0, 1]

          if(x < p):    
               # Move on the next point    
               old_tmo = deepcopy(new_tmo)
               old_su = deepcopy(new_su)
               input_settings = new_input_settings
               spatial_loop_comb = new_spatial_loop_comb
               mem_scheme = new_mem_scheme
               mac_costs = new_mac_costs
               old_value = new_value

               if p >= 1:
                    explotation_counter += 1
               else:
                    exploration_counter += 1
               
               # We want to maximize utilization, minimize energy and pareto_score
               if old_value < best_value:
                    best_tmo = old_tmo
                    best_su = old_su
                    best_input_settings = input_settings
                    best_spatial_loop_comb = spatial_loop_comb
                    best_mem_scheme = mem_scheme
                    best_mac_costs = mac_costs
                    best_value = old_value
                    best_ut = new_utilization
          else:
               rejection_counter += 1

          value_list.append(new_utilization)

     end_time = time.time()
     exec_time = end_time - start_time

     if verbose == 1:
          print("On ", iter, "iterations :", explotation_counter, "explotation and", exploration_counter, "exploration")
     
     if plot:
          plt.figure(1)
          plt.hist(value_list)
          plt.show()
          '''
          plt.figure(1)
          plt.title('Utilization of accepted state evolution during the run')
          plt.xlabel("Iteration")
          plt.ylabel("Utilization")
          plt.plot([*range(len(accepted_value_list))], accepted_value_list)
          plt.figure(2)
          plt.title('Alpha evolution during the run')
          plt.xlabel("Temporal Mapping Size")
          plt.ylabel("Alpha")
          plt.plot([*range(len(accepted_p_list))], accepted_p_list, "o")
          plt.figure(3)
          plt.title('Heatmap of Explotation Swap(i, j)')
          plt.xlabel("i")
          plt.ylabel("j")
          plt.imshow(explotation_swap_array, cmap='hot', interpolation='nearest')
          plt.figure(4)
          plt.title('Heatmap of Exploration Swap(i, j)')
          plt.xlabel("i")
          plt.ylabel("j")
          plt.imshow(exploration_swap_array, cmap='hot', interpolation='nearest')
          plt.show()'''

     if opt == 'latency':
          results_queue.put([best_value, best_ut, best_tmo, exec_time, opt])
     else:
          results_queue.put([best_value, best_tmo, exec_time, opt])
          
     return best_value, best_tmo, exec_time
