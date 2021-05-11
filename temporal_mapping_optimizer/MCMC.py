from temporal_mapping_optimizer.cost_esimator import *
from temporal_mapping_optimizer import loop_type_to_ids

from loma import limit_lpf, get_prime_factors

from sympy import factorint, isprime
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import random
import time


def tmo_swap(tmo, i, j):
     # Operate a swap between i and j on a given tmo
     swapped_tmo = deepcopy(tmo)
     temp = swapped_tmo[i]
     swapped_tmo[i] = swapped_tmo[j]
     swapped_tmo[j] = temp

     return swapped_tmo

def evaluate_tmo(tmo, input_settings, spatial_loop_comb, mem_scheme, layer, mac_costs):
     return get_temporal_loop_estimation(tmo, input_settings, spatial_loop_comb, mem_scheme, layer, mac_costs)

def form_tmo(layer_architecture, spatial_unrolling):

        layer_spec_temporal = {}
        ids_to_loop_type = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}

        # Extract the naive TM from the layer architecture contained in layer_post
        tmo = []
        for loop_type in ['B','K','C','OY','OX','FY','FX']:
            layer_spec_temporal[loop_type] = layer_architecture[loop_type]   

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

def get_lpf_limited_tmo(layer_architecture, spatial_unrolling, limit_lpf):

     lpf_tmo = []
     layer_spec_temporal = form_tmo(layer_architecture, spatial_unrolling)
     layer_spec_pf, layer_spec_pf_count, total_lpf_count = get_prime_factors(layer_spec_temporal, limit_lpf)

     for loop_type in list(layer_spec_pf.keys()):
          for i in range(len(layer_spec_pf[loop_type])):
               loop_size = layer_spec_pf[loop_type]
               for number_of_loop in range(layer_spec_pf_count[loop_type][i]):
                    lpf_tmo.append((loop_type_to_ids[loop_type], loop_size[i]))

     return lpf_tmo

def get_min_lpf_size(layer_architecture, spatial_unrolling):

     tmo = []
     layer_spec_temporal = form_tmo(layer_architecture, spatial_unrolling)

     for loop_id, loop_size in list(layer_spec_temporal.items()):
            if loop_size != 1:
                tmo.append((loop_id, loop_size))

     return len(tmo)

def get_max_lpf_size(layer_architecture, spatial_unrolling):

     tmo = []
     layer_spec_temporal = form_tmo(layer_architecture, spatial_unrolling)

     for loop_id, loop_size in list(layer_spec_temporal.items()):
            if loop_size != 1:
                tmo.append((loop_id, loop_size))

     # Break it down to LPF (Loop Prime Factor)
     tmo_pf = []
     for inner_loop in tmo:
          if inner_loop[1] == 1:
               tmo_pf.append(inner_loop)
          else:
               factors = factorint(inner_loop[1])
               for factor in factors.items():
                    for pow in range(factor[1]):
                         tmo_pf.append((inner_loop[0], factor[0]))
     
     return len(tmo_pf)
    

def mcmc(temporal_mapping_ordering, iter, layer, im2col_layer, layer_rounded,
         spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling, plot=False):

     start_time = time.time()

     # Hyperparameters
     #iter = 2000
     temperature = 0.05
     rho = 0.999

     accepted_p_list = []
     accepted_utilization_list = []
     explotation_counter = 0
     exploration_swap_array = np.zeros((len(temporal_mapping_ordering), len(temporal_mapping_ordering)), dtype=float)
     explotation_swap_array = np.zeros((len(temporal_mapping_ordering), len(temporal_mapping_ordering)), dtype=float)

     # Initialize mac costs
     mac_costs = calculate_mac_level_costs(layer, layer_rounded, input_settings, mem_scheme, ii_su)
     # Extract the list of tuple from the tmo
     curr_tmo = temporal_mapping_ordering
     # Initalization of a random starting point
     random.shuffle(curr_tmo)

     curr_energy, curr_utilization = evaluate_tmo(curr_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)
     best_tmo = curr_tmo
     best_utilization = curr_utilization   

     test_tmo = [(5, 8), (6, 192), (1, 3), (5, 16), (3, 13)]
     [(3, 13), (6, 16), (5, 128), (1, 3), (6, 12)]
     test_energy, test_utilization = evaluate_tmo(test_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)
     #print("TEST TMO :", test_utilization)

     for k in range(iter):
          i = np.random.randint(0, len(curr_tmo)) 
          #range(0, len(curr_tmo)) -> delete(i)
          j = np.random.randint(0, len(curr_tmo))
          temp_tmo = tmo_swap(curr_tmo, i, j)
          temp_energy, temp_utilization = evaluate_tmo(temp_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)

          x = np.random.rand()
          p = np.exp((temp_utilization - curr_utilization) / temperature)
          temperature = temperature * rho

          if(x < p):        
               curr_tmo = temp_tmo.copy()
               curr_utilization = temp_utilization
               explotation_counter += 1
               explotation_swap_array[i, j] += 1

               accepted_utilization_list.append(curr_utilization)
               if p <= 1:
                    accepted_p_list.append(p)
               
               if(curr_utilization > best_utilization):
                    best_tmo = curr_tmo
                    best_utilization = curr_utilization
          else:
               exploration_swap_array[i, j] += 1         
          

     end_time = time.time()
     exec_time = end_time - start_time

     #print("Best utilization :", best_utilization)
     #print("On ", iter, "iterations :", explotation_counter, "explotation and", 2000 - explotation_counter, "exploration")

     if plot:
          plt.figure(1)
          plt.title('Utilization of accepted state evolution during the run')
          plt.xlabel("Iteration")
          plt.ylabel("Utilization")
          plt.plot([*range(len(accepted_utilization_list))], accepted_utilization_list)
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
          plt.show()

     return best_utilization, best_tmo, exec_time