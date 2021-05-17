from temporal_mapping_optimizer.cost_esimator import *
from temporal_mapping_optimizer import loop_type_to_ids
from temporal_mapping_optimizer.queue import Queue

from loma import limit_lpf, get_prime_factors
import classes as cls

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
         spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling, layer_post, opt, plot=False):

     """
     print("Testing the queue !")

     q = Queue(max_size=16)
     c_loop = ('C', 2)
     k_loop = ('K', 2)

     q.enqueue(c_loop)
     q.print_items()
     q.enqueue(k_loop)
     q.enqueue(c_loop)
     q.print_items()
     out = q.enqueue(c_loop)
     q.print_items()
     print("out :", out)
     out = q.enqueue(('OY',13))
     q.print_items()
     print("out :", out)
     """

     # Where can I update the SU ?
     # print(spatial_unrolling) # we dont use spatial_unrolling obj in cost eval ? 
     # So just remove loops from tmo should impact performances, or mapping.yaml
     # give the SU via another obj to cost eval. 
     # print(input_settings.spatial_unrolling_single) # maybe given with this arg
     # Have to define mem level to access the su
     # First try even su
     # Look like changing input_settings.spatial_unrolling_single doesn't change performances
     # Probably need to change spatial loop comb and mem scheme, I'll ask Arne on Monday

     # Seems to work for energy but not for utilization
     # Updating spatial_loop_comb obj and input_settings don't seems to have impact
     
     print(mem_scheme.spatial_unrolling)

     start_time = time.time()

     ### Hyperparameters ###
     temperature = 0.05*10
     rho = 0.999 # Temperature Decay

     # Plot lists
     accepted_p_list = []
     accepted_value_list = []
     explotation_counter = 0
     exploration_swap_array = np.zeros((len(temporal_mapping_ordering), len(temporal_mapping_ordering) + 1), dtype=float)
     explotation_swap_array = np.zeros((len(temporal_mapping_ordering), len(temporal_mapping_ordering) + 1), dtype=float)

     # Initialize mac costs
     mac_costs = calculate_mac_level_costs(layer, layer_rounded, input_settings, mem_scheme, ii_su)
     # Extract the list of tuple from the tmo
     start_tmo = temporal_mapping_ordering

     # Initalization and Evaluation of a random starting point
     random.shuffle(start_tmo)
     start_energy, start_utilization = evaluate_tmo(start_tmo, input_settings, spatial_loop_comb, mem_scheme, [im2col_layer, layer_rounded], mac_costs)

     if opt == "energy":
          best_value = start_energy.item()
          old_value = start_energy.item()
     elif opt == "utilization":
          best_value = start_utilization
          old_value = start_utilization
     elif opt == "pareto":
          best_value = start_energy/start_utilization
          old_value = start_energy/start_utilization
     best_tmo = start_tmo
     old_tmo = start_tmo

     su_max_size = 8
     old_su = Queue(su_max_size)

     new_input_settings = deepcopy(input_settings)
     new_spatial_loop_comb = deepcopy(spatial_loop_comb)
     new_mem_scheme = deepcopy(mem_scheme)
     
     best_su = old_su
     best_input_settings = input_settings
     best_spatial_loop_comb = spatial_loop_comb
     best_mem_scheme = mem_scheme

     for k in range(iter):
          
          # Note : the tmo size is dynamic here depending on how many loops are in the su queue
          su_idx = len(old_tmo)

          # Uniforme random sampling in the neighborhoods
          i = np.random.randint(0, len(old_tmo)) 
          j = np.random.randint(0, len(old_tmo) + 1)

          if j == su_idx:
               # Put the loop at pos i into the su queue and put queue output into the tmo
               if old_tmo[i][1] > su_max_size:
                    continue
               new_tmo = deepcopy(old_tmo)
               new_su = deepcopy(old_su)
               q_output = new_su.enqueue(new_tmo[i])
               new_tmo.pop(i)
               for loop in q_output:
                    new_tmo.insert(i, loop)

               # Update SU object 
               new_input_settings = deepcopy(input_settings) # save it in case we don't use this new su
               new_input_settings.spatial_unrolling_single['W'][1] = new_su.items
               new_input_settings.spatial_unrolling_single['I'][1] = new_su.items
               new_input_settings.spatial_unrolling_single['O'][1] = new_su.items

               new_spatial_loop_comb = deepcopy(spatial_loop_comb)
               [spatial_loop, spatial_loop_fractional] = new_spatial_loop_comb
               spatial_loop = cls.SpatialLoop.extract_loop_info(new_input_settings.spatial_unrolling_single, layer_post)
               new_spatial_loop_comb = [spatial_loop, spatial_loop_fractional]

               new_mem_scheme = deepcopy(mem_scheme)
               new_mem_scheme.spatial_unrolling = new_input_settings.spatial_unrolling_single

          else:
               # Apply the selected swap
               new_su = old_su
               new_input_settings = input_settings
               new_tmo = tmo_swap(old_tmo, i, j)

          # Evaluate the quality of the new tmo + su
          new_energy, new_utilization = evaluate_tmo(new_tmo, new_input_settings, new_spatial_loop_comb, new_mem_scheme, [im2col_layer, layer_rounded], mac_costs)

          # Compute the acceptance probability p of the new tmo
          if opt == "energy":
               new_value = new_energy.item()
               p = np.exp(((old_value / new_value) - 1) / temperature)
          elif opt == "utilization":
               new_value = new_utilization
               p = np.exp((new_value - old_value) / temperature)
          elif opt == "pareto":
               new_value = new_energy.item()/new_utilization
               p = np.exp(((old_value / new_value) - 1) / temperature)

          # Sample x to make the choice and update temperature
          x = np.random.rand() # x belongs to [0, 1]
          temperature = temperature * rho

          if(x < p):    
               # Move on the next point    
               old_tmo = new_tmo.copy()
               old_su = deepcopy(new_su)
               input_settings = new_input_settings
               spatial_loop_comb = new_spatial_loop_comb
               mem_scheme = new_mem_scheme
               old_value = new_value

               #print("New SU :", old_su.items)

               # Plot data saving
               explotation_counter += 1
               explotation_swap_array[i, j] += 1
               accepted_value_list.append(old_value)
               if p <= 1:
                    accepted_p_list.append(p)
               
               # We want to maximize utilization, minimize energy and pareto_score
               if ((opt == "energy" or opt == "pareto") and old_value < best_value) or (opt == "utilization" and old_value > best_value):
                    best_tmo = old_tmo
                    best_su = old_su
                    best_input_settings = input_settings
                    best_spatial_loop_comb = spatial_loop_comb
                    best_mem_scheme = mem_scheme
                    best_value = old_value
          else:
               exploration_swap_array[i, j] += 1
          
     # Save the exec time for plots
     end_time = time.time()
     exec_time = end_time - start_time

     #print("On ", iter, "iterations :", explotation_counter, "explotation and", 2000 - explotation_counter, "exploration")
     print("Best value :", best_value)
     print("Best_su :", best_su.items)
     print("Best input settings su :", best_input_settings.spatial_unrolling_single)

     # Visualization option
     if plot:
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
          plt.show()

     return best_value, best_tmo, best_su, exec_time