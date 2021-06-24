from temporal_mapping_optimizer.cost_esimator import *
from temporal_mapping_optimizer import loop_type_to_ids, ids_to_loop_type
from temporal_mapping_optimizer.queue import Spatial_Unrolling_Queue

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

def update_cost_obj(new_su, input_settings, mem_scheme, layer, layer_rounded, layer_post, ii_su):

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
    
    for operand in ['W','I','O']:
        for mem_idx, mem_level in enumerate(sm_fixed[operand]):
                for loop_idx, loop in enumerate(mem_level):
                    sm_fixed[operand][mem_idx][loop_idx] = list(loop)

    # Init of New Obj
    new_mem_scheme = deepcopy(mem_scheme)
    new_input_settings = deepcopy(input_settings)

    #ii_su = 0

    new_input_settings.spatial_unrolling_single['W'] = sm_fixed['W']
    new_input_settings.spatial_unrolling_single['I'] = sm_fixed['I']
    new_input_settings.spatial_unrolling_single['O'] = sm_fixed['O']
    new_input_settings.flooring_single = flooring_fixed

    #new_mem_scheme.spatial_unrolling = [new_input_settings.spatial_unrolling_single]
    #new_mem_scheme.flooring = [new_input_settings.flooring_single]

    new_mem_scheme.spatial_unrolling[ii_su] = new_input_settings.spatial_unrolling_single
    new_mem_scheme.flooring[ii_su] = new_input_settings.flooring_single
    new_mem_scheme.mem_unroll_complete = {'mem_unroll_active': mem_unroll_active, 'mem_unroll_total': mem_unroll_total}

    new_spatial_unrolling = [new_input_settings.spatial_unrolling_single]

    new_spatial_loop = cls.SpatialLoop.extract_loop_info(new_mem_scheme.spatial_unrolling[ii_su], layer_post)
    new_spatial_loop_fractional = cls.SpatialLoop.extract_loop_info(new_mem_scheme.fraction_spatial_unrolling[ii_su], layer_post)
    new_spatial_loop_comb = [new_spatial_loop, new_spatial_loop_fractional]

    new_mac_costs = calculate_mac_level_costs(layer, layer_rounded, new_input_settings, new_mem_scheme, ii_su)

    new_mem_scheme.mem_utilization_rate, good_scheme = utilization_rate_optimizer(new_mem_scheme.mem_size,
                                                                                    new_mem_scheme.spatial_unrolling[ii_su],
                                                                                    layer_post,
                                                                                    new_input_settings.precision,
                                                                                    new_mem_scheme.mem_utilization_rate,
                                                                                    new_spatial_loop.unit_unique)

    return new_input_settings, new_mem_scheme, new_mac_costs, new_spatial_loop_comb
