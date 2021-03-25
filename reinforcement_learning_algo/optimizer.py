import random

from sympy.ntheory import factorint

from reinforcement_learning_algo.cost_esimator import get_temporal_loop_estimation, calculate_mac_level_costs

loop_types_list = ["FX", "FY", "OX", "OY", "C", "K", "B"]
# Corresponding number for each loop_type {"FX": 1, "FY": 2, "OX": 3, "OY": 4, "C": 5, "K": 6, "B": 7}
loop_type_to_ids = {key: value + 1 for value, key in enumerate(loop_types_list)}
# Corresponding number for each loop_type: {1: "FX", 2: "FY", 3: "OX", 4: "OY", 5: "C", 6: "K", 7: "B"}
ids_to_loop_type = {value: key for key, value in loop_type_to_ids.items()}

operand_cost_types = ['W', 'I', 'O']
operand_cost_template = {key: [] for key in operand_cost_types}


def form_temporal_mapping_ordering(layer_post):
    # Extract the naive TM from the layer architecture contained in layer_post
    temporal_mapping_ordering = []
    for loop_type, loop_id in loop_type_to_ids.items():
        temporal_mapping_ordering.append((loop_id, layer_post[loop_type]))
    return temporal_mapping_ordering


def uneven_to_even_mapping(uneven_temporal_mapping):

    even_temporal_mapping = [uneven_temporal_mapping[0]]
    for i in range(1, len(uneven_temporal_mapping)):
        if uneven_temporal_mapping[i][0] == even_temporal_mapping[-1][0]:
            even_temporal_mapping[-1] = (uneven_temporal_mapping[i][0], 
                                        uneven_temporal_mapping[i][1] + even_temporal_mapping[-1][1])
        else:
            even_temporal_mapping.append(uneven_temporal_mapping[i])
    print(even_temporal_mapping)
    return even_temporal_mapping

def find_tm_primary_factors(temporal_mapping_ordering):
    # Break it down to LPF (Loop Prime Factor)
    temporal_mapping_primary_factors = []
    for inner_loop in temporal_mapping_ordering:
        if inner_loop[1] == 1:
            temporal_mapping_primary_factors.append(inner_loop)
        else:
            factors = factorint(inner_loop[1])
            for factor in factors.items():
                for pow in range(factor[1]):
                    temporal_mapping_primary_factors.append((inner_loop[0], factor[0]))
    return temporal_mapping_primary_factors


def randomize_temporal_mapping(tm_primary_factors):
    # Shuffle the LPF_TM to get a random initalization into the design space
    random.shuffle(tm_primary_factors)
    return tm_primary_factors


def initialize_temporal_mapping(temporal_mapping_ordering) -> list:
    temporal_mapping_pm_ordering = find_tm_primary_factors(temporal_mapping_ordering)
    temporal_mapping_pm_ordering = randomize_temporal_mapping(temporal_mapping_pm_ordering)
    print(temporal_mapping_ordering)
    print(temporal_mapping_pm_ordering)
    return temporal_mapping_pm_ordering


def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer_, layer_post, im2col_layer, layer_rounded, spatial_loop_comb,
                                  input_settings, mem_scheme, ii_su):
    print('--------- Reinforcement Learning Temporal Mapping Optimization ---------')
    if temporal_mapping_ordering is None:
        temporal_mapping_ordering = form_temporal_mapping_ordering(layer_post)
    uneven_temporal_mapping_ordering = initialize_temporal_mapping(temporal_mapping_ordering)
    even_temporal_mapping_ordering = uneven_to_even_mapping(uneven_temporal_mapping_ordering)

    layer = [im2col_layer, layer_rounded]
    mac_costs = calculate_mac_level_costs(layer_, layer_rounded, input_settings, mem_scheme, ii_su)
    energy, utilization = get_temporal_loop_estimation(uneven_temporal_mapping_ordering, input_settings, spatial_loop_comb,
                                                       mem_scheme, layer, mac_costs)
    print(f'Energy: {energy}')
    print(f'Utilization: {utilization}')
    return
