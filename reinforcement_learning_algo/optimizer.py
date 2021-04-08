import random

import numpy as np
from sympy.ntheory import factorint

from reinforcement_learning_algo import loop_type_to_ids
from reinforcement_learning_algo.cost_esimator import (
    get_temporal_loop_estimation,
    calculate_mac_level_costs,
)
from reinforcement_learning_algo.policy_networks.mlp import (
    MLP,
)
from reinforcement_learning_algo.reinforce import (
    PolicyGradient,
)


def form_temporal_mapping_ordering(layer):
    # Extract the naive TM from the layer architecture contained in layer
    temporal_mapping_ordering = []
    for loop_type, loop_id in loop_type_to_ids.items():
        temporal_mapping_ordering.append((loop_id, layer[loop_type]))
    return temporal_mapping_ordering


def pf_to_compressed_mapping(pf_temporal_mapping):
    compressed_temporal_mapping = [pf_temporal_mapping[0]]
    for i in range(1, len(pf_temporal_mapping)):
        if pf_temporal_mapping[i][0] == compressed_temporal_mapping[-1][0]:
            compressed_temporal_mapping[-1] = (
                pf_temporal_mapping[i][0],
                pf_temporal_mapping[i][1] * compressed_temporal_mapping[-1][1],
            )
        else:
            compressed_temporal_mapping.append(pf_temporal_mapping[i])
    return compressed_temporal_mapping


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
    # Shuffle the LPF_TM to get a random initialization into the design space
    random.shuffle(tm_primary_factors)
    return tm_primary_factors


def initialize_temporal_mapping(temporal_mapping_ordering) -> list:
    temporal_mapping_pm_ordering = find_tm_primary_factors(temporal_mapping_ordering)
    temporal_mapping_pm_ordering = randomize_temporal_mapping(temporal_mapping_pm_ordering)
    return temporal_mapping_pm_ordering


def temporal_mapping_baseline_performances(temporal_mapping_ordering, max_step, layer, im2col_layer,
                                           layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su):
    """
    Description:
      Evaluate the baseline performances (energy and utilization) of a given temporal mapping by generating n = max_step
      random mapping and averaging the result
    """
    energy_list = []
    utilization_list = []

    if temporal_mapping_ordering is None:
        temporal_mapping_ordering = form_temporal_mapping_ordering(layer.size_list_output_print)

    temporal_mapping_pf_ordering = initialize_temporal_mapping(temporal_mapping_ordering)

    for i in range(max_step):
        temporal_mapping_pf_ordering = randomize_temporal_mapping(temporal_mapping_pf_ordering)
        mac_costs = calculate_mac_level_costs(
            layer, layer_rounded, input_settings, mem_scheme, ii_su)
        energy, utilization = get_temporal_loop_estimation(
            temporal_mapping_pf_ordering,
            input_settings,
            spatial_loop_comb,
            mem_scheme,
            layer=[im2col_layer, layer_rounded],
            mac_costs=mac_costs,
        )

        energy_list.append(energy)
        utilization_list.append(utilization)

    return np.mean(energy_list) / (10 ** 12), np.mean(utilization_list)


def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su):
    print("--------- Reinforcement Learning Temporal Mapping Optimization ---------")
    layer_architecture = layer.size_list_output_print
    if temporal_mapping_ordering is None:
        temporal_mapping_ordering = form_temporal_mapping_ordering(layer_architecture)
    temporal_mapping_pf_ordering = initialize_temporal_mapping(temporal_mapping_ordering)
    observation_state_length = 22
    action_state_length = int((observation_state_length * (observation_state_length - 1)) / 2)
    neural_network = MLP(observation_state_length, action_state_length)

    print(temporal_mapping_baseline_performances(temporal_mapping_ordering, 50, layer, im2col_layer,
                                                 layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su))

    policy_gradient = PolicyGradient(
        neural_network, temporal_mapping_pf_ordering, layer,
        im2col_layer, layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su)

    policy_gradient.training(learning_rate=1e-2, reward_stop_condition=0.5, gamma=0.9, log_interval=1,
                             observation_state_length=observation_state_length, episode_utilization_stop_condition=0.8,
                             timestamp_number=100)
