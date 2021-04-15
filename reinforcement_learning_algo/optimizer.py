import numpy as np
from copy import deepcopy
import random
import itertools

from reinforcement_learning_algo.core.state import TemporalMappingState
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


def temporal_mapping_baseline_performances(temporal_mapping_ordering, max_step, layer, im2col_layer,
                                           layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su):
    """
    Description:
      Evaluate the baseline performances (energy and utilization) of a given temporal mapping by generating n = max_step
      random mapping and averaging the result
    """
    energy_list = []
    utilization_list = []

    temporal_mapping_pf_ordering = TemporalMappingState(temporal_mapping_ordering,
                                                        layer_architecture=layer.size_list_output_print)
    for i in range(max_step):
        temporal_mapping_pf_ordering.randomize_temporal_mapping()
        mac_costs = calculate_mac_level_costs(
            layer, layer_rounded, input_settings, mem_scheme, ii_su)
        energy, utilization = get_temporal_loop_estimation(
            temporal_mapping_pf_ordering.value,
            input_settings,
            spatial_loop_comb,
            mem_scheme,
            layer=[im2col_layer, layer_rounded],
            mac_costs=mac_costs,
        )

        energy_list.append(energy)
        utilization_list.append(utilization)

    return np.mean(energy_list) / (10 ** 12), np.mean(utilization_list)

### Start of WIP ###

def tm_swap(temporal_mapping_ordering, idx1, idx2):

        tm = deepcopy(temporal_mapping_ordering)
        temp = tm[idx1]
        tm[idx1] = tm[idx2]
        tm[idx2] = temp
        return tm

def get_filtered_action_probs(tm, action_list, action_probs):

     # Filter useless swap
    loop_types = [loop_type for loop_type, weight in tm]
    for i, action in enumerate(action_list):
        pos1, pos2 = action
        if tm[pos1] == tm[pos2]:
            action_probs[i] = 0
        else:
            k = set(loop_types[pos1:pos2 + 1])
            if len(k) == 1:
                action_probs[i] = 0
        
    return action_probs

def get_action(tm):

    tm_size = len(tm)
    action_list = list(itertools.combinations(range(tm_size), 2))
    # Create uniform prob distribution for each action
    action_probs = [1/len(action_list)] * len(action_list)

    # And filter useless swap
    action_probs = get_filtered_action_probs(tm, action_list, action_probs)
    
    # Choose the action to perform randomly
    action_idx = random.choices(range(len(action_list)), action_probs, k=1)[0]

    return action_list[action_idx]

def get_average_neighborhood_utilization(temporal_mapping_ordering, layer, im2col_layer, layer_rounded,
                                         spatial_loop_comb, input_settings, mem_scheme, ii_su):

    tm = deepcopy(temporal_mapping_ordering)
    tm_size = len(tm)
    neighbors_list = []
    utilization_list = []
    action_list = list(itertools.combinations(range(tm_size), 2))
    # Create uniform prob distribution for each action
    action_probs = [1/len(action_list)] * len(action_list)

    # And filter useless swap
    action_probs = get_filtered_action_probs(tm, action_list, action_probs)

    for i in range(len(action_probs)):
        if action_probs[i] != 0:
            neighbors_list.append(tm_swap(tm, action_list[i][0], action_list[i][1]))
    
    # Compute current state utilization
    mac_costs = calculate_mac_level_costs(
        layer, layer_rounded, input_settings, mem_scheme, ii_su)
    current_tm_energy, current_tm_utilization = get_temporal_loop_estimation(
        tm,
        input_settings,
        spatial_loop_comb,
        mem_scheme,
        layer=[im2col_layer, layer_rounded],
        mac_costs=mac_costs,
    )

    # Compute neighborhood utilization
    for neighbor in neighbors_list:
        
        energy, utilization = get_temporal_loop_estimation(
            neighbor,
            input_settings,
            spatial_loop_comb,
            mem_scheme,
            layer=[im2col_layer, layer_rounded],
            mac_costs=mac_costs,
        )
        utilization_list.append(utilization)

    print("State utilization : ", current_tm_utilization, "\t Mean neighborhood utilization : ", np.mean(utilization_list),
    "\t Max neighborhood utiliz : ", np.max(utilization_list), "\t Min neighborhood utiliz : ", np.min(utilization_list))

### End of WIP ###

def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su):
    print("--------- Reinforcement Learning Temporal Mapping Optimization ---------")
    temporal_mapping_pf_ordering = TemporalMappingState(temporal_mapping_ordering,
                                                        layer_architecture=layer.size_list_output_print)
    observation_state_length = 22
    action_state_length = int((observation_state_length * (observation_state_length - 1)) / 2)

    tm = temporal_mapping_pf_ordering.value
    action_list = list(itertools.combinations(range(len(tm)), 2))
    
    
    for i in range(10):
        get_average_neighborhood_utilization(tm, layer, im2col_layer, layer_rounded,
                                         spatial_loop_comb, input_settings, mem_scheme, ii_su)
        action = get_action(tm)
        tm = tm_swap(tm, action[0], action[1])
    

    '''
    neural_network = MLP(observation_state_length, action_state_length)

    policy_gradient = PolicyGradient(
        neural_network, temporal_mapping_pf_ordering, layer,
        im2col_layer, layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su)

    policy_gradient.training(learning_rate=1e-2, reward_stop_condition=0.4, gamma=0.9, log_interval=1,
                             observation_state_length=observation_state_length, episode_utilization_stop_condition=0.8,
                             timestamp_number=21)
    '''