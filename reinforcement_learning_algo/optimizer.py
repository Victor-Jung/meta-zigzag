import numpy as np

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

def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling):

    print("--------- Reinforcement Learning Temporal Mapping Optimization ---------")

    temporal_mapping_pf_ordering = TemporalMappingState(spatial_unrolling, temporal_mapping_ordering,
                                                        layer_architecture=layer.size_list_output_print)

    print("Starting Ordering : ", temporal_mapping_pf_ordering.value)

    observation_state_length = len(temporal_mapping_pf_ordering.value)
    action_state_length = int((observation_state_length * (observation_state_length - 1)) / 2)
    neural_network = MLP(observation_state_length, action_state_length)

    policy_gradient = PolicyGradient(
        neural_network, temporal_mapping_pf_ordering, layer,
        im2col_layer, layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su, spatial_unrolling)

    policy_gradient.training(learning_rate=1e-2, reward_stop_condition=0.3, gamma=0.9, log_interval=1,
                             observation_state_length=observation_state_length, episode_utilization_stop_condition=0.8,
                             timestamp_number=10, render=True, save_weights=False)
