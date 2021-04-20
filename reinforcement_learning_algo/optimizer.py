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


def rl_temporal_mapping_optimizer(temporal_mapping_ordering, layer, im2col_layer, layer_rounded,
                                  spatial_loop_comb, input_settings, mem_scheme, ii_su):
    print("--------- Reinforcement Learning Temporal Mapping Optimization ---------")
    temporal_mapping_pf_ordering = TemporalMappingState(temporal_mapping_ordering,
                                                        layer_architecture=layer.size_list_output_print)
    observation_state_length = 22
    action_state_length = int((observation_state_length * (observation_state_length - 1)) / 2)
    neural_network = MLP(observation_state_length, action_state_length)

    print(temporal_mapping_baseline_performances(temporal_mapping_pf_ordering.value, 50, layer, im2col_layer,
                                                 layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su))

    policy_gradient = PolicyGradient(
        neural_network, temporal_mapping_pf_ordering, layer,
        im2col_layer, layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su)

    policy_gradient.training(learning_rate=1e-2, reward_stop_condition=0.2 , gamma=0.9, log_interval=1,
                             observation_state_length=observation_state_length, episode_utilization_stop_condition=1,
                             timestamp_number=25, render=False)
    # policy_gradient.run_episode(starting_temporal_mapping=temporal_mapping_pf_ordering, episode_max_step=30)
