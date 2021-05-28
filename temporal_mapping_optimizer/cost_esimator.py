import cost_model_funcs as cmf
from loma_copy import (
    allocate_memory_for_tl_order,
    create_loop_objects,
    perform_greedy_mapping,
    get_utilization,
    find_total_cost_layer,
)


def calculate_mac_level_costs(layer_, layer_rounded, input_settings, mem_scheme, ii_su):
    active_mac_cost = cmf.get_active_mac_cost(layer_, input_settings.mac_array_info["single_mac_energy"])
    idle_mac_cost = cmf.get_idle_mac_cost(
        layer_,
        layer_rounded,
        input_settings.mac_array_info["array_size"],
        input_settings.mac_array_info["idle_mac_energy"],
        mem_scheme.spatial_unrolling,
    )[ii_su]
    mac_costs = [active_mac_cost, idle_mac_cost]
    return mac_costs


def get_temporal_loop_estimation(
    temporal_mapping_ordering,
    input_settings,
    spatial_loop_comb,
    mem_scheme,
    layer,
    mac_costs,
):
    """
    The function for calculating energy, cost for temporal mapping ordering.

    Parameters
    ----------
    temporal_mapping_ordering: a dict with the number of permutations (size) per each layer, ex.: ((layer_type, value))
    input_settings: InputSettings object
    spatial_loop_comb: a list of mem_scheme.spatial_unrolling, mem_scheme.fraction_spatial_unrolling SpatialLoop objects
    mem_scheme: MemorySchema object
    layer: a list of layer_origin(the original 3D/7D layer), layer_rounded(rounded 3D/7D layer), depending on im2col_enable
    mac_costs: [active_mac_cost, idle_mac_cost]

    Returns
    -------
    """
    # Get the active and idle MAC cost
    [active_mac_cost, idle_mac_cost] = mac_costs
    # Layer
    [layer_origin, layer_rounded] = layer
    # Spatial unrolling
    [spatial_loop, spatial_loop_fractional] = spatial_loop_comb

    # memory allocation part
    allocated_order = allocate_memory_for_tl_order(
        temporal_mapping_ordering,
        spatial_loop,
        layer_origin,
        input_settings,
        mem_scheme.nodes,
    )
    temporal_loop, loop = create_loop_objects(layer_rounded, allocated_order, spatial_loop, input_settings)
    loop_fractional = perform_greedy_mapping(
        layer_origin, allocated_order, spatial_loop_fractional, loop, input_settings
    )
    # utilization part
    utilization = get_utilization(
        layer_rounded,
        temporal_loop,
        spatial_loop_comb,
        loop,
        input_settings,
        mem_scheme,
    )
    total_cost_layer = find_total_cost_layer(
        allocated_order,
        loop_fractional,
        utilization,
        active_mac_cost,
        idle_mac_cost,
        mem_scheme,
        input_settings,
    )
    energy = total_cost_layer
    latency = utilization.latency_no_load
    utilization = utilization.mac_utilize_no_load
    
    return (
        energy,
        utilization,
        latency,
    )
