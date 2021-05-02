import math
import random

import numpy as np
import torch
from sympy import factorint
from torch.autograd import Variable

from reinforcement_learning_algo import loop_type_to_ids


class TemporalMappingState:

    def __init__(self, spatial_unrolling, temporal_mapping=None, layer_architecture=None):
        if temporal_mapping is None:
            self.form_temporal_mapping_ordering(layer_architecture, spatial_unrolling)
        else:
            self.value = temporal_mapping
        self.find_temporal_mapping_primary_factors()
        self.randomize_temporal_mapping()
        self.energy = 0
        self.utilization = 0

    def pf_to_compressed_mapping(self):
        compressed_temporal_mapping = [self.value[0]]
        for i in range(1, len(self.value)):
            if self.value[i][0] == compressed_temporal_mapping[-1][0]:
                compressed_temporal_mapping[-1] = (
                    self.value[i][0],
                    self.value[i][1] * compressed_temporal_mapping[-1][1],
                )
            else:
                compressed_temporal_mapping.append(self.value[i])
        self.value = compressed_temporal_mapping
        return self.value

    def form_temporal_mapping_ordering(self, layer_architecture, spatial_unrolling):

        layer_spec_temporal = {}
        ids_to_loop_type = {1: 'FX', 2: 'FY', 3: 'OX', 4: 'OY', 5: 'C', 6: 'K', 7: 'B'}

        # Extract the naive TM from the layer architecture contained in layer_post
        self.value = []
        for loop_type, loop_id in loop_type_to_ids.items():
            layer_spec_temporal[loop_id] = layer_architecture[loop_type]   

        # Update the temporal layer spec to remove the already spatially unrolled dimensions.
        for level in range(0, len(spatial_unrolling['W'])):
            for [loop_type_number, su_factor] in spatial_unrolling['W'][level]:
                try:
                    pf = layer_spec_temporal[loop_type_number]
                except:
                    continue
                q, rem = divmod(pf, su_factor)
                assert rem == 0 # pf/su_factor should have remainder 0
                layer_spec_temporal[loop_type_number] = q

        # Then filter the 1-size loops
        for loop_id, loop_size in list(layer_spec_temporal.items()):
            if loop_size != 1:
                self.value.append((loop_id, loop_size))
                
        return self.value

    def find_temporal_mapping_primary_factors(self):
        # Break it down to LPF (Loop Prime Factor)
        temporal_mapping_primary_factors = []
        for inner_loop in self.value:
            if inner_loop[1] == 1:
                temporal_mapping_primary_factors.append(inner_loop)
            else:
                factors = factorint(inner_loop[1])
                for factor in factors.items():
                    for pow in range(factor[1]):
                        temporal_mapping_primary_factors.append((inner_loop[0], factor[0]))
        self.value = temporal_mapping_primary_factors
        return self.value

    def randomize_temporal_mapping(self):
        # Shuffle the LPF_TM to get a random initalization into the design space
        random.shuffle(self.value)
        return self.value

    def initialize_temporal_mapping(self):
        self.find_temporal_mapping_primary_factors()
        self.randomize_temporal_mapping()
        return self.value

    def encode_temporal_mapping(self):
        encoded_temporal_mapping = []
        for loop_type, loop_weight in self.value:
            encoded_value = loop_type + 10 ** - \
                math.floor(math.log10(loop_weight) + 1) * loop_weight
            encoded_temporal_mapping.append(encoded_value)
        self.value = encoded_temporal_mapping
        return encoded_temporal_mapping

    def pad_temporal_mapping(self, max_length=22):
        for i in range(max_length - len(self.value)):
            self.value.append(0)
        return self.value

    def unpad_temporal_mapping(self):
        self.value = list(filter(lambda x: x != 0, self.value))
        return self.value

    def tm_swap(self, idx1, idx2):
        try:
            temp = self.value[idx1]
            self.value[idx1] = self.value[idx2]
            self.value[idx2] = temp
        except:
            pass
        return self

    def make_encoded_state_vector(self, max_input_size=22):
        self.encode_temporal_mapping()
        self.pad_temporal_mapping(max_input_size)
        encoded_padded_state = torch.from_numpy(np.asarray(self.value)).float()
        encoded_padded_state = Variable(encoded_padded_state)
        return encoded_padded_state

    def get_state_dict(self, energy, utilization):
        self.energy = energy
        self.utilization = utilization
        return {
            "temporal_mapping": self,
            "energy": self.energy,
            "utilization": self.utilization,
        }

    def get_dict(self):
        return {
            "temporal_mapping": self.value,
            "energy": self.energy,
            "utilization": self.utilization,
        }
