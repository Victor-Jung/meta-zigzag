import math
import random

import numpy as np
import torch
from sympy import factorint
from torch.autograd import Variable

from reinforcement_learning_algo import loop_type_to_ids


class TemporalMappingState:

    def __init__(self, temporal_mapping=None, layer_architecture=None):
        if temporal_mapping is None:
            self.form_temporal_mapping_ordering(layer_architecture)
        else:
            self.value = temporal_mapping
        self.find_temporal_mapping_primary_factors()
        self.randomize_temporal_mapping()
        self.energy = 0
        self.utilization = 0
        self.encoded = False
        self.ordering_size = len(self.value)

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

    def form_temporal_mapping_ordering(self, layer_architecture):
        # Extract the naive TM from the layer architecture contained in layer_post
        self.value = []
        for loop_type, loop_id in loop_type_to_ids.items():
            self.value.append((loop_id, layer_architecture[loop_type]))
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
        """
        Encode self.value
        self.encoded --> True
        """
        self.ordering_size = len(self.value)
        mapping = self.value
        matrix = np.zeros((7,len(mapping)), dtype=int)
        order = 0
        for block, primeFactor in mapping:
            matrix[block - 1][order] = primeFactor
            order += 1
        self.value = matrix
        self.encoded = True

    def pad_temporal_mapping(self, max_length=22):
        if self.encoded == True:
            padding = np.zeros((7,max_length - len(self.value[0])))
            self.value = np.hstack((self.value, padding))
            return self.value
        else:
            for i in range(max_length- len(self.value)):
                self.value.append(0)
            return self.value

    def unpad_temporal_mapping(self):
        if self.encoded == True:
            self.value = self.value[:,:self.ordering_size]
        else:
            self.value = list(filter(lambda x: x != 0, self.value))
        return self.value

    def tm_swap(self, idx1, idx2):
        """
        Column[idx1] <--> Column[idx2]
        """
        if self.encoded == True:
            self.value[:, [idx1, idx2]] = self.value[:, [idx2, idx1]]
        else:
            temp = self.value[idx1]
            self.value[idx1] = self.value[idx2]
            self.value[idx2] = temp
        return self.value

    def make_encoded_state_vector(self, max_input_size=22):
        """
        Encode --> Pad --> toTensor and Resize
        """
        self.encode_temporal_mapping()
        self.pad_temporal_mapping(max_input_size)
        encoded_padded_state = torch.from_numpy(np.asarray(self.value)).float()
        encoded_padded_state = Variable(encoded_padded_state)
        encoded_padded_state = encoded_padded_state.resize(1,1,7,max_input_size)
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
