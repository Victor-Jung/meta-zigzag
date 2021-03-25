import os
import copy
import math
import random
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from reinforcement_learning_algo.cost_esimator import *

#from reinforcement_learning_algo.optimizer import initialize_temporal_mapping, get_temporal_loop_estimation

# Here our network is a MLP (Multi Layer Perceptron)
class PolicyGradient:
    """
    Create policy network which takes state featues as input and outputs unnormalized 
    action values.
    """

    def __init__(self, neural_network, temporal_mapping_ordering, layer_, layer_post, 
                 im2col_layer, layer_rounded, spatial_loop_comb,input_settings, mem_scheme, ii_su):

        super(PolicyGradient, self).__init__()

        # Init our policy net
        self.policy_net = neural_network

        # Starting Temporal Mapping
        self.starting_temporal_mapping = temporal_mapping_ordering

        # Cost Estimation Parameters
        self.layer_ = layer_
        self.layer_post = layer_post
        self.im2col_layer = im2col_layer
        self.layer_rounded = layer_rounded
        self.spatial_loop_comb = spatial_loop_comb
        self.input_settings = input_settings
        self.mem_scheme = mem_scheme
        self.ii_su = ii_su
        self.layer = [self.im2col_layer, self.layer_rounded]
        self.mac_costs = calculate_mac_level_costs(self.layer_, self.layer_rounded, 
                                                   self.input_settings, self.mem_scheme, self.ii_su)


    def encode_temporal_mapping(self, tm):

        newSeq = []
        for i in tm:
            enc = i[0]+10**-math.floor(math.log10(i[1])+1)*i[1]
            newSeq.append(enc)
        return newSeq

    def pad_temporal_mapping(self, tm, max_length):
        for i in range(max_length - len(tm)):
            tm.append(0)
        return tm

    def unpad_temporal_mapping(self, tm):
        unpadded_tm = list(filter(lambda x: x != 0, tm))
        return unpadded_tm

    def tm_swap(self, idx1, idx2, tm):
        temp = tm[idx1]
        tm[idx1] = tm[idx2]
        tm[idx2] = temp
        return tm

    def step(self, state, action):

        padded_state = self.pad_temporal_mapping(state, 30)
        padded_next_state = self.tm_swap(action[0], action[1], padded_state)
        next_state = self.unpad_temporal_mapping(padded_next_state)

        energy, utilization = get_temporal_loop_estimation(next_state, self.input_settings, self.spatial_loop_comb,
                                                           self.mem_scheme, self.layer, self.mac_costs)

        reward = 1/(energy/100000000)
        return next_state, reward

    def action_idx_to_swap(self, action_idx, input_size):

        swap_list = []
        starting_idx = 1
        for i in range(input_size):
            for j in range(starting_idx, input_size):
                swap_list.append([i, j])
            starting_idx += 1

        return swap_list[action_idx]


    def training(self, starting_TM, num_episode, episode_max_step, batch_size, learning_rate, gamma):

        # Batch History
        state_pool = []
        action_pool = []
        reward_pool = []
        episode_durations = []

        # Here the only way to end an episode is with a counter
        steps = 0

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)

        for ep in range(num_episode):
            
            # Init at a random state, equivalent env.reset() with gym
            state = copy.deepcopy(starting_TM)
            random.shuffle(state)

            for time_step in count():
                
                # Encode and pad the state to fit in the policy network
                encoded_state = self.encode_temporal_mapping(state)
                encoded_padded_state = self.pad_temporal_mapping(encoded_state, 30)

                encoded_padded_state = torch.from_numpy(np.asarray(encoded_padded_state)).float()
                encoded_padded_state = Variable(encoded_padded_state)

                probs = self.policy_net(encoded_padded_state)
                action_idx = np.random.choice(len(probs), 1, p=probs.detach().numpy())
                action = self.action_idx_to_swap(action_idx[0], 30)

                # Take a step into the env and get the next state and the reward
                next_state, reward = self.step(state, action)
                print("Reward : ", reward)

                # Save for the history
                state_pool.append(encoded_padded_state)
                action_pool.append(action_idx)
                reward_pool.append(reward)

                state = next_state
                steps += 1

                if time_step >= episode_max_step:
                    episode_durations.append(time_step + 1)
                    break

            # Update Policy
            if ep > 0 and ep % batch_size == 0:

                # Compute the discounted return (discounted reward sum)
                running_add = 0
                for i in reversed(range(steps)):
                    if reward_pool[i] == 0:
                        running_add = 0
                    else:
                        running_add = running_add * gamma + reward_pool[i]
                        reward_pool[i] = running_add

                # Normalize reward
                reward_mean = np.mean(reward_pool)
                reward_std = np.std(reward_pool)
                for i in range(steps):
                    reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

                # Gradient Desent
                self.optimizer.zero_grad()

                for i in range(steps):

                    state = state_pool[i]
                    action_idx = Variable(torch.FloatTensor([action_pool[i]]))
                    reward = reward_pool[i]

                    probs = self.policy_net(state)
                    m = Categorical(probs)
                    # Negtive score function x reward
                    loss = -m.log_prob(action_idx) * reward
                    loss.backward()

                self.optimizer.step()

                state_pool = []
                action_pool = []
                reward_pool = []
                steps = 0
        
