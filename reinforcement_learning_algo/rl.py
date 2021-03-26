import copy
import math
import random
from itertools import count

import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from reinforcement_learning_algo.cost_esimator import *


# from reinforcement_learning_algo.optimizer import initialize_temporal_mapping, get_temporal_loop_estimation

# Here our network is a MLP (Multi Layer Perceptron)
class PolicyGradient:
    """
    Create policy network which takes state featues as input and outputs unnormalized 
    action values.
    """

    def __init__(self, neural_network, temporal_mapping_ordering, layer_, layer_post,
                 im2col_layer, layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su):

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

    def encode_temporal_mapping(self, temporal_mapping):
        encoded_temporal_mapping = []
        for loop_type, loop_weight in temporal_mapping:
            encoded_value = loop_type + 10 ** -math.floor(math.log10(loop_weight) + 1) * loop_weight
            encoded_temporal_mapping.append(encoded_value)
        return encoded_temporal_mapping

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

    def step(self, state, action, max_input_size=30):

        padded_state = self.pad_temporal_mapping(state, max_input_size)
        padded_next_state = self.tm_swap(action[0], action[1], padded_state)
        next_state = self.unpad_temporal_mapping(padded_next_state)
        energy, utilization = get_temporal_loop_estimation(next_state, self.input_settings, self.spatial_loop_comb,
                                                           self.mem_scheme, self.layer, self.mac_costs)

        reward = 1 / (energy / 10 ** 8)
        return next_state, reward

    def generate_swap_list(self, input_size):
        swap_list = []
        starting_idx = 1
        for i in range(input_size):
            for j in range(starting_idx, input_size):
                swap_list.append([i, j])
            starting_idx += 1
        return swap_list

    def make_encoded_state_vector(self, state, max_input_size):
        encoded_state = self.encode_temporal_mapping(state)
        encoded_padded_state = self.pad_temporal_mapping(encoded_state, max_input_size)
        encoded_padded_state = torch.from_numpy(np.asarray(encoded_padded_state)).float()
        encoded_padded_state = Variable(encoded_padded_state)
        return encoded_padded_state

    def get_action(self, probability_vector):
        action_idx = np.random.choice(len(probability_vector), 1, p=probability_vector.detach().numpy())[0]
        action = self.swap_list[action_idx]
        return action_idx, action

    def compute_discounted_reward(self, reward_pool, steps, gamma):
        running_add = 0
        for i in reversed(range(steps)):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * gamma + reward_pool[i]
                reward_pool[i] = running_add
        return reward_pool

    def normalize_reward(self, reward_pool, steps):
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(steps):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        return reward_pool

    def optimize(self, writer, episode, steps, state_pool, action_pool, reward_pool):
        self.optimizer.zero_grad()
        running_loss = 0
        for step in range(steps):
            state = state_pool[step]
            action_idx = Variable(torch.FloatTensor([action_pool[step]]))
            reward = reward_pool[step]

            probability_vector = self.policy_net(state)
            m = Categorical(probability_vector)
            # Negtive score function x reward
            loss = -m.log_prob(action_idx) * reward
            loss.backward()
            running_loss += loss.item()
            self.optimizer.step()
            iteration = (episode - 1) * steps + step
            print(f"Iteration{iteration} — Step {step} Training loss: {running_loss / steps}")
            writer.add_scalar("loss x epoch", loss.item(), iteration)
        return

    def training(self, starting_tm, num_episode, episode_max_step, batch_size, learning_rate, gamma):
        writer = SummaryWriter()
        # Batch History
        state_pool = []
        action_pool = []
        reward_pool = []
        episode_durations = []

        # Here the only way to end an episode is with a counter
        steps = 0
        input_size = len(starting_tm)
        max_input_size = 30
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.swap_list = self.generate_swap_list(max_input_size)

        for episode in range(num_episode):

            # Init at a random state, equivalent env.reset() with gym
            state = copy.deepcopy(starting_tm)
            random.shuffle(state)

            for time_step in count():

                # Encode and pad the state to fit in the policy network
                encoded_padded_state = self.make_encoded_state_vector(state, max_input_size)
                probability_vector = self.policy_net(encoded_padded_state)
                action_idx, action = self.get_action(probability_vector)

                # Take a step into the env and get the next state and the reward
                next_state, reward = self.step(state, action)
                print(f"Episode: {episode}, Step: {time_step}, Reward: {reward}")

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
            if episode > 0 and episode % batch_size == 0:
                # Compute the discounted return (discounted reward sum)
                reward_pool = self.compute_discounted_reward(reward_pool, steps, gamma)
                # Normalize reward
                reward_pool = self.normalize_reward(reward_pool, steps)
                # Gradient Desent
                self.optimize(writer, episode, steps, state_pool, action_pool, reward_pool)
                state_pool = []
                action_pool = []
                reward_pool = []
                steps = 0

        writer.close()
