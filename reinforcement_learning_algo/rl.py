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
            encoded_value = loop_type + 10 ** - \
                math.floor(math.log10(loop_weight) + 1) * loop_weight
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

    def step(self, state, action, energy_pool, time_step, max_input_size=30):

        padded_state = self.pad_temporal_mapping(state, max_input_size)
        padded_next_state = self.tm_swap(action[0], action[1], padded_state)
        next_state = self.unpad_temporal_mapping(padded_next_state)
        energy, utilization = get_temporal_loop_estimation(next_state, self.input_settings, self.spatial_loop_comb,
                                                           self.mem_scheme, self.layer, self.mac_costs)

        '''if time_step == 0:
            previous_energy = 0
        else:
            previous_energy = energy_pool[-1]
        energy_pool.append(energy)

        if energy - previous_energy > 0:
            reward = 1
        else:
            reward = 0'''

        reward = 1/(energy/(10**12))

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
        encoded_padded_state = self.pad_temporal_mapping(
            encoded_state, max_input_size)
        encoded_padded_state = torch.from_numpy(
            np.asarray(encoded_padded_state)).float()
        encoded_padded_state = Variable(encoded_padded_state)
        return encoded_padded_state

    def get_action(self, probability_vector):
        action_idx = np.random.choice(
            len(probability_vector), 1, p=probability_vector.detach().numpy())[0]
        action = self.swap_list[action_idx]
        return action_idx, action

    def compute_discounted_rewards(self, reward_pool, steps, gamma):
        discounted_returns = []
        for t in range(steps):
            discounted_return = 0
            gamma_power = 0
            for r in reward_pool[t:]:
                discounted_return = discounted_return + \
                    reward_pool[t] * (gamma ** gamma_power)
                gamma_power += 1
            discounted_returns.append(discounted_return)

        return discounted_returns

    def normalize_reward(self, reward_pool, steps):
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(steps):
            if reward_std != 0:
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std
            else:
                reward_pool[i] = (reward_pool[i] - reward_mean)

        return reward_pool

    def optimize(self, writer, episode, steps, reward_pool, log_probability_list):

        loss_list = []
        # Gt is the symbol for the return (ie : the sum of discounted rewards)
        for log_prob, Gt in zip(log_probability_list, reward_pool):
            loss = -log_prob * Gt
            loss_list.append(loss)

        loss = torch.stack(loss_list).sum()
        loss.backward()
        self.optimizer.zero_grad()
        self.optimizer.step()
        iteration = (episode - 1) * steps
        print(f"Iteration{iteration} — Training loss: {loss.item()}")
        writer.add_scalar("loss x epoch", loss.item(), iteration)


    def training(self, starting_tm, num_episode, episode_max_step, batch_size, learning_rate, gamma):

        writer = SummaryWriter()

        # Batch History
        state_pool = []
        action_pool = []
        reward_pool = []
        energy_pool = []
        log_probability_list = []
        episode_durations = []

        # Here the only way to end an episode is with a counter
        steps = 0
        input_size = len(starting_tm)
        max_input_size = 30
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.swap_list = self.generate_swap_list(max_input_size)

        for episode in range(num_episode):

            # Init at a random state, equivalent env.reset() with gym
            state = copy.deepcopy(starting_tm)
            #random.shuffle(state)

            for time_step in count():

                # Encode and pad the state to fit in the policy network
                encoded_padded_state = self.make_encoded_state_vector(state, max_input_size)
                probability_vector = self.policy_net(encoded_padded_state)
                action_idx, action = self.get_action(probability_vector)

                m = Categorical(probability_vector)
                action_sample = m.sample()
                log_probability_list.append(m.log_prob(action_sample))
                #log_probability_list.append(torch.log(probability_vector.squeeze(0)[action_idx]))

                # Take a step into the env and get the next state and the reward

                next_state, reward = self.step(state, action, energy_pool, time_step)

                # Save for the history
                state_pool.append(encoded_padded_state)
                action_pool.append(action_idx)
                reward_pool.append(reward)

                state = next_state
                steps += 1

                if time_step >= episode_max_step - 1:
                    episode_durations.append(time_step + 1)
                    print(f"Episode: {episode} Average Reward: {np.mean(reward_pool)}")
                    writer.add_scalar("Reward x epoch", np.mean(reward_pool), episode)
                    break

            # Update Policy
            if episode % batch_size == 0:
                # Compute the discount ed return (discounted reward sum)
                reward_pool = self.compute_discounted_rewards(reward_pool, steps, gamma)
                # Normalize reward
                reward_pool = self.normalize_reward(reward_pool, steps)
                # Gradient Desent
                self.optimize(writer, episode, steps, reward_pool, log_probability_list)
                state_pool = []
                action_pool = []
                reward_pool = []
                log_probability_list = []
                steps = 0

        writer.close()
        
    def run_episode(self, starting_temporal_mapping, episode_max_step):

        state = starting_temporal_mapping
        energy_pool = []

        for time_step in count():

            # Encode and pad the state to fit in the policy network
            encoded_padded_state = self.make_encoded_state_vector(state, 30)
            probability_vector = self.policy_net(encoded_padded_state)
            action_idx, action = self.get_action(probability_vector)

            # Take a step into the env and get the next state and the reward
            next_state, reward = self.step(state, action, energy_pool, time_step)
            # print("Reward : ", reward)
            state = next_state

            if time_step >= episode_max_step - 1:
                break
        
        return state
