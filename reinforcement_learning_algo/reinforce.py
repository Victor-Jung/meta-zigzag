from copy import deepcopy
from itertools import count

import numpy as np
import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from reinforcement_learning_algo import eps
from reinforcement_learning_algo.core.environment import Environment
from reinforcement_learning_algo.core.action import Action
from reinforcement_learning_algo.cost_esimator import *


class PolicyGradient:
    """
    Create policy network which takes state features as input and outputs denormalized
    action values.
    """

    def __init__(self, neural_network, temporal_mapping_ordering, layer,
                 im2col_layer, layer_rounded, spatial_loop_comb, input_settings, mem_scheme, ii_su):

        super(PolicyGradient, self).__init__()

        # Init our policy net
        self.policy_net = neural_network

        # Starting Temporal Mapping
        self.starting_temporal_mapping = temporal_mapping_ordering

        # Cost Estimation Parameters
        self.loss = 0
        self.layer = layer
        self.layer_post = layer.size_list_output_print
        self.im2col_layer = im2col_layer
        self.layer_rounded = layer_rounded
        self.spatial_loop_comb = spatial_loop_comb
        self.input_settings = input_settings
        self.mem_scheme = mem_scheme
        self.ii_su = ii_su
        self.full_layer = [self.im2col_layer, self.layer_rounded]
        self.mac_costs = calculate_mac_level_costs(self.layer, self.layer_rounded,
                                                   self.input_settings, self.mem_scheme, self.ii_su)

    def optimize(self, writer, episode, steps, reward_pool, log_probability_list, verbose=0):
        pass

    def select_action(self, state):
        encoded_padded_state = deepcopy(state)
        encoded_padded_state = encoded_padded_state['temporal_mapping']
        encoded_padded_state = encoded_padded_state.make_encoded_state_vector()
        self.probability_vector = self.policy_net(encoded_padded_state)
        m = Categorical(self.probability_vector)
        action = m.sample()
        self.policy_net.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def calculate_rewards(self, gamma=0.9) -> list:
        R = 0
        returns = []
        for reward in self.policy_net.rewards[::-1]:
            R = reward + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        # normalize rewards
        if (returns.std() + eps) > 0:
            returns = (returns - returns.mean()) / (returns.std() + eps)
        return returns

    def calculate_loss(self, returns) -> list:
        policy_loss = []
        for log_prob, reward in zip(self.policy_net.saved_log_probs, returns):
            policy_loss.append(-log_prob * reward)
        return policy_loss

    def finish_episode(self, gamma=0.9):
        """
        Function for calculating rewards, loss and doing backward propagation.
        :return:
            policy_loss - summarized loss
        """
        returns = self.calculate_rewards(gamma)
        policy_loss = self.calculate_loss(returns)
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        # clean memory
        del self.policy_net.rewards[:]
        del self.policy_net.saved_log_probs[:]
        return policy_loss

    def training(self, learning_rate=1e-2, reward_stop_condition=0.5, gamma=0.9, log_interval=1, observation_state_length=22,
                 episode_utilization_stop_condition=0.8, timestamp_number=50):
        writer = SummaryWriter()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        env = Environment(layer=self.layer, im2col_layer=self.im2col_layer, layer_rounded=self.layer_rounded,
                          spatial_loop_comb=self.spatial_loop_comb, input_settings=self.input_settings,
                          mem_scheme=self.mem_scheme, ii_su=self.ii_su, mac_costs=self.mac_costs,
                          observation_state_length=observation_state_length,
                          utilization_threshold=episode_utilization_stop_condition, timestamp_threshold=timestamp_number)
        step = 0
        best_result = ()
        max_reward = 0
        for i_episode in count(1):
            done = False
            state, episode_reward = env.reset(), 0
            episode_rewards = []
            for timestamp in range(1, 10000):  # Don't do infinite loop while learning
                if done:
                    env.reset()
                    break
                action = self.select_action(state)
                state, reward, done, info = env.step(action, timestemp=timestamp)
                if reward > max_reward:
                    max_reward = reward
                    best_result = state
                self.policy_net.rewards.append(reward)
                episode_reward += reward
                episode_rewards.append(reward)
                step += 1
                writer.add_scalar("Episode reward", reward, step)
            writer.add_scalar("Episode mean reward", np.mean(episode_rewards), step)
            running_reward = np.mean(episode_rewards)
            loss = self.finish_episode(gamma)
            if i_episode % log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, episode_reward, running_reward))
                writer.add_scalar("Loss", loss, i_episode)
                writer.add_scalar("Reward", running_reward, i_episode)
            if running_reward >= reward_stop_condition:
                best_result["temporal_mapping"] = best_result["temporal_mapping"].value
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, timestamp))
                print(f"Best result: {best_result}\treward{max_reward}")
                break

    def run_episode(self, starting_temporal_mapping, episode_max_step):
        pass
