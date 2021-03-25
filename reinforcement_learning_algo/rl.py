import os
import copy
import random
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torch.distributions import Categorical

# Here our network is a MLP (Multi Layer Perceptron)
class PolicyNetwork(nn.Module):
    """
    Create policy network which takes state featues as input and outputs unnormalized 
    action values.
    """

    def __init__(self, observation_space_length, action_space_length):
        super(PolicyNetwork, self).__init__()

        self.action_space_length = action_space_length
        self.observation_space_length = observation_space_length
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc2 = nn.Linear(256, self.num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def training(starting_TM, num_episode, episode_max_step, batch_size, learning_rate, gamma):

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    episode_durations = []

    # Here the only way to end an episode is with a counter
    steps = 0

    policy_net = PolicyNetwork()
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)

    for ep in range(num_episode):
        
        # Init at a random state, equivalent env.reset() with gym
        state = copy.deepcopy(starting_TM)
        random.shuffle(state)

        for time_step in count():

            # Pick a sample in this action space
            probs = policy_net(state)
            action_space = Bernoulli(probs)
            action = action_space.sample()

            # Take a step into the env and get the next state and the reward
            next_state, reward = [], 42

            # Save for the history
            state_pool.append(state)
            action_pool.append(float(action))
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
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]

                probs = policy_net(state)
                m = Bernoulli(probs)
                # Negtive score function x reward
                loss = -m.log_prob(action) * reward
                loss.backward()

            optimizer.step()

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0
        
