import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, observation_space_length, action_space_length):

        super(MLP, self).__init__()

        self.action_space_length = action_space_length
        self.observation_space_length = observation_space_length

        self.actor_fc1 = nn.Linear(self.observation_space_length, 420)
        self.actor_dropout = nn.Dropout(p=0.6)
        self.actor_fc2 = nn.Linear(420, self.action_space_length)

        self.critic_fc1 = nn.Linear(self.observation_space_length, 256)
        self.critic_fc2 = nn.Linear(256, 1)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state):

        action_scores = F.relu(self.actor_dropout(self.actor_fc1(state)))
        action_scores = self.actor_fc2(action_scores)

        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)

        return value, action_scores
