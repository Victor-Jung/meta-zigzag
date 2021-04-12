import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, observation_space_length, action_space_length):

        super(MLP, self).__init__()

        self.action_space_length = action_space_length
        self.observation_space_length = observation_space_length
        self.fc1 = nn.Linear(self.observation_space_length, 420)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(420, self.action_space_length)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        action_scores = self.fc2(x)
        # return F.softmax(action_scores, dim=0)
        return action_scores
