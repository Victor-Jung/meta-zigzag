import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, observation_space_length, action_space_length):

        super(MLP, self).__init__()

        self.action_space_length = action_space_length
        self.observation_space_length = observation_space_length
        self.fc1 = nn.Linear(self.observation_space_length, 256)
        self.fc2 = nn.Linear(256, self.action_space_length)
        self.fc3 = nn.Linear(self.action_space_length, self.action_space_length)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x
