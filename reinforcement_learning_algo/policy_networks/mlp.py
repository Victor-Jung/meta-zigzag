# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable


# class MLP(nn.Module):
#     def __init__(self, observation_space_length, action_space_length):

#         super(MLP, self).__init__()

#         self.action_space_length = action_space_length
#         self.observation_space_length = observation_space_length
#         self.fc1 = nn.Linear(self.observation_space_length, 420)
#         self.dropout = nn.Dropout(p=0.6)
#         self.fc2 = nn.Linear(420, self.action_space_length)
#         self.saved_log_probs = []
#         self.rewards = []

#     def forward(self, x):
#         x = F.relu(self.dropout(self.fc1(x)))
#         action_scores = self.fc2(x)
#         # return F.softmax(action_scores, dim=0)
#         return action_scores

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, observation_space_length, action_space_length):

        super(MLP, self).__init__()

        self.action_space_length = action_space_length
        self.observation_space_length = observation_space_length
    #     self.fc1 = nn.Linear(self.observation_space_length, 420)
    #     self.dropout = nn.Dropout(p=0.6)
    #     self.fc2 = nn.Linear(420, self.action_space_length)
    #     self.saved_log_probs = []
    #     self.rewards = []

    # def forward(self, x):
    #     x = F.relu(self.dropout(self.fc1(x)))
    #     action_scores = self.fc2(x)
    #     # return F.softmax(action_scores, dim=0)
    #     return action_scores

    
        ##### CONV #####
        self.conv1 = nn.Conv2d(1, 6, 3)
        #self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 12, 3)
        #(observationSpace - 4)*(7)*12 = 
        self.action_space_length = action_space_length
        self.flatten_space = (observation_space_length - 4)*3*12
        #self.observation_space_length = observation_space_length
        self.middle = (self.flatten_space + self.action_space_length)//2
        ##### DENSE #####
        self.fc1 = nn.Linear(self.flatten_space, self.middle)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(self.middle, self.action_space_length)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        #x = self.pool(x = F.relu(conv1(x)))
        #x = self.pool(x = F.relu(conv2(x)))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = F.relu(self.dropout(self.fc1(x)))
        action_scores = self.fc2(x)
        return action_scores
        #return F.softmax(action_scores, dim=0)
