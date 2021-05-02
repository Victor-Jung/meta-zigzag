import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

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
        self.path = "reinforcement_learning_algo/policy_networks/model.h5"

    def save(self):
        torch.save(self.state_dict(),
                   self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))
        self.eval()

    # make a class prediction for one row of data
    def predict(self, row, model):
        # convert row to data
        # row = Tensor(row)
        # make prediction
        yhat = model(row)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        return yhat
