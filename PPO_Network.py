import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch

import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorCritic(nn.Module):
    def __init__(self, n_inputs, n_actions, config):
        super(ActorCritic, self).__init__()
        fc1_units = config.fc1_units
        fc2_units = config.fc2_units
        lr = config.lr

        self.linear1 = nn.Linear(n_inputs, fc1_units)
        self.linear2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.InstanceNorm1d(fc2_units)
        # self.linear3 = nn.Linear(fc2_units, 64)

        self.actor_mean = nn.Linear(fc2_units, n_actions)
        # self.actor_std = nn.Linear(fc2_units, n_actions)
        self.critic = nn.Linear(fc2_units, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.reset_parameters()
        self.logit_std = nn.Parameter(torch.zeros(n_actions))
        self.lr = lr

    def update_lr(self, lr=2e-4):
        self.optimizer = optim.Adam(self.parameters(),lr=lr)

    def reset_parameters(self):
        """
        Params:
        ===
            initializes weights
        :return:
        """
        self.linear1.weight.data.uniform_(*hidden_init(self.linear1))
        self.linear2.weight.data.uniform_(*hidden_init(self.linear2))
        # self.linear3.weight.data.uniform_(-3e-3, 3e-3)

    # In a PyTorch model, you only have to define the forward pass. PyTorch computes the backwards pass for you!
    def forward(self, x, action=None):
        # print(x[1].shape)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        mean = F.tanh(self.actor_mean(x))
        std = F.softplus(self.logit_std)
        v = self.critic(x)
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = torch.clamp(dist.sample(),-1.0,1.0)
        log_prob = dist.log_prob(action)

        return action, log_prob, dist.entropy(), v