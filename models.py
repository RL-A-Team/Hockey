import torch
import torch.nn as nn


# Define the actor network
class ActorParameterized(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorParameterized, self).__init__()
        self.action_dim = int(action_dim/2)
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.fc_input = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc_output = nn.Linear(self.hidden_dim, self.action_dim)
        self.fc_a0 = nn.Linear(self.hidden_dim, 1)
        self.fc_k = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc_input(state))
        x = self.relu(self.fc_hidden(x))
        k_t = self.relu(self.fc_k(x))
        a0 = self.relu(self.fc_a0(x))
        x = self.tanh(k_t * self.outpu(x) - k_t * a0)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.action_dim = int(action_dim/2)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x


# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.action_dim = int(action_dim/2)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(self.state_dim + self.action_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x