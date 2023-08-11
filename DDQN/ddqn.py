import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

# Actor network (-> policy approximation)  ---------- ---------- ----------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_actions, hidden_dim = [300, 200], epsilon=1e-6):
        super(Actor, self).__init__()

	# initialize parameters
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.epsilon = epsilon

        # action scaling and bias -> map actions to the desired range
        self.action_scale = torch.FloatTensor((action_dim.high[:n_actions] - action_dim.low[:n_actions]) / 2.)
        self.action_bias = torch.FloatTensor((action_dim.high[:n_actions] + action_dim.low[:n_actions]) / 2.)

        # layers of the neural network
        self.linear1 = nn.Linear(state_dim[0], hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        # save outputs of the network 
        # mean and log_std parameterize the action distribution
        self.mean_linear = nn.Linear(hidden_dim[1], n_actions)
        self.log_std_linear = nn.Linear(hidden_dim[1], n_actions)

    def forward(self, state):
        # forward pass through the network
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        # parameterize the action distribution
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # evaluation -> detach mean and log_std from computation graph
        if not self.training: 
            mean = mean.detach()
            log_std = log_std.detach()

        return mean, log_std

    def sample(self, state):
        # sample action from current policy distribution
        mean, log_std = self.forward(state)
        std = log_std.exp()
 
        # issue: sometimes tensor contains NaN-values
        # fill in zeroes if NaN values are encountered
        if torch.isnan(mean).any():
            mean = torch.where(torch.isnan(mean), torch.tensor(0.0), mean)
        # std must not be all zeroes -> fill in 1e-6
        if torch.isnan(std).any():
            std = torch.where(torch.isnan(std), torch.tensor(1e-6), std)

        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
          
        log_prob = normal.log_prob(x_t)

        # calculate log probability
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(axis=1, keepdim=True)

        return action, log_prob

# Critic network (-> approximate Q-value function)  ---------- ---------- ----------
class Critic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=[300,200]):
        super(Critic, self).__init__()
        # layers of the neural network
        self.linear1 = nn.Linear(state_dim[0] + n_actions, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], 1)
        self.ln1 = nn.LayerNorm(hidden_dim[0])
        self.ln2 = nn.LayerNorm(hidden_dim[1])

    def forward(self, state, action):
        # forward pass through the network (-> estimate Q-value)        
        x = torch.cat([state, action], 1)
        x = self.ln1(F.relu(self.linear1(x)))
        x = self.ln2(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x
        
# ReplayBuffer (-> store and sample transitions) ---------- ---------- ----------
class ReplayBuffer:
    def __init__(self, max_size):
        # initialize parameters
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size=max_size

    def add_transition(self, transitions_new):
        # add a new transition to the buffer
        # replace old transitions if buffer full
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx,:] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        # randomly sample batch of transitions from buffer
        if batch > self.size:
            batch = self.size
        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds,:]

    def get_all_transitions(self):
        # get all transitions from buffer
        return self.transitions[0:self.size]


# DDDQNAgent (-> combine Actor and Critic networks, implement DDDQN algorithm) ---------- ---------- ----------
class DDDQNAgent:
    def __init__(self, state_dim, action_dim, n_actions, hidden_dim, alpha, tau, lr,
                 discount, batch_size, epsilon, max_size):
        
        # initialize agent parameters, create networks
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.tau = tau
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.max_size = max_size

        # create replay buffer
        self.replay_buffer = ReplayBuffer(max_size=self.max_size)

        # create the 2 Q-value networks and their target networks
        self.critic_1 = Critic(self.state_dim, self.n_actions, self.hidden_dim)
        self.critic_2 = Critic(self.state_dim, self.n_actions, self.hidden_dim)
        self.critic_target_1 = Critic(self.state_dim, self.n_actions, self.hidden_dim)
        self.critic_target_2 = Critic(self.state_dim, self.n_actions, self.hidden_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=self.lr)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=self.lr)
        
        # actor network and optimizer
        self.actor = Actor(self.state_dim, self.action_dim, self.n_actions, self.hidden_dim, self.epsilon)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        
        # critic loss function (smooth L1 loss)
        self.critic_loss = nn.SmoothL1Loss()

    def select_action(self, state):
        # select an action 
        # given: current state from the actor network
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(state)            
    
        return action

    def update(self):
        # update actor and critic networks (using DDDQN algorithm)
                
        transitions = self.replay_buffer.sample(batch=self.batch_size)
        s = torch.FloatTensor(np.stack(transitions[:, 0]))
        a = torch.FloatTensor(np.stack(transitions[:, 1])[:, None]).squeeze(dim=1)
        r = torch.FloatTensor(np.stack(transitions[:, 2])[:, None])
        sp = torch.FloatTensor(np.stack(transitions[:, 3]))
        n = torch.FloatTensor(np.stack(transitions[:, 4])[:, None])

        with torch.no_grad():
            # online actor network -> select next actions
            next_actions, _ = self.actor.sample(sp)

            # target critic networks -> Q-values for next state
            next_q1 = self.critic_target_1(sp, next_actions)
            next_q2 = self.critic_target_2(sp, next_actions)

            # take minimum Q-value for each next state-action pair
            next_q_target = torch.min(next_q1, next_q2)

            # DDQL update rule -> target Q-value
            target_q_values = r + (1 - n) * self.discount * next_q_target

        # update Critic Networks
        critic1_pred = self.critic_1(s, a)
        critic2_pred = self.critic_2(s, a)
        critic1_loss = self.critic_loss(critic1_pred, target_q_values)
        critic2_loss = self.critic_loss(critic2_pred, target_q_values)
        self.critic_optimizer_1.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.zero_grad()
        critic2_loss.backward()
        self.critic_optimizer_2.step()

        # updates for target critic networks
        for target_param, param in zip(self.critic_target_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.critic_target_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        # actor loss
        new_actions, log_prob = self.actor.sample(s)
        q_new_actions = torch.min(
            self.critic_1(s, new_actions),
            self.critic_2(s, new_actions)
        )
        actor_loss = (self.alpha * log_prob - q_new_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic1_loss, critic2_loss, actor_loss

    def store_transition(self, transition):
        # store a transition to replay buffer
        self.replay_buffer.add_transition(transition)


