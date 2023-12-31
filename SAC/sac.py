import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, n_actions, hidden_dim=[300, 200], epsilon=1e-6):
        """
        :param state_dim: (int)
            dimension of the state space
        :param action_dim: Box
            dimension of the action space
        :param n_actions: int
            number of actions
        :param hidden_dim: List[int, int]
            dimensions of the hidden layers
        :param epsilon: float
            input noise
        """

        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.n_actions = n_actions
        self.epsilon = epsilon

        # scaling values of the action
        self.action_scale = torch.FloatTensor((action_dim.high[:n_actions] - action_dim.low[:n_actions]) / 2.)
        self.action_bias = torch.FloatTensor((action_dim.high[:n_actions] + action_dim.low[:n_actions]) / 2.)

        # define the NN layers
        self.linear1 = nn.Linear(state_dim[0], hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])

        self.mean_linear = nn.Linear(hidden_dim[1], n_actions)
        self.log_std_linear = nn.Linear(hidden_dim[1], n_actions)

    def forward(self, state):
        """
        :param state: Tensor
            State of the environment
        :return: (Tensor, Tensor)
            Mu and log_sigma of the learned Gaussian
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mu = self.mean_linear(x)
        log_sigma = self.log_std_linear(x)

        return mu, log_sigma

    def sample(self, state):
        """
        :param state: Tensor
            State of the environment
        :return: (Tensor, Tensor, Tensor)
            Sampeled and scaled action, log probability of the action, scaled mu of the Gaussian
        """
        mu, log_sigma = self.forward(state)
        std = log_sigma.exp()
        normal = Normal(mu, std)

        # reparameterization trick
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x)

        log_prob -= torch.log(self.action_scale * (1 - y.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(axis=1, keepdim=True)
        mu = torch.tanh(mu) * self.action_scale + self.action_bias

        return action, log_prob, mu


class Critic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dim=[300, 200]):
        """

        :param state_dim: (int)
            dimension of the state space
        :param n_actions: int
            number of actions
        :param hidden_dim: List[int, int]
            dimensions of the hidden layers
        """
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim[0] + n_actions, hidden_dim[0])
        self.linear2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.linear3 = nn.Linear(hidden_dim[1], 1)
        self.ln1 = nn.LayerNorm(hidden_dim[0])
        self.ln2 = nn.LayerNorm(hidden_dim[1])

    def forward(self, state, action):
        """

        :param state: Tensor
            State of the environment
        :param action: Tensor
            Taken action
        :return: Tensor
            Estimated soft Q-Values of the state-action pairs
        """
        x = torch.cat([state, action], 1)
        x = self.ln1(F.relu(self.linear1(x)))
        x = self.ln2(F.relu(self.linear2(x)))
        x = self.linear3(x)
        return x


class ReplayBuffer:
    def __init__(self, max_size=100000):
        """

        :param max_size: int
            maximal size of the buffer
        """
        self.transitions = np.asarray([])
        self.size = 0
        self.current_idx = 0
        self.max_size = max_size

    def add_transition(self, transitions_new):
        """

        :param transitions_new: Tuple
            Transition to add to the buffer
        """
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)

        self.transitions[self.current_idx, :] = np.asarray(transitions_new, dtype=object)
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        """

        :param batch: int
            number of transitions to sample
        :return: np.array
            sampled transitions
        """
        if batch > self.size:
            batch = self.size

        self.inds = np.random.choice(range(self.size), size=batch, replace=False)
        return self.transitions[self.inds, :]

    def get_all_transitions(self):
        """

        :return: np.array
            all tranistions stored in the buffer
        """
        return self.transitions[0:self.size]


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, alpha=0.6, beta=0.4, beta_annealing=0.0001, **kwargs):
        """ Initialize Prioritized Replay Buffer as described in
            Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).

        :param alpha: float
            Prioritization of transitions degree
        :param beta: float
            Initial importance sampling correction degree
        :param beta_annealing: float
            Factor to anneal beta over time
        """
        super(PrioritizedReplayBuffer, self).__init__(**kwargs)

        self.priorities = np.zeros((self.max_size,), dtype=np.float32)
        self.alpha = alpha
        self.beta_0 = beta
        self.beta_annealing = beta_annealing

    def add_transition(self, transitions_new):
        """

        :param transitions_new: Tuple
            tranistion to add to the replay buffer
        """
        if self.size == 0:
            blank_buffer = [np.asarray(transitions_new, dtype=object)] * self.max_size
            self.transitions = np.asarray(blank_buffer)
            max_prio = 1e-5  # not 0 to prevent numerical errors
        else:
            max_prio = self.priorities.max()

        self.transitions[self.current_idx, :] = np.asarray(transitions_new, dtype=object)
        self.priorities[self.current_idx] = max_prio
        self.size = min(self.size + 1, self.max_size)
        self.current_idx = (self.current_idx + 1) % self.max_size

    def sample(self, batch=1):
        """

        :param batch: int
            number of transitions to sample
        :return: (np.array, np.array, np.array
            sampled transitions, indices of these transitions, weights of these transitions
        """
        if batch > self.size:
            batch = self.size

        probabilities = self.priorities[:self.size] ** self.alpha
        P = probabilities / probabilities.sum()

        self.inds = np.random.choice(range(self.size), batch, p=P)

        # beta annealing
        beta = min(1, self.beta_0 + (1. - self.beta_0) * self.beta_annealing)

        weights = (self.size * P[self.inds]) ** (-beta)
        weights = np.array(weights / weights.max())

        return self.transitions[self.inds, :], self.inds, weights

    def update_priorities(self, indices, new_priorities):
        """

        :param indices: np.array
            indices of the transitions whose priorities should be updated
        :param new_priorities: np.array
        :return:
        """
        self.priorities[indices] = new_priorities


# Define the soft actor-critic agent
class SACAgent():
    def __init__(self, state_dim, action_dim, n_actions=4, hidden_dim=[300, 200], alpha=0.2, tau=5e-3, lr=1e-3,
                 discount=0.99, batch_size=256, autotune=False, loss='l1', deterministic_action=False,
                 prio_replay_buffer=False):
        """ SAC agent

        :param state_dim: tuple
            Dimensions of the state
        :param action_dim: Box
            Dimensions of the actions
        :param n_actions: int
            Number of actions
        :param hidden_dim: list
            List of hidden dimensions in the actor and critic network
        :param alpha: float
            Level of Entropy regularization
        :param tau: float
            Soft target updating factor
        :param lr: float
            Learning rate
        :param discount: float
            Discount factor
        :param batch_size: int
            Number of transitions in update step
        :param autotune: bool
            Autotune the alpha parameter
        :param loss: str
            'l1' loss or 'l2' loss
        :param deterministic_action: bool
            use deterministic actions (scaled learned mean)
        :param prio_replay_buffer: bool
            Use prioritized replay buffer instead the usual one
        """

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.tau = tau
        self.lr = lr
        self.discount = discount
        self.batch_size = batch_size
        self.deterministic_action = deterministic_action
        self.prio_replay_buffer = prio_replay_buffer

        if self.prio_replay_buffer:
            self.replay_buffer = PrioritizedReplayBuffer()
        else:
            self.replay_buffer = ReplayBuffer()

        # Init actor and critic, their optimizer and the target networks
        self.actor = Actor(self.state_dim, self.action_dim, self.n_actions, self.hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)

        self.critic_1 = Critic(self.state_dim, self.n_actions, self.hidden_dim)
        self.critic_target_1 = Critic(self.state_dim, self.n_actions, self.hidden_dim)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=self.lr)

        self.critic_2 = Critic(self.state_dim, self.n_actions, self.hidden_dim)
        self.critic_target_2 = Critic(self.state_dim, self.n_actions, self.hidden_dim)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=self.lr)

        if loss == 'l1':
            self.critic_loss = nn.SmoothL1Loss()
        elif loss == 'l2':
            self.critic_loss = nn.MSELoss()
        else:
            raise ValueError(f'Loss {loss} not defined! Please define loss either as l1 or l2')

        # Based on https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details
        self.autotune = autotune
        if self.autotune:
            self.target_entropy = -torch.Tensor(n_actions)
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = alpha

    def set_deterministic(self, deterministic_action):
        """ set the deterministic value """
        self.deterministic_action = deterministic_action

    def select_action(self, state):
        """ return an action """
        state = torch.FloatTensor(state).unsqueeze(0)

        action, log_sigma, mu = self.actor.sample(state)

        try:
            if self.deterministic_action:
                action = mu
        except:
            action = mu

        return action.detach().numpy()[0]

    def update(self):
        """ update the networks"""
        if self.prio_replay_buffer:
            transitions, inds, weights = self.replay_buffer.sample(batch=self.batch_size)
            weights = torch.FloatTensor(weights).unsqueeze(1)
        else:
            transitions = self.replay_buffer.sample(batch=self.batch_size)
        s = torch.FloatTensor(np.stack(transitions[:, 0]))
        a = torch.FloatTensor(np.stack(transitions[:, 1])[:, None]).squeeze(dim=1)
        r = torch.FloatTensor(np.stack(transitions[:, 2])[:, None])
        sp = torch.FloatTensor(np.stack(transitions[:, 3]))
        d = torch.FloatTensor(np.stack(transitions[:, 4])[:, None])

        with torch.no_grad():
            next_actions, log_prob, _ = self.actor.sample(sp)

            target_q_values = torch.min(
                self.critic_1(sp, next_actions),
                self.critic_2(sp, next_actions),
            ) - self.alpha * log_prob
            q_target = r + (1 - d) * self.discount * target_q_values

        # Update Critic Networks
        critic1_pred = self.critic_1(s, a)
        critic2_pred = self.critic_2(s, a)

        if self.prio_replay_buffer:
            # MSE loss using weighted error
            td_error1 = q_target - critic1_pred
            td_error2 = q_target - critic2_pred
            critic1_loss = 0.5 * (td_error1 ** 2 * weights).mean()
            critic2_loss = 0.5 * (td_error2 ** 2 * weights).mean()
            priorities = abs(((td_error1 + td_error2) / 2 + 1e-5)).squeeze().detach().numpy()
            self.replay_buffer.update_priorities(inds, priorities)
        else:
            critic1_loss = self.critic_loss(critic1_pred, q_target)
            critic2_loss = self.critic_loss(critic2_pred, q_target)

        self.critic_optimizer_1.zero_grad()
        critic1_loss.backward()
        self.critic_optimizer_1.step()
        self.critic_optimizer_2.zero_grad()
        critic2_loss.backward()
        self.critic_optimizer_2.step()

        # Update Actor Network
        new_actions, log_prob, _ = self.actor.sample(s)
        q_new_actions = torch.min(
            self.critic_1(s, new_actions),
            self.critic_2(s, new_actions)
        )

        actor_loss = (self.alpha * log_prob - q_new_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.autotune:
            alpha_loss = (-self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.)

        # Soft Updates
        for target_param, param in zip(self.critic_target_1.parameters(),
                                       self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.critic_target_2.parameters(),
                                       self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return critic1_loss.item(), critic2_loss.item(), actor_loss.item(), alpha_loss.item()

    def store_transition(self, transition):
        """ Store the transition in the replay buffer """
        self.replay_buffer.add_transition(transition)
