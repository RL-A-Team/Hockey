import torch
import numpy as np
import gymnasium as gym
import pickle
from ReplayBuffer import ReplayBuffer
from Noice import OUNoise
from models import Feedforward, QFunction

class DDPGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, action_dim ,gamma, tau,
                 hidden_size_actor=[128, 128], lr_actor=0.0001, hidden_size_critic=[128, 128, 64], lr_critic=0.0001,
                 update_target_every = 100):

        self.observation_space = observation_space
        self.obs_dim=self.observation_space.shape[0]
        self.action_space = action_space
        self.action_n = int(action_dim/2)

        self.action_noise = OUNoise((self.action_n))

        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer(buffer_size=10e6)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # critic Network
        self.critic = QFunction(observation_dim=self.obs_dim,
                           action_dim=self.action_n,
                           hidden_sizes= hidden_size_critic).to(self.device)
        # target critic Network
        self.critic_target = QFunction(observation_dim=self.obs_dim,
                                  action_dim=self.action_n,
                                  hidden_sizes= hidden_size_critic).to(self.device)
        # actor Network
        self.actor = Feedforward(input_size=self.obs_dim,
                                  hidden_sizes= hidden_size_actor,
                                  output_size=self.action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh()).to(self.device)
        # target actor Network
        self.actor_target = Feedforward(input_size=self.obs_dim,
                                         hidden_sizes= hidden_size_actor,
                                         output_size=self.action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh()).to(self.device)
        self.hard_target_update()

        self.optimizer=torch.optim.Adam(self.actor.parameters(),
                                        lr=lr_actor,
                                        eps=0.000001)

        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),
                                               lr=lr_critic)

        self.loss = torch.nn.SmoothL1Loss()

        self.train_iter = 0
        self.update_target_every = update_target_every

    def hard_target_update(self):
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())

    def soft_target_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def act(self, observation, eps=None):
        if eps is None:
            eps = 0
        observation = torch.FloatTensor(observation).to(self.device)
        action = self.actor.predict(observation) + eps*self.action_noise()  # action in -1 to 1 (+ noise)
        action = self.action_space.low[:4] + (action + 1.0) / 2.0 * self.action_space.high[:4] - self.action_space.low[:4]
        return action

    def load_model(self, path_actor, path_critic=None):
        self.actor.load_state_dict(torch.load(path_actor, map_location=self.device))
        if path_critic is not None:
            self.critic.load_state_dict(torch.load(path_critic, map_location=self.device))
        self.hard_target_update()

    def store_transition(self, ob, a, ob_new, reward, done):
        self.replay_buffer.add(ob, a, ob_new, reward, done)

    def state(self):
        return (self.critic.state_dict(), self.actor.state_dict())

    def restore_state(self, state):
        self.critic.load_state_dict(state[0])
        self.actor.load_state_dict(state[1])
        self.hard_target_update()

    def reset(self):
        self.action_noise.reset()

    def train(self, batch_size, iter_fit=32):
        actor_losses = []
        critic_losses = []
        self.train_iter+=1
        if self.train_iter % self.update_target_every == 0:
            self.soft_target_update()
        for i in range(iter_fit):

            states, actions, next_states, rewards, dones = self.replay_buffer.sample(batch_size)
            # Convert to PyTorch tensors
            states = torch.tensor(np.asarray(states), dtype=torch.float32).to(self.device)
            actions = torch.tensor(np.asarray(actions), dtype=torch.float32).to(self.device)
            next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32).to(self.device)
            rewards = torch.tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
            dones = torch.tensor(np.asarray(dones), dtype=torch.float32).unsqueeze(1).to(self.device)

            # optimize the Q objective
            
            q_next = self.critic_target.Q_value(next_states, self.actor_target.forward(next_states.to(self.device)))
            td_targets = rewards + self.gamma * (1.0-dones) * q_next
            self.critic.train()  # put model in training mode
            self.critic_optimizer.zero_grad()
            pred = self.critic.Q_value(states, actions)
            critic_loss = self.loss(pred, td_targets)
            critic_loss.backward()
            self.critic_optimizer.step()

            # optimize actor objective
            self.optimizer.zero_grad()
            self.actor.train()
            q = self.critic.Q_value(states, self.actor.forward(states))
            actor_loss = -torch.mean(q)
            actor_loss.backward()
            self.optimizer.step()

            critic_losses.append(critic_loss.item())
            actor_losses.append(actor_loss.item())

        return critic_losses, actor_losses
