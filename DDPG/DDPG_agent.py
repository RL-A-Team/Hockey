import torch
import torch.optim as optim
import numpy as np
from models import Actor, Critic
from memory import ReplayBuffer

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_size_actor=64, hidden_size_critic=64, buffer_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(state_dim, action_dim, hidden_size_actor).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size_actor).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)

        self.critic = Critic(state_dim, action_dim, hidden_size_critic).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size_critic).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size)
        self.loss_fn = nn.MSELoss()

    def load_mode(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def get_action(self, state, epsilon, obs_noice = 0.1, evaluate=False, action_space = None):
        self.actor.eval()
        state = torch.FloatTensor(state).to(self.device)
        if (np.random.random() < epsilon and not evaluate):
            action = action_space.sample()[:4]
        else:
            with torch.no_grad():
                action = self.actor(state).cpu().numpy() + epsilon * np.random.normal(0, obs_noice)
        return action

    def store_experience(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)


    def train(self, batch_size, gamma=0.99, tau=0.001):
        self.actor.train()
        losses = []
        if (self.replay_buffer.get_size() >= batch_size):
            losses = self.update(batch_size, gamma, tau)
        return losses

    def update(self,  batch_size, gamma, tau):
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(batch_size)
        # Convert to PyTorch tensors
        states = torch.tensor(np.asarray(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.asarray(actions), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(np.asarray(dones), dtype=torch.float32).unsqueeze(1).to(self.device)

        # Critic loss
        self.critic_optimizer.zero_grad()
        target_Q = rewards + gamma * (1 - dones) * self.critic_target(next_states, self.actor_target(next_states))
        current_Q = self.critic(states, actions)
        critic_loss = self.loss_fn(current_Q, target_Q)

        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        self.actor_optimizer.zero_grad()
        qvals = -self.critic(states, self.actor(states))
        actor_loss = torch.mean(qvals)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()
