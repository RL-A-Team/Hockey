import numpy as np

class ReplayBuffer:
        def __init__(self, buffer_size):
            self.buffer_size = buffer_size
            self.buffer = []

        def add(self, state, action, next_state, reward, done):
            experience = (state, action, next_state, reward, done)
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(experience)
            else:
                self.buffer.pop(0)
                self.buffer.append(experience)

        def get_size(self):
            return len(self.buffer)

        def sample(self, batch_size):
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[idx] for idx in indices]
            states, actions, next_states, rewards, dones = zip(*batch)
            return states, actions, next_states, rewards, dones
