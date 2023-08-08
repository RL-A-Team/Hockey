import numpy as np

class ReplayBuffer:
        def __init__(self, buffer_size):
            self.max_buffer_size = buffer_size
            self.buffer = []
            self.buffer_size = 0

        def add(self, state, action, next_state, reward, done):
            experience = (state, action, next_state, reward, done)
            if len(self.buffer) < self.max_buffer_size:
                self.buffer.append(experience)
            else:
                self.buffer.pop(0)
                self.buffer.append(experience)
            self.buffer_size += 1

        def get_size(self):
            return self.buffer_size

        def sample(self, batch_size):
            if batch_size > self.buffer_size:
                batch_size = self.buffer_size
            indices = np.random.choice(self.buffer_size, batch_size, replace=False)
            batch = [self.buffer[idx] for idx in indices]
            states, actions, next_states, rewards, dones = zip(*batch)
            return states, actions, next_states, rewards, dones
