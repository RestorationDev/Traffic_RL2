from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return(
            np.float32(states),
            np.int64(actions),
            np.float32(rewards),
            np.float32(next_states),
            np.float32(dones),
        )
    def __len__(self):
        return len(self.buffer)