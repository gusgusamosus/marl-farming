import random

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size

    def push(self, episode):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(episode)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
