import numpy as np
import random as rand
from collections import deque


class Memory:

    def __init__(self, size, batch_size):
        self.batch_size = params.batch_size
        self.memory = deque(maxlen=size)

    def add(self, action, reward, screen, terminal):
        if len(self.memory) >= self.memory.maxlen:
            self.memory.popleft()
        self.memory.append( (action, reward, screen, terminal) )

    def getSample(self):
        return rand.sample(self.memory, self.batch_size)

    def reset(self):
        self.memory.clear()