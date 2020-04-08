from collections import deque
from copy import deepcopy
import numpy as np


class RolloutReplayBuffer(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.rollout_memory = deque(maxlen=memory_size)
        self.reward_memory = deque(maxlen=memory_size)
        self.count_memory = deque(maxlen=memory_size)

    def insert(self, rollout, reward, count):
        self.rollout_memory.append(deepcopy(rollout))
        self.reward_memory.append(deepcopy(reward))
        self.count_memory.append(deepcopy(count))

    def recall(self, num):
        idxs = np.random.choice(len(self.rollout_memory), num)
        rollout_recalled = [self.rollout_memory[idx] for idx in idxs]
        reward_recalled = [self.reward_memory[idx] for idx in idxs]
        count_recalled = [self.count_memory[idx] for idx in idxs]

        return rollout_recalled, reward_recalled, count_recalled
