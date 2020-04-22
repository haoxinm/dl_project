from collections import deque
from copy import deepcopy
import numpy as np


class RolloutReplayBuffer(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.rollout_memory = deque(maxlen=memory_size)
        # self.stat_memory = deque(maxlen=memory_size)

    def insert(self, rollout):
        rollout = deepcopy(rollout)
        rollout.to('cpu')
        self.rollout_memory.append(rollout)
        # cpu_stat = dict()
        # for key, value in stat.items():
        #     cpu_stat[key] = value.to('cpu')
        # self.stat_memory.append(cpu_stat)

    def recall(self, num):
        idxs = np.random.choice(len(self.rollout_memory), num)
        rollout_recalled = [self.rollout_memory[idx] for idx in idxs]
        # stat_recalled = [self.stat_memory[idx] for idx in idxs]

        return rollout_recalled

    def __len__(self):
        return len(self.rollout_memory)
