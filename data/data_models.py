import random

import numpy as np
import torch
import constants
from dataclasses import dataclass
from torch.nn import functional as F


@dataclass
class Inst:
    op: str
    arity: int
    # let's use the convention that negative indices refer to input
    dst: int
    src: int

    def to_vec(self):
        ops_vec = F.one_hot(torch.tensor(constants.FUNCTION_SET.index(self.op)), num_classes=8)
        src_vec = F.one_hot(torch.tensor(self.src))
        dst_vec = F.one_hot(torch.tensor(self.dst))
        return torch.cat([ops_vec, src_vec, dst_vec], dim=0)


@dataclass
class Program:
    instructions: list

    def to_vec(self):
        torch.cat(list(map(lambda inst: inst.to_vec(), self.instructions)), dim=0)


@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray
    # let's use the convention that negative indices refer to input
    rewards: np.ndarray
    returns: np.ndarray
    initial_states: tuple


@dataclass
class Batch:
    #TODO: might ultimately better to make this a managed collection of Episodes not Summary Statistics
    normalized_batch_returns: torch.Tensor
    batch_observations: torch.Tensor
    # let's use the convention that negative indices refer to input
    batch_actions: torch.Tensor

    # TODO: this will eventually be weighed by reward not sampled at random
    def sample(self, batch_size: int) -> tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.bool), np.array(next_states))
