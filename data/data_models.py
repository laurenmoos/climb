import random
from collections import namedtuple


import numpy as np
import torch
import data.constants
from dataclasses import dataclass
from torch.nn import functional as F


Task = namedtuple('Point', 'function_set arity num_input_registers num_output_registers dataset constraints')

@dataclass
class Task:
    function_set:list
    arity: dict
    num_input_registers: int
    num_output_registers: int
    dataset: str
    constraints: list

    def instruction_shape(self) -> int:
        return self.num_input_registers + self.num_output_registers + len(self.function_set)

@dataclass
class Inst:
    op: str
    arity: int
    # let's use the convention that negative indices refer to input
    dst: int
    src: int

    def to_vec(self, task: Task):
        ops_vec = F.one_hot(torch.tensor(task.function_set.index(self.op)), num_classes=8)
        src_vec = F.one_hot(torch.tensor(self.src), num_classes=task.num_input_registers)
        dst_vec = F.one_hot(torch.tensor(self.dst), num_classes=task.num_output_registers)

        return torch.cat([ops_vec, src_vec, dst_vec], dim=0)

    @staticmethod
    def to_vec(task:Task, src: int, dst: int, op:str):
        ops_vec = F.one_hot(torch.tensor(task.function_set.index(op)), num_classes=8)
        src_vec = F.one_hot(torch.tensor(src), num_classes=task.num_input_registers)
        dst_vec = F.one_hot(torch.tensor(dst), num_classes=task.num_output_registers)

        return torch.cat([ops_vec, src_vec, dst_vec], dim=0)

@dataclass
class Program:
    instructions: list


@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray
    # let's use the convention that negative indices refer to input
    rewards: np.ndarray
    undiscounted_return: int
    initial_states: tuple
    program: Program


