import torch
from gym import Env
import numpy as np
from data.data_models import Task

class Compiler:

    def __init__(self, task):
        self.task = task

    def execute(self, code, num_registers, out_registers):
        # now it is evaluating columns of instruction? not sure
        return self._execute_vec(code, num_registers, out_registers)

    def _I(self, dst, data):
        return data[abs(data) % abs(dst)]

    def _IV(self, src, src_regs):
        return src_regs[abs(src_regs) % src]

    def _evaluate_inst_vec(self, inst, R, D):
        # Add a dimension to everything
        s_regs = D if inst.src < 0 else R
        d_regs = R
        if inst.arity == 2:
            eval_op = d_regs[inst.dst, :] = inst.op.(self._IV(d_regs, inst.dst), self._IV(s_regs, inst.src))
        elif inst.arity == 1:
            eval_op = d_regs[inst.dst, :] = inst.op.(self._IV(s_regs, inst.src))
        else:
            eval_op = d_regs[inst.dst, :] = inst.op()
        return eval_op

    def _execute_vec(self, code, config, out_registers, num_registers, max_steps, make_trace):
        if config:
            max_steps = self.task.sequence_length
            out_registers = self.task.num_output_registers
            num_registers = self.task.num_input_registers
        trace_len = max(1, min(code.shape[0], max_steps))
        trace = bitarray(len(R), trace_len)
        # TODO: with trace for now  maybe just make it a json dictionary
        steps = 0
        for (pc, inst) in enumerate(code):
            if pc > max_steps:
                break
            self._evaluate_inst_vec(inst, R, D)
            if make_trace:
                trace[pc = pc] = R

            steps += 1

        return R[out_registers, :], trace




class VirtualMachine(Env):

    def __init__(self, task: Task):
        super(VirtualMachine, self).__init__()

        self.task = task

        # most naive observation is just the previous instruction
        self.observation_shape = task.instruction_shape
        self.observation_space = np.ones(self.observation_shape)

        self.action_space = task.vec_to_inst

        # optional, these would be use to constrain the sampling of actions
        self.constraints = task.constraints

        # max steps == maximum sequence length
        self.sequence_length = task.sequence_length

        # upon initialization the episode return of the environment is 0
        self.episode_reward = 0

        # initialize steps left
        self.steps_left = self.sequence_length

    def step(self, action: np.ndarray):
        episode_terminated = False

        action = self.task.inst_to_onehot(self.action_space[int(action)])

        #TODO: this is where you'll evaluate if the new instruction (sampled action) is invalid

        # TODO: what to include in the context window/observation space is a critical algorithm design decision
        self.observation_space = action
        # since the paper only rewards expressions once they are constructed, this is a dummy reward
        reward = 0
        # TODO: can function sensitivity without access to the oracle function be an extrinsic inst x inst reward?
        self.episode_reward += reward

        self.steps_left -= 1

        if not self.steps_left:
            # reinit
            self.steps_left = self.sequence_length
            episode_terminated = True

        # the next state is the next subsequence or the action that was selected from the model
        return action, self.episode_reward, episode_terminated, []

    def reset(self):
        self.episode_reward = 0

        # return the initial set of observations (vector with observation_shape number of empty
        # instructions
        return torch.empty(self.observation_shape, dtype=torch.float)

    def arity(self, inst: Inst) -> int:
        return self.task.arity[inst.op]

    #TODO: make this insitu constraints
    def semantic_intron(self, action) -> bool:
        # TODO: make this enum
        inst = Inst(action)
        return inst.op in ['and', 'or', 'mov'] and inst.src == inst.dst

    # def get_effective_indices(self, program, out_regs):
    #     active_regs = out_regs
    #     active_indices = []
    #     for (i, inst) in program[::-1]:
    #         self.semantic_intron(inst)
    #         if inst.dst in active_regs:
    #             active_indices.append(i)
    #             filter(lambda reg: reg != inst.dst, active_regs)
    #             active_regs.append(inst.dst)
    #             active_regs.append(inst.src)
    #     return active_indices[::-1]
    #
    # def strip_introns(self, code, out_regs):
    #     return self.get_effective_indices(code, out_regs)
