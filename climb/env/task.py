import torch
import itertools
from climb.env.data_models import Inst, semantic_intron
from torch.nn import functional as F


class Task:
    """
    Initialized only once from configuration, task parameterizes search by fixing the machine architecture, as well
    as problem specific characteristics such as max sequence length, function set etc.
    """

    def __init__(self, function_set: list, num_regs: int, num_data_regs:int, output_regs: list, dataset: str, constraints: list,
                 sequence_length:int, arity:dict):

        self.function_set = function_set
        self.num_regs = num_regs
        self.num_data_regs = num_data_regs
        self.output_regs = list(output_regs)
        self.dataset = dataset
        self.constraints = constraints
        self.instruction_shape = self.number_of_possible_insts()
        self.inst_to_vec, self.vec_to_inst = self.index_mappings()
        self.sequence_length = sequence_length
        self.arity = arity

    def index_mappings(self):
        """
        :return: mapping and reverse mapping between instructions and their one-hot encoded index
        note: this is for a very simple one-hot embedding approach
        """
        #TODO: this should be all regsiters times the number of executable regs - maybe minus introns
        a = [range(self.num_regs), range(self.num_data_regs), self.function_set]
        instructions = list(itertools.product(*a))

        l, m = {}, {}
        for idx, inst in enumerate(instructions):
            instruction = Inst(inst[0], inst[1], inst[2])
            l[instruction] = idx
            m[idx] = instruction
        return l, m

    def number_of_possible_insts(self):
        """
        :return: number of possible instructions given the machine architecture and the function set
        """
        return self.num_regs * (self.num_regs + self.num_data_regs) * len( self.function_set)

    # TODO: improve the instruction embedding
    def inst_to_onehot(self, inst_offset: int):
        """
        :return: naive one-hot encoded embedding of any possible instruction given the task
        """
        one_hot = F.one_hot(torch.tensor(inst_offset), num_classes=self.instruction_shape)
        one_hot = one_hot.type(torch.FloatTensor)
        return torch.tensor(one_hot, dtype=torch.float)

    @staticmethod
    def constraint(action):
        """
        :return: is a given action (instruction) is an intron
        this is in an in-situ constraint to produce programs with only meaningful instructions
        for now: ones that do not contain instructions that are semantic introns
        """
        return not semantic_intron(action)
