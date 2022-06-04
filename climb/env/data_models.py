from climb.env.constants import op_string_to_ops, OPSTRING, arity_dict


def semantic_intron(inst) -> bool:
    # TODO: alternative construction for an action string
    return inst.op in [OPSTRING.AND, OPSTRING.OR, OPSTRING.MOV] and inst.src == inst.dst


# TODO: need another test_data model candidate expression that encapsulates atttributes and function logic
class Op:

    def __init__(self, op_str, fx, arity):
        self.op_str = op_str
        self.fx = fx
        self.arity = arity

    @staticmethod
    def from_string(op_str):
        op_str = OPSTRING(op_str)
        return Op(op_str, op_string_to_ops[op_str], arity_dict[op_str])


class Inst:

    def __init__(self, src: int, dst: int, op: str):
        self.src = src
        self.dst = dst
        self.op = Op.from_string(op)

    @staticmethod
    def from_string(inst_str):
        """
        :param inst_str: in polish notation
        :return: Inst instance
        """
        tokens = inst_str.split(" ")
        # Lucca uses Hungarian notation
        return Inst(int(tokens[2]), int(tokens[1]), tokens[0])

    def __eq__(self, other):
        return self.op == other.op \
               and self.op.arity == other.arity_dict \
               and self.dst == other.dst \
               and self.src == other.src

    def __str__(self):
        """
        :return: string representation of instructions in polish notation
        """
        pr_print = {"OP:": self.op.op_str, "SRC": self.src, "DST": self.dst}
        return str(pr_print)

    def __hash__(self):
        return hash((self.op, self.src, self.dst))


class CandidateExpression:
    """
    Encapsulating the representations of a candidate expression.
    code: constructor initialized with an episode of instructions sampled from the policy.
    """

    def __init__(self, code: list):
        self.code = code

    # TODO: decompilation function

    @staticmethod
    def from_string(program_str, delimiter=';'):
        """
        :param delimiter: configurable with default ;
        :param program_str: polish notation
        :return: candidate expression
        """
        #TODO: add an exception as this is a pretty brittle way of doing things
        insts = program_str.split(delimiter)

        code = []
        for inst_str in insts:
            code.append(Inst.from_string(inst_str))
        return code

    def __str__(self):
        """
        :return: string representation of instructions in polish notation
        """
        inst_strs = []
        for inst in self.code:
            inst_strs.append(str(inst))

        return ''.join(inst_strs)
