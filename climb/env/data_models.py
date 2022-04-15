from constants import op_string_to_ops, OPSTRING, arity


# TODO: make this insitu constraints
def semantic_intron(self, action) -> bool:
    # TODO: alternative construction for an action string
    inst = Inst.from_string(action)
    return inst.op in [OPSTRING.AND, OPSTRING.OR, OPSTRING.MOV] and inst.src == inst.dst


# TODO: need another test_data model candidate expression that encapsulates atttributes and function logic
class Op:

    def __init__(self, program_str, fx, arity):
        self.program_str = program_str
        self.fx = fx
        self.arity = arity


class Inst:

    def __init__(self, src: int, dst: int, op: Op):
        self.src = src
        self.dst = dst
        self.op = op

    @staticmethod
    def from_string(str):
        """
        :param str: in polish notation
        :return: Inst instance
        """
        tokens = str.split(" ")
        op_str = OPSTRING(tokens[0])
        op = Op(op_str, op_string_to_ops[op_str], arity[op_str])
        # Lucca uses Hungarian notation
        return Inst(int(tokens[2]), int(tokens[1]), op)

    # TODO: ideally task would encapsulate any constructors for instruction
    # TODO create an instructor that parses a string of src, dst, op
    def __eq__(self, other):
        return self.op == other.op and self.op.arity == other.arity and self.dst == other.dst and self.src == other.src

    # TODO: this appears to be a to string method

    def reg_type(self, x):
        return 'D' if x < 0 else 'R'


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
            inst_strs.append(''.join([str(inst.op), str(inst.src), str(inst.dst)]))

        return ''.join(inst_strs)
