from unittest import TestCase
from test_utils import test_linear_vm


class TestVirtualMachine(TestCase):

    def test_step(self):
        vm = test_linear_vm()
        vm.reset()
        for step in range(vm.task.sequence_length):
            vm.step(1)

    def test_reward_for_program_state(self):
        vm = test_linear_vm()

        vm.reward_for_program_state()
        self.fail()
