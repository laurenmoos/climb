from torch.utils.data.dataset import T_co

import ReplayBuffer
from torch.utils.data import IterableDataset


class RLDataset(IterableDataset):
    """
    Iterable dataset which wraps the replaybuffer as it is updated
    with experiences during training.
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200):
        super(RLDataset).__init__()
        self.buffer = buffer
        self.sample_size = sample_size

    def __getitem__(self, index) -> T_co:
        pass

    def __iter__(self) -> tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
