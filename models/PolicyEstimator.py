import pytorch_lightning
import torch
from torch import nn


class PolicyEstimator:
    # TODO: consider using built-in pytorch lightning if you don't have to customize

    def __init__(self, env):
        self.num_observations = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions),
            nn.Softmax(dim=-1)
        )

    def predict(self, observation):
        return self.network(torch.FloatTensor(observation))
