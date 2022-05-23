from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical


def get_log_prob(pi: Categorical, actions: torch.Tensor):
    """
    Args:
        pi: torch distribution
        actions: actions taken by distribution
    Returns:
        log probability of the action under pi
    """
    return pi.log_prob(actions)


def create_mlp(input_shape: Tuple[int], n_actions: int, hidden_sizes: list = [128, 128]):
    """
    Simple Multi-Layer Perceptron network
    """
    net_layers = [nn.Linear(input_shape[0], hidden_sizes[0]), nn.ReLU()]

    for i in range(len(hidden_sizes) - 1):
        net_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        net_layers.append(nn.ReLU())
    net_layers.append(nn.Linear(hidden_sizes[-1], n_actions))

    return nn.Sequential(*net_layers)


class ActorCategorical(nn.Module):
    """
    Policy network, for discrete action spaces, which returns a distribution
    and an action given an observation
    """
    def __init__(self, actor_net):
        super().__init__()

        self.actor_net = actor_net

    def forward(self, states):
        logits = self.actor_net(states.float())
        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions


class ActorCriticAgent(object):
    """
    Actor Critic Agent used during trajectory collection. It returns a
    distribution and an action given an observation. Agent based on the
    implementations found here: https://github.com/Shmuma/ptan/blob/master/ptan/agent.py
    """

    def __init__(self, actor_net: nn.Module, critic_net: nn.Module):
        self.actor_net = actor_net
        self.critic_net = critic_net

    @torch.no_grad()
    def __call__(self, state: torch.Tensor, device: str) -> Tuple:
        """
        Takes in the current state and returns the agents policy, sampled
        action, log probability of the action, and value of the given state
        Args:
            state: current state of the environment
            device: the device used for the current batch
        Returns:
            torch dsitribution and randomly sampled action
        """

        state = state.to(device=device)

        pi, actions = self.actor_net(state)
        log_p = get_log_prob(pi, actions)

        value = self.critic_net(state.float())

        return pi, actions, log_p, value
