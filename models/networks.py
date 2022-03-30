import logging
from typing import Union, Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torchvision import models
from torchsummary import summary


def create_mlp(input_shape: Tuple[int], n_actions: int, hidden_sizes: list = [27, 10]):
    """
    Simple Multi-Layer Perceptron network
    """
    logging.log(logging.WARN, f"Input shap is {input_shape[0]}")
    torch.set_default_dtype(torch.float64)
    net_layers = []
    net_layers.append(nn.Linear(30, 27))
    net_layers.append(nn.ReLU())
    net_layers.append(nn.Linear(27, 10))
    net_layers.append(nn.ReLU())
    net_layers.append(nn.Linear(10, n_actions))
    mod = nn.Sequential(*net_layers).float()
    return mod


class ActorCategorical(nn.Module):
    """
    Policy network, for discrete action spaces, which returns a distribution
    and an action given an observation
    """

    def __init__(self, actor_net):
        """
        Args:
            input_shape: observation shape of the environment
            n_actions: number of discrete actions available in the environment
        """
        super().__init__()

        self.actor_net = actor_net

    def forward(self, states):
        logging.log(logging.WARN, f"State datatype before logits {states.dtype}")
        logits = self.actor_net(states.float())
        logging.log(logging.WARN, f"Logits {logits} State shape {logits.shape}")
        pi = Categorical(logits=logits)
        actions = pi.sample()

        return pi, actions

    def get_log_prob(self, pi: Categorical, actions: torch.Tensor):
        """
        Takes in a distribution and actions and returns log prob of actions
        under the distribution
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
        """
        return pi.log_prob(actions)


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
            states: current state of the environment
            device: the device used for the current batch
        Returns:
            torch dsitribution and randomly sampled action
        """
        logging.log(logging.WARN, f"State datatype {state.dtype}")

        state = state.to(device=device)

        logging.log(logging.WARN, f"state {state}")
        pi, actions = self.actor_net(state)
        log_p = self.get_log_prob(pi, actions)

        value = self.critic_net(state.float())

        return pi, actions, log_p, value

    def get_log_prob(self,
                     pi: Union[Categorical, Normal],
                     actions: torch.Tensor) -> torch.Tensor:
        """
        Takes in the current state and returns the agents policy, a sampled
        action, log probability of the action, and the value of the state
        Args:
            pi: torch distribution
            actions: actions taken by distribution
        Returns:
            log probability of the acition under pi
        """
        return self.actor_net.get_log_prob(pi, actions)
