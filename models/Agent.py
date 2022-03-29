
import torch

import numpy as np
from data.data_models import Episode
from env.cockatrice import evaluate


class Agent:
    """
    Encapsulates interactions with the environment.
    """

    def __init__(self, env, device, estimator, config):

        self.state = None
        self.env = env

        self.device = device

        self.estimator = estimator

        self.max_sequence_length = config['sequence_length']
        self.num_layers = config['num_layers']
        self.gru_unit_size = config['gru_unit_size']

        self.reset()

    def reset(self):
        """ resets environment and updates state """
        self.state = self.env.reset()

    def get_action(self, init_state):
        # TODO: action should be a tuple that contains an action and an action probability
        action, final_state = self.estimator.sample(self.state[np.newaxis, np.newaxis, :], init_state)

        return action

    @torch.no_grad()
    def run_episode(self):
        """
        Generates a single program via sampling from the action space
        L of possible instructions via forward pass until max sequence length is reached.
        """

        # states: prior tokens, actions: next token, rewards: fitness
        states, actions, rewards = [], [], []
        init_states = tuple([] for _ in range(self.num_layers))

        init_state = tuple([np.zeros((1, self.gru_unit_size)) for _ in range(self.num_layers)])

        for i in range(self.max_sequence_length):

            action = self.get_action(init_state)
            # TODO: in the paper they only compute a reward for the entire sequence
            next_state, episode_reward, done, _ = self.env.step(action)

            states.append(self.state)
            actions.append(action)
            rewards.append(episode_reward)
            [np.concatenate(init_states[i], init_state[i][0], axis=0) for i in range(self.num_layers)]

            init_states = tuple(np.array(init_states[i]) for i in range(self.num_layers))

        # the terminal state will be the completed program
        candidate_expression = states[-1]

        undiscounted_return = evaluate(candidate_expression)
        # TODO: we are only concerned with the un-discounted total return of an episode
        return Episode(np.array(states), np.array(actions), np.array(rewards), undiscounted_return, init_states), done
