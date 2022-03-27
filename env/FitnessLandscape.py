import gym
from gym import Env
import numpy as np
import cockatrice


class FitnessLandscape(Env):
    # implement step with acccess to the fitness function, possibly can call augmented batch

    def __init__(self, config):
        super(FitnessLandscape, self).__init__()

        # note this can be customized with observation shape consisting of
        # something analogous to the parent and sibling node of the instruction

        # presuming we send the parent and the sibling of the instruction via the
        # symbolic expression tree
        self.observation_shape = (2, 3)
        self.observation_space = np.array(([[1, 1, 1], [1, 1, 1]]))

        self.action_space = config.function_set

        # optional, these would be use to constrain the sampling of actions
        self.constraints = config.constraints

        # max steps == maximum sequence length
        self.sequence_length = config['sequence_length']

        # upon initialization the episode return of the environment is 0
        self.episode_reward = 0

    def step(self, action):
        episode_terminated = False

        assert self.action_space.contains(action), "Invalid Action"

        steps_left = self.sequence_length

        sub_sequence = np.concatenate((self.observation_space, action), axis=0)
        # this implementation computes reward for each sub-sequence, not what is done in paper
        reward = cockatrice.evaluate(sub_sequence)
        self.episode_reward += reward

        steps_left -= 1
        if not steps_left:
            episode_terminated = True

        return sub_sequence, self.episode_reward, episode_terminated, []

    def reset(self):
        self.episode_reward = 0

        # return the initial set of observations (vector with observation_shape number of empty
        # instructions
        return np.empty(self.observation_shape)

    def render(self, mode="human"):
        # TODO: add visualization?
        pass

    def close(self):
        pass
