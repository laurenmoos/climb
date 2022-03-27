import torch
from torch import optim, nn

from pytorch_lightning import LightningModule

import numpy as np
from env import FitnessLandscape
import scipy
from scipy import signal
from collections import OrderedDict

from data.data_models import Batch, Episode

from env import RLDataset

from torch.utils.data import DataLoader


class PolicyGradient(LightningModule):

    def __init__(self, config):
        super().__init__()
        # rl specific parameters
        self.gamma = config.gamma
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.batch_episodes = config.batch_episodes
        self.entropy_beta = config.entropy_beta
        self.avg_reward_len = config.avg_reward_len
        self.epoch_len = config.epoch_len

        # estimator specific parameters
        self.num_layers = config.num_layers
        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.gru_unit_size = config.gru_unit_size

        self.num_episodes = config.num_episodes
        self.batch_size = config.batch_size

        self.env = FitnessLandscape(config)

        self.estimator = nn.GRU(self.env.observation_shape, self.gru_unit_size, self.num_layers)

        self.total_reward = 0
        self.episode_reward = 0

        # populate buffer
        self.populate(config.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """
        Fill replay buffer with configurable number of steps
        """
        for i in range(steps):
            self.run_episode()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.estimator.network.parameters(), self.learning_rate)
        return [optimizer]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict action space probabilities for the current state given observations
        """
        self.estimator.predict(obs)

    def loss(self, batch: Batch) -> torch.Tensor:
        states, actions, rewards, dones, = batch.states, batch.actions, batch.rewards, batch.dones

        # the estimator here uses a batch that is the same size as a trajectory
        logprob = torch.log(self.estimator.predict(states))
        batch_actions = actions.reshape(len(actions), 1)
        selected_logprobs = batch.rewards * torch.gather(logprob, 1, batch_actions).squeeze()
        return -selected_logprobs.mean()

    def train_dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, )
        return dataloader

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        device = self.get_device(batch)
        total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []

        for i in range(self.batch_size):
            episode, done = self.run_episode()
            rewards, states, actions = episode.rewards, episode.states, episode.actions

            # discount rewards with discount factor
            batch_rewards.extend(rewards)
            batch_observations.extend(states)
            batch_actions.extend(actions)
            total_rewards.append(sum(rewards))

            # get a little more tense, a little tensor
            batch_rewards = torch.FloatTensor(batch_rewards)
            batch_observations = torch.FloatTensor(batch_observations)
            batch_actions = torch.LongTensor(batch_actions)

            batch_returns = self.compute_monte_carlo_returns(batch_rewards)

            normalized_batch_returns = (batch_returns - np.mean(batch_returns)) / (np.std(batch_returns) + 1e-8)

            # TODO: I think batch should have returns and not rewards
            loss = self.loss(Batch(normalized_batch_returns, batch_observations, batch_actions))

            log = {'total_reward': torch.tensor(self.total_reward).to(device),
                   'reward': torch.tensor(rewards).to(device),
                   'steps': torch.tensor(self.global_step).to(device)}

            return OrderedDict({'loss': loss, 'log': log, 'progress_bar': log})

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'

    def _compute_monte_carlo_returns(self, rewards):
        return signal.lfilter([1.], [1, -self.discount], rewards[::-1])[::-1]

    def _run_episode(self):

        state = self.env.reset()
        # states: prior tokens, actions: next token, rewards: fitness
        states, actions, rewards = [], [], []
        init_states = tuple([] for _ in range(self.num_layers))

        init_state = tuple([np.zeros((1, self.gru_unit_size)) for _ in range(self.num_layers)])

        # sample the action
        action, final_state = self.estimator.sample(state[np.newaxis, np.newaxis, :], init_state)

        # TODO: in the paper they only compute a reward for the entire sequence
        next_state, episode_reward, done, _ = self.env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(episode_reward)
        [init_states[i].append(init_state[i][0]) for i in range(self.num_layers)]

        init_states = tuple(np.array(init_states[i]) for i in range(self.num_layers))
        return Episode(np.array(states), np.array(actions), np.array(rewards), np.array(rewards), init_states)
