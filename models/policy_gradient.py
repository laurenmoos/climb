import torch

import pytorch_lightning as pl
import torch.optim as optim

import os
from env.experience_source_dataset import ExperienceSourceDataset
from models.networks import create_mlp, ActorCriticAgent, ActorCategorical
from data.data_models import Task
from env.fitness_landscape import FitnessLandscape

from env.cockatrice import evaluate

import pandas as pd

from torch.utils.data import DataLoader


class PolicyGradient(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        # rl hyper-parameters
        rl_config = config["policy_gradient_algo"]
        self.gamma = rl_config['gamma']
        self.lr = rl_config['lr']
        self.batch_size = rl_config['batch_size']
        self.batch_episodes = rl_config['batch_episodes']
        self.entropy_beta = rl_config['entropy_beta']
        self.avg_reward_len = rl_config['avg_reward_len']
        self.epoch_len = rl_config['epoch_len']
        # RL control flow parameters
        self.max_episode_len = rl_config['num_episodes']
        self.steps_per_epoch = rl_config['batch_size']

        self.lr_actor = 3e-4
        self.lr_critic = 1e-3

        # estimator specific parameters
        policy_config = rl_config['policy_estimator']

        self.num_layers = policy_config['num_layers']
        self.sequence_length = policy_config['sequence_length']
        self.learning_rate = policy_config['learning_rate']
        self.gru_unit_size = policy_config['gru_unit_size']

        self.save_hyperparameters()
        # task specific parameters and data loading
        self.task_config = config["task"]
        self.n_inp_reg = self.task_config["num_input_registers"]
        self.n_out_reg = self.task_config["num_output_registers"]
        self.function_set = self.task_config["function_set"]
        self.arity = self.task_config["arity"]
        self.dataset = self.task_config["dataset"]
        self.constraints = self.task_config["constraints"]
        self.task = Task(self.function_set, self.n_inp_reg, self.n_out_reg, self.dataset, self.constraints)
        #TODO: will be interesting experiment if there is some kind of diversity metric or incremental reward
        self.env = FitnessLandscape(self.task)

        # TODO: while this will likely remain an MLP, it deserves a bit more thought
        input_shape = (self.task.instruction_shape,)
        self.critic = create_mlp(input_shape, len(self.env.action_space))

        # TODO: replace this with a recurrent policy model
        input_shape = (self.task.instruction_shape,)
        actor_model = create_mlp(input_shape, len(self.env.action_space))
        self.actor = ActorCategorical(actor_model)

        self.agent = ActorCriticAgent(self.actor, self.critic)

        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = torch.tensor(self.env.reset(), dtype=float)

    def configure_optimizers(self) -> tuple:
        """ Initialize Adam optimizer"""
        optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        """
        Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic
        for each data sample.
        """
        for i in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Predict action space probabilities for the current state given observations
        """
        pi, action = self.actor(x)
        value = self.critic(x)

        return pi, action, value

    def calc_advantage(self, rewards: list, values: list, last_value: float) -> list:
        """Calculate the advantage given rewards, state values, and the last value of episode
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def _compute_episode_rewards(self, candidate_expression):
        """
        computes the normalized mean square root error on the configured dataset
        """
        # this is where you evaluate the expression on the tuples
        df = pd.read_csv(os.getcwd() + "/" + self.dataset)

        input_registers, output_registers = df[df.columns[:self.n_inp_reg]], df[df.columns[:-self.n_out_reg]]

        assert len(input_registers) == len(output_registers)

        error = 0
        for inp, out in zip(input_registers, output_registers):
            error += evaluate(candidate_expression, inp, out)

        return 1 - error

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = ExperienceSourceDataset(self.train_batch)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def train_batch(self, ) -> tuple:
        """
        Contains the logic for generating trajectory data to train policy and value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in range(self.steps_per_epoch):
            pi, action, log_prob, value = self.agent(self.state.float(), self.device)

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)

            # TODO: I believe this should be the product of all of the values of the episode
            self.ep_values.append(torch.max(value))
            self.state = torch.FloatTensor(next_state)

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    with torch.no_grad():
                        _, _, _, value = self.agent(self.state, self.device)
                        last_value = value
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self._compute_episode_rewards(last_value)
                # advantage
                self.batch_adv += self.calc_advantage(self.ep_rewards, self.ep_values, last_value)
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = torch.FloatTensor(self.env.reset())

            if epoch_end:
                train_data = zip(
                    self.batch_states, self.batch_actions, self.batch_logp,
                    self.batch_qvals, self.batch_adv)

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (self.steps_per_epoch - steps_before_cutoff) / nb_episodes

                self.epoch_rewards.clear()

    def actor_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        pi, _ = self.actor(state)
        logp = self.actor.get_log_prob(pi, action)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state, action, logp_old, qval, adv) -> torch.Tensor:
        value = self.critic(state)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(self, batch: tuple, batch_idx, optimizer_idx):
        """
        updates actor and critic network via a batch from the replay buffer
        """
        state, action, old_logp, qval, adv = batch

        # normalize advantages
        adv = (adv - adv.mean()) / adv.std()

        self.log("avg_ep_len", self.avg_ep_len, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_ep_reward", self.avg_ep_reward, prog_bar=True, on_step=False, on_epoch=True)
        self.log("avg_reward", self.avg_reward, prog_bar=True, on_step=False, on_epoch=True)

        if optimizer_idx == 0:
            loss_actor = self.actor_loss(state, action, old_logp, qval, adv)
            self.log('loss_actor', loss_actor, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            return loss_actor

        elif optimizer_idx == 1:
            loss_critic = self.critic_loss(state, action, old_logp, qval, adv)
            self.log('loss_critic', loss_critic, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            return loss_critic

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'
