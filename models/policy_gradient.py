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


# TODO: metric design - batch diversity, buffer diversity, if autoregressive context window diversity esp for critic
class PolicyGradient(pl.LightningModule):

    def __init__(self, config):
        # TODO: as you experiment, replace these hyperparameters with defaults to pair down a bit
        super().__init__()
        # rl hyper-parameters
        rl_config = config["policy_gradient_algo"]
        self.gamma = rl_config['gamma']
        self.lr = rl_config['lr']
        self.batch_size = rl_config['batch_size']
        self.entropy_beta = rl_config['entropy_beta']
        self.avg_reward_len = rl_config['avg_reward_len']
        self.epoch_len = rl_config['epoch_len']
        self.nb_optim_iters = rl_config['nb_optim_iters']
        self.clip_ratio = rl_config['clip_ratio']
        # RL control flow parameters

        self.lr_actor = rl_config['lr_actor']
        self.lr_critic = rl_config['lr_critic']

        # task specific parameters and data loading
        self.task_config = config["task"]
        self.n_inp_reg = self.task_config["num_input_registers"]
        self.n_out_reg = self.task_config["num_output_registers"]
        self.function_set = self.task_config["function_set"]
        self.arity = self.task_config["arity"]
        self.dataset = self.task_config["dataset"]
        self.constraints = self.task_config["constraints"]
        # An episode is the construction of a vector of instructions / candidate program
        # for now setting epoch == episode, but might change for recurrent policy network
        self.sequence_length = self.task_config["sequence_length"]
        self.steps_per_epoch = self.sequence_length

        self.save_hyperparameters()
        self.task = Task(self.function_set, self.n_inp_reg, self.n_out_reg, self.dataset, self.constraints,
                         self.sequence_length)
        # TODO: will be interesting experiment if there is some kind of diversity metric or incremental reward
        self.env = FitnessLandscape(self.task)

        # TODO: while this will likely remain an MLP, it deserves a bit more thought
        input_shape = (self.task.instruction_shape,)
        # critic sampling a random action from the distribution formed by the inner state of the (currently) MLP model
        self.critic = create_mlp(input_shape, 1)

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

        self.state = self.env.reset()

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

    # setting default to no discount as the reward isn't discounted in symbolic optimization paper
    # TODO: leaving method as is in case we end up calculating a relevant incremental reward
    def discount_rewards(self, rewards, discount=1.0):
        """Calculate the discounted rewards of all rewards in list
        Args:
            rewards: list of rewards/advantages
        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: list, values: list, last_value: float) -> list:
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta)

        return adv

    def _compute_episode_rewards(self, candidate_expression):
        """
        computes the normalized mean square root error on the configured dataset
        """
        # this is where you evaluate the expression on the tuples
        df = pd.read_csv(os.getcwd() + "/" + self.dataset)

        input_registers, output_registers = df[df.columns[:self.n_inp_reg]], df[df.columns[:-self.n_out_reg]]

        assert len(input_registers) == len(output_registers)

        cum_error = []
        for inp, out in zip(input_registers, output_registers):
            cum_error.append(evaluate(candidate_expression, inp, out))

        return cum_error

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
        train an actor/critic agent on a batch of candidate programs evaluated for fit on a sample of input/output tuples
        """
        for step in range(self.steps_per_epoch):
            pi, action, log_prob, value = self.agent(self.state.float(), self.device)

            next_state, reward, done, _ = self.env.step(action.cpu().numpy())
            self.episode_step += 1

            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)

            # value is a learned sample via the critic network so we can expect a float scalar here
            self.ep_values.append(value.item())
            self.state = torch.FloatTensor(next_state)

            epoch_end = step == (self.steps_per_epoch - 1)
            # has a reward been collected for each instruction added to the sequence
            terminal = len(self.ep_rewards) == self.sequence_length

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

                # cumulative reward
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
