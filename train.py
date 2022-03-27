
import pytorch_lightning as pl
import logging
from argparse import ArgumentParser
import yaml
from pl_bolts.models.rl import VanillaPolicyGradient
from models import PolicyGradient

from gym.envs.registration import register


parser = ArgumentParser()

parser.add_argument()

trainer = pl.Trainer()
parser = trainer.add_argparse_args(parser)
args = parser.parse_args()

config_file = args["config"]

try:
    with open(config_file) as file:
        logging.log(logging.INFO, f'Running with config {yaml.dump(file)}')
        config = yaml.load(file, Loader=yaml.FullLoader)

        # this dataset is the input, output tuples
        dataset = config["dataset"]
        assert dataset, "Dataset config missing."

        # RNN parameters
        rnn_config = config["rnn"]
        assert rnn_config, "Policy estimator config missing."
        rl_config = config["policy_gradient_algo"]
        assert rl_config, "General RL config missing"
except FileNotFoundError:
    logging.log(logging.INFO, f'{config_file} not found')
    raise FileNotFoundError(f'{config_file} not found')

model = PolicyGradient(config)
trainer = pl.Trainer(
    gpus=1,
    distributed_backend='dp',
    max_epochs=10000,
    early_stop_callback=False,
    val_check_interval=100
)

trainer.fit(model)

register(id='fitness-landscape',entry_point='env.FitnessLandscape:FitnessLandscape',)
vpg = VanillaPolicyGradient("CartPole-v0")
trainer.fit(vpg)

