import sys
sys.path.append("/Users/lauren/climb")
import pytorch_lightning as pl
from argparse import ArgumentParser
import yaml
from models.policy_gradient import PolicyGradient
import os


def load_config(config_file: str) -> dict:
    try:
        with open(config_file) as file:
            # TODO: eventually just use args as yaml is kind of overkill
            run_config = yaml.safe_load(file)

            task_config = run_config["task"]["dataset"]
            assert task_config, "Task config missing."

            rl_config = run_config["policy_gradient_algo"]
            assert rl_config, "General RL config missing"

            return run_config
    except FileNotFoundError:
        raise FileNotFoundError(f'{config_file} not found')


if __name__ == "__main__":
    # parse args
    parser = ArgumentParser()

    parser.add_argument('--config')

    trainer = pl.Trainer()
    parser = trainer.add_argparse_args(parser)
    args = parser.parse_args()

    config = load_config(os.getcwd() + '/' + args.config)

    # TODO: have a switch statement here to run model or a benchmark
    model = PolicyGradient(config)
    trainer = pl.Trainer(
        gpus=0,
        max_epochs=1000
    )

    trainer.fit(model)
