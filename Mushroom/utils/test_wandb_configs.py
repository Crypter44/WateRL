import pprint
import time

from Mushroom.utils.wandb_handler import build_param_product, create_updated_config, wandb_training

base_config = {
    "n_epochs": 100,
    "actor": {
        "n_features": 64,
        "lr": 0.001,
        "activation": "tanh"
    }
}

params = {
    "n_epochs": [10, 20],
    "actor.lr": [0.0001, 0.001],
    "seed": [1, 2]
}


def test_create_configs():
    configs = build_param_product(params)
    print()
    for c in configs:
        pprint.pprint(create_updated_config(base_config, c))


def test_wandb_training():
    print()

    def train(run, path):
        print(f"Training {run.name}")
        time.sleep(5)
        print(f"Finished training {run.name}")

    wandb_training(
        project="test_wandb_training",
        group="TestWithNewConfig",
        base_config=base_config,
        params=params,
        train=train,
        base_path="."
    )

