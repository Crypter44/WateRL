import itertools
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any

import wandb


def wandb_training(
        project: str,
        group: str,
        base_config: dict,
        params: dict | list[dict],
        train: callable,
        base_path: str,
        **wandb_args
):
    api = wandb.Api()
    past_runs = api.runs(path=f'bt-fluidnetop/{project}', order='-created_at')
    try:
        latest_run = past_runs[0]
        job_counter = latest_run.config.get('job_counter', -1) + 1
    except IndexError:
        job_counter = 0
    base_path += str(job_counter)

    if type(params) is dict:
        param_sets = build_param_product(params)
    elif type(params) is list:
        param_sets = params
    else:
        raise ValueError("params must be either a dict or a list of dicts.")

    for i, ps in enumerate(param_sets):
        name = "-".join([str(v) for v in ps.values()])
        config = create_updated_config(base_config, ps)
        update_param_in_config(config, "automated_training_params", ps)
        update_param_in_config(config, "job_counter", job_counter)

        path = base_path + f"/{name}/"
        os.makedirs(path, exist_ok=True)

        print(f"Creating run {i + 1}/{len(param_sets)} with name {name} in job {job_counter}.")
        run = wandb.init(
            project=project,
            group=group,
            config=config,
            dir=f"{base_path}",
            name=name,
            **wandb_args
        )

        start = datetime.now()
        train(run, path)
        end = datetime.now()

        run.alert(
            f"Finished training run {i + 1}/{len(param_sets)}!",
            f"In job \'{job_counter}\' the run \'{run.name}\' is done. "
            f"In total {i + 1} / {len(param_sets)} runs are completed.\n"
            f"This run took: {str(end - start).split('.')[0]}\n"
            f"Estimated time until completion of all runs: "
            f"{str(((end - start) * (len(param_sets) - (i+1)))).split('.')[0]}",
            level="INFO"
        )

        # save run config dict to json file
        with open(f"{path}/config.json", "w") as f:
            json.dump(run.config.as_dict(), f, indent=4)

        run.finish()


def build_param_product(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    param_sets = []
    param_names = list(params.keys())
    for combination in itertools.product(*params.values()):
        param_sets.append(dict(zip(param_names, combination)))
    return param_sets


def create_updated_config(base_config: dict, params: dict) -> dict:
    config = deepcopy(base_config)
    for param, value in params.items():
        config = update_param_in_config(config, param, value)
    return config


def update_param_in_config(config: dict, param_to_update: str, value) -> dict:
    keys = param_to_update.split(".")
    param = config
    for key in keys[:-1]:
        param = param[key]

    param[keys[-1]] = value
    return config


def create_log_dict(agents, mdp, score):
    log = {
        "score/score": score[2],
        "score/min_score": score[0],
        "score/max_score": score[1],
    }
    for i, a in enumerate(agents):
        log |= {
            f"agent_{i}/{key}": value for key, value in a.get_debug_info(entries_as_list=False).items()
        }
        if hasattr(a, "policy") and hasattr(a.policy, "get_sigma"):
            log |= {
                f"agent_{i}/sigma": a.policy.get_sigma()
            }
    log |= {
        f"mdp/eval/{key}": value for key, value in mdp.get_debug_info().items()
    }
    return log
