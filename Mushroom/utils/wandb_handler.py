import itertools
import json
import os
from copy import deepcopy
from datetime import datetime
import multiprocessing as mp
from typing import Any

import wandb


def wandb_training(
        project: str,
        group: str,
        base_config: dict,
        params: dict | list[dict],
        train: callable,
        base_path: str,
        time_limit_in_sec: int | None = None,
        inactivity_timeout: int | None = None,
        **wandb_args
):
    api = wandb.Api()
    past_runs = api.runs(path=f'bt-fluidnetop/{project}', order='-config.job_counter.value')
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

    main_run = wandb.init(
        project=project,
        group=group,
        config=base_config | {"job_counter": job_counter},
        dir=f"{base_path}",
        name=f"main-{job_counter}",
        **wandb_args
    )

    for i, ps in enumerate(param_sets):
        name = "-".join([str(v) for v in ps.values()])
        config = create_updated_config(base_config, ps)
        update_param_in_config(config, "automated_training_params", ps)
        update_param_in_config(config, "job_counter", job_counter)

        path = base_path + f"/{name}/"
        os.makedirs(path, exist_ok=True)
        # save run config dict to json file

        with open(path + "config.json", "w") as f:
            json.dump(config, f, indent=4)

        def target(shared_dict, **kwargs):
            print(f"Creating run {i + 1}/{len(param_sets)} with name {name} in job {job_counter}.")
            run = wandb.init(
                project=project,
                group=group,
                config=config,
                dir=f"{base_path}",
                name=name,
                **wandb_args
            )
            shared_dict['run_id'] = run.id
            train(run, path, **kwargs)

        with mp.Manager() as manager:
            shared_dict = manager.dict()

            if inactivity_timeout is None:
                p = mp.Process(target=target, args=(shared_dict,))
                p.start()
                start = datetime.now()
                p.join(time_limit_in_sec)
                end = datetime.now()
            else:
                start = datetime.now()
                alive = mp.Event()
                safe = mp.Event()
                p = mp.Process(target=target, args=(shared_dict,), kwargs={'alive': alive, 'safe': safe})
                p.start()
                while p.is_alive() and not safe.is_set():
                    if alive.wait(inactivity_timeout):
                        if (datetime.now() - start).total_seconds() >= time_limit_in_sec:
                            print("Time limit reached.")
                            break
                        if safe.is_set():
                            p.join(time_limit_in_sec - (datetime.now() - start).total_seconds())
                            break
                        alive.clear()
                        continue
                    else:
                        print("Inactivity timeout reached.")
                        break
                end = datetime.now()

            if p.is_alive():
                p.terminate()
                p.join()
                main_run.alert(
                    f"Training run {i + 1}/{len(param_sets)} was cancelled due to inactivity or time limit!",
                    f"In job \'{job_counter}\' the run \'{name}\' was cancelled due to a time limit of {time_limit_in_sec} seconds. "
                    f"In total {i + 1} / {len(param_sets)} runs are completed.\n"
                    f"This run took: {str(end - start).split('.')[0]}\n"
                    f"Estimated time until completion of all runs: "
                    f"{str(((end - start) * (len(param_sets) - (i + 1)))).split('.')[0]}",
                    level="ERROR"
                )
            else:
                if p.exitcode != 0:
                    main_run.alert(
                        f"Training run {i + 1}/{len(param_sets)} failed!",
                        f"In job \'{job_counter}\' the run \'{name}\' failed. "
                        f"In total {i + 1} / {len(param_sets)} runs are completed.\n"
                        f"This run took: {str(end - start).split('.')[0]}\n"
                        f"Estimated time until completion of all runs: "
                        f"{str(((end - start) * (len(param_sets) - (i + 1)))).split('.')[0]}",
                        level="ERROR"
                    )
                else:
                    main_run.alert(
                        f"Finished training run {i + 1}/{len(param_sets)}!",
                        f"In job \'{job_counter}\' the run \'{name}\' is done. "
                        f"In total {i + 1} / {len(param_sets)} runs are completed.\n"
                        f"This run took: {str(end - start).split('.')[0]}\n"
                        f"Estimated time until completion of all runs: "
                        f"{str(((end - start) * (len(param_sets) - (i + 1)))).split('.')[0]}",
                        level="INFO"
                    )

    main_run.finish()
    print(f"Finished all {len(param_sets)} runs.")


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
