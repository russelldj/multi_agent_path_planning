import itertools
import json
import logging
import os
import time
from collections import defaultdict
from multiprocessing import Process
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from call_function_with_timeout import SetTimeout
from tqdm import tqdm

from multi_agent_path_planning.config import BENCHMARK_DIR, MEAN_KEY, STD_KEY, VIS_DIR
from multi_agent_path_planning.lifelong_MAPF.datastuctures import Agent, Map, TaskSet
from multi_agent_path_planning.lifelong_MAPF.dynamics_simulator import (
    BaseDynamicsSimulator,
)
from multi_agent_path_planning.lifelong_MAPF.helpers import *
from multi_agent_path_planning.lifelong_MAPF.lifelong_MAPF import (
    lifelong_MAPF_experiment,
)
from multi_agent_path_planning.lifelong_MAPF.mapf_solver import CBSSolver
from multi_agent_path_planning.lifelong_MAPF.task_allocator import RandomTaskAllocator
from multi_agent_path_planning.lifelong_MAPF.task_factory import RandomTaskFactory

JSON_PATH = Path(VIS_DIR, "results.json")


def make_key_tuple_from_config_dict(config_dict):
    key_tuple = (
        config_dict["map_file"].name,
        config_dict["num_agents"],
        config_dict["task_factory_cls"].get_name(),
        config_dict["task_allocator_cls"].get_name(),
        config_dict["mapf_solver_cls"].get_name(),
    )
    return key_tuple


def save_to_json(results_dict, savepath: Path):
    results_dict = dict(results_dict)
    results_dict = {str(k): v for k, v in results_dict.items()}
    os.makedirs(savepath.parent, exist_ok=True)
    with open(savepath, "w") as f:
        json.dump(results_dict, f)


def break_up_by_key_elements(data, ind):
    """_summary_

    Args:
        data (_type_): _description_
        ind (_type_): _description_

    Returns:
        _type_: Each unique value for that trait is a key. Each value is a dict mapping from the full
                config to the values
    """
    # Preprocess to get a list of keys
    list_of_tuples = [tuple(x[1:-1].split(", ")) for x in data.keys()]
    list_of_tuples = [tuple((eval(y) for y in x)) for x in list_of_tuples]
    values = list(data.values())

    selected_values = [(x[ind]) for x in list_of_tuples]
    unique_values, inv = np.unique(selected_values, return_inverse=True)
    output_dict = {}
    for unique_value in unique_values:
        matching_inds = np.where(unique_value == inv)[0]
        output_dict[unique_value] = {
            list_of_tuples[i]: values[i] for i in matching_inds
        }
    return output_dict


def vis_from_json(savepath):
    with open(savepath, "r") as f:
        data = json.load(f)
    plot_success_versus_metric(data)
    breakpoint()


def plot_success_versus_metric(data, breakup_metric, versus_metric):
    broken_by_number_of_agents = break_up_by_key_elements(data=data, ind=1)
    breakpoint()


def plot_metrics_for_given_n_agents(
    map_files, num_agents, fracs_successful, means_and_stds, output_folder
):
    os.makedirs(output_folder, exist_ok=True)
    # Show fraction successful
    plt.plot(fracs_successful, label="Fraction successful")
    plt.xticks(
        range(len(map_files)), [map_file.name for map_file in map_files], rotation=45
    )
    plt.title(
        f"Fraction of successful runs within the timelimit\n for {num_agents} agents"
    )
    plt.legend()
    plt.savefig(Path(output_folder, f"n_agents:{num_agents}_fraction.png"))
    plt.close()
    means_and_stds = zipunzip_list_of_dicts(means_and_stds)
    for metric_name, metrics in means_and_stds.items():
        metrics = zipunzip_list_of_dicts(metrics)
        means = np.array(metrics[MEAN_KEY])
        stds = np.array(metrics[STD_KEY])
        x = list(range(len(map_files)))
        plt.plot(x, means, label=f"{metric_name} with one std bound")
        plt.fill_between(x, means - stds, means + stds, alpha=0.3)
        plt.xticks(x, [map_file.name for map_file in map_files], rotation=45)
        plt.title(f"Metric: {metric_name}, for {num_agents} agents")
        plt.legend()
        plt.savefig(
            Path(output_folder, f"n_agents:{num_agents}_metric_{metric_name}.png")
        )
        plt.close()
    plt.close("all")


def zipunzip_list_of_dicts(list_of_dicts):
    if len(list_of_dicts) == 0:
        return {}
    try:
        reformatted_dict = {k: [d[k] for d in list_of_dicts] for k in list_of_dicts[0]}
    except KeyError:
        # This should never be encountered and needs to be inpected
        breakpoint()
    return reformatted_dict


def compute_mean_and_std_for_dict(dict_of_metrics):
    aggregate_metrics_dict = {}
    for metric_name, metrics in dict_of_metrics.items():
        mean = np.mean(metrics)
        std = np.std(metrics)
        aggregate_metrics_dict[metric_name] = {MEAN_KEY: mean, STD_KEY: std}
    return aggregate_metrics_dict


def create_n_random_agents_in_freespace(map_instance: Map, n_agents):
    starts = map_instance.get_random_unoccupied_locs(n_samples=n_agents)
    agent_list = []
    for i, start in enumerate(starts):
        agent_list.append(Agent(loc=start, ID=f"agent_{i:03d}"))
    agent_set = AgentSet(agent_list)
    return agent_set


def singlerun_experiment_runner(
    map_file,
    num_agents,
    task_factory_cls=RandomTaskFactory,
    task_allocator_cls=RandomTaskAllocator,
    mapf_solver_cls=CBSSolver,
    max_timesteps=100,
    verbose=False,
    timeout_seconds=10,
):
    map_instance = Map(map_file)
    initial_agents = create_n_random_agents_in_freespace(
        map_instance=map_instance, n_agents=num_agents
    )

    lifelong_MAPF_experiment_w_timeout = SetTimeout(
        lifelong_MAPF_experiment, timeout=timeout_seconds
    )
    start = time.time()
    (is_done, is_timeout, error_message, results) = lifelong_MAPF_experiment_w_timeout(
        map_instance=map_instance,
        initial_agents=initial_agents,
        task_factory=task_factory_cls(map_instance),
        task_allocator=task_allocator_cls(),
        mapf_solver=mapf_solver_cls(),
        max_timesteps=max_timesteps,
        verbose=verbose,
    )
    runtime = time.time() - start
    if is_timeout:
        return None
    else:
        metrics = results[1]
        metrics["runtime"] = runtime
        return metrics


def multirun_experiment_runner(
    map_folder=Path(BENCHMARK_DIR, "8x8_obst12"),
    map_glob="*",
    nums_agents=(2, 3, 4, 5),
    task_factory_classes=(RandomTaskFactory,),
    task_allocator_classes=(RandomTaskAllocator,),
    mapf_solver_classes=(CBSSolver,),
    n_maps=10,
    max_timesteps=100,
    timeout_seconds=10,
    num_random_trials=10,
    verbose=True,
):
    map_files = sorted(map_folder.glob(map_glob))[:n_maps]

    config_tuples = list(
        itertools.product(
            map_files,
            nums_agents,
            task_factory_classes,
            task_allocator_classes,
            mapf_solver_classes,
        )
    )
    # Repeat each option num_random_trials times
    config_tuples = list(
        itertools.chain.from_iterable(
            (itertools.repeat(config_tuples, num_random_trials))
        )
    )
    np.random.shuffle(config_tuples)
    config_dicts = [
        {
            "map_file": t[0],
            "num_agents": t[1],
            "task_factory_cls": t[2],
            "task_allocator_cls": t[3],
            "mapf_solver_cls": t[4],
        }
        for t in config_tuples
    ]
    logging.warning(f"Running {len(config_dicts)} number of different configurations")

    results_dict = defaultdict(list)
    progress_bar = tqdm(config_dicts)
    for config_dict in progress_bar:
        experiment_key = make_key_tuple_from_config_dict(config_dict)
        progress_bar.set_description(str(experiment_key))
        experiment_result = singlerun_experiment_runner(
            verbose=verbose,
            timeout_seconds=timeout_seconds,
            max_timesteps=max_timesteps,
            **config_dict,
        )
        results_dict[experiment_key].append(experiment_result)
        save_to_json(results_dict=results_dict, savepath=JSON_PATH)


if __name__ == "__main__":
    vis_from_json(JSON_PATH)
    multirun_experiment_runner()
