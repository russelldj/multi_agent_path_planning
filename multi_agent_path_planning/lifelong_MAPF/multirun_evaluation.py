import itertools
import json
import logging
import os
import time
from collections import defaultdict
from multiprocessing import Process
from pathlib import Path

import argparse
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

NAMES_TO_INDS = {
    "map_file": 0,
    "num_agents": 1,
    "task_factory_cls": 2,
    "task_allocator_cls": 3,
    "mapf_solver_cls": 4,
}


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


def break_up_by_key_elements(data, metric):
    """_summary_

    Args:
        data (_type_): _description_
        ind (_type_): _description_

    Returns:
        _type_: Each unique value for that trait is a key. Each value is a dict mapping from the full
                config to the values
    """
    # Preprocess to get a list of keys
    ind = NAMES_TO_INDS[metric]
    selected_keys = [x[ind] for x in data.keys()]
    keys = list(data.keys())
    values = list(data.values())

    unique_keys, inv = np.unique(selected_keys, return_inverse=True)
    output_dict = {}
    for i in range(max(inv) + 1):
        matching_inds = np.where(i == inv)[0]
        unique_key = unique_keys[i]
        output_dict[unique_key] = {keys[i]: values[i] for i in matching_inds}
    return output_dict


def vis_from_json(savepath):
    with open(savepath, "r") as f:
        data = json.load(f)
    values = data.values()
    keys = [tuple(x[1:-1].split(", ")) for x in data.keys()]
    keys = [tuple((eval(y) for y in x)) for x in keys]
    data = {k: v for k, v in zip(keys, values)}
    # "map_file": 0,
    # "num_agents": 1,
    # "task_factory_cls": 2,
    # "task_allocator_cls": 3,
    # "mapf_solver_cls": 4,
    plot_all_data(
        data,
        breakup_config="map_file",
        versus_config="num_agents",
        compare_config="task_allocator_cls",
        vis_metric="pathlength",
    )


def plot_one_data(data, versus_config, compare_config, vis_metric):
    # On key per value that we're comparing across
    by_comparison_metric = break_up_by_key_elements(data, compare_config)
    for compare_key, values_for_compare_key in by_comparison_metric.items():
        # Break up by versus metric
        broken_up_by_versus = break_up_by_key_elements(
            values_for_compare_key, metric=versus_config
        )
        fracs_valid = []
        means = []
        stds = []
        for v in broken_up_by_versus.values():
            metrics_for_one_config = list(v.values())[0]
            valid_metrics = [x for x in metrics_for_one_config if x is not None]
            valid_values = [x[vis_metric] for x in valid_metrics]
            frac_valid = len(valid_metrics) / len(metrics_for_one_config)
            fracs_valid.append(frac_valid)
            if len(valid_values) > 0:
                means.append(np.mean(valid_values))
                stds.append(np.std(valid_values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        f, axs = plt.subplots(1, 2)
        means = np.array(means)
        stds = np.array(stds)
        x_values = np.array(list(broken_up_by_versus.keys()))
        axs[0].plot(x_values, means)
        axs[0].fill_between(x_values, means - stds, means + stds, alpha=0.3)
        axs[1].plot(x_values, fracs_valid)

        axs[0].set_ylabel(f"Metric: {vis_metric}")
        axs[1].set_ylabel(f"Fraction valid")

        axs[0].set_xlabel(versus_config)
        axs[1].set_xlabel(versus_config)
        plt.show()
        # Compute mean and std for each metric
        # Plot


def plot_all_data(data, breakup_config, versus_config, compare_config, vis_metric):
    """_summary_

    Args:
        data (_type_): _description_
        breakup_metric (_type_): Make a different plot for each of these (categorical) values
        versus_metric (_type_): Use this metric (float) as the x axis
        compare_metric (_type_): Create a different line on the same plot for this (categorical) value
    """
    broken_ups = break_up_by_key_elements(data=data, metric=breakup_config)
    for broken_up in broken_ups.values():
        plot_one_data(
            broken_up,
            versus_config=versus_config,
            compare_config=compare_config,
            vis_metric=vis_metric,
        )


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-existing-json", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.vis_existing_json:
        vis_from_json(JSON_PATH)
    else:
        multirun_experiment_runner()
