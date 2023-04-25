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

from multi_agent_path_planning.lifelong_MAPF.helpers import *
from multi_agent_path_planning.lifelong_MAPF.lifelong_MAPF import (
    lifelong_MAPF_experiment,
)
from multi_agent_path_planning.lifelong_MAPF.mapf_solver import CBSSolver
from multi_agent_path_planning.lifelong_MAPF.task_allocator import (
    RandomTaskAllocator,
    LinearSumTaskAllocator,
    RandomTaskAllocator_IdleKmeans,
    RandomTaskAllocator_IdleCurrent,
    LinearSumAllocator_IdleKmeans,
    LinearSumAllocator_IdleCurrent,
)
from multi_agent_path_planning.lifelong_MAPF.task_factory import RandomTaskFactory


NAMES_TO_INDS = {
    "map_file": 0,
    "num_agents": 1,
    "task_factory_func": 2,
    "task_allocator_cls": 3,
    "mapf_solver_cls": 4,
}


def make_key_tuple_from_config_dict(config_dict):
    key_tuple = (
        config_dict["map_file"].name,
        config_dict["num_agents"],
        str(config_dict["task_factory_func"]),
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


def plot_one_data(data, compare_config, versus_config, vis_metric, plot_title):
    # On key per value that we're comparing across
    by_comparison_metric = break_up_by_key_elements(data, compare_config)
    f, axs = plt.subplots(1, 2)
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
            try:
                valid_values = np.array(valid_values)
            # Ragged array
            except ValueError:
                # Flatten first
                flat_valid_values = list(
                    itertools.chain(*[itertools.chain(*v) for v in valid_values])
                )
                valid_values = np.array(flat_valid_values)

            if len(valid_values) > 0:
                means.append(np.mean(valid_values))
                stds.append(np.std(valid_values))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        means = np.array(means)
        stds = np.array(stds)
        x_values = np.array(list(broken_up_by_versus.keys()))
        axs[0].plot(x_values, means, label=compare_key)
        axs[0].fill_between(x_values, means - stds, means + stds, alpha=0.3)
        axs[1].plot(x_values, fracs_valid, label=compare_key)

        axs[0].set_ylabel(f"Metric: {vis_metric}")
        axs[1].set_ylabel(f"Fraction valid")

        int_x_ticks = range(
            int(np.floor(min(x_values))), int(np.ceil(max(x_values))) + 1
        )
        axs[0].set_xticks(int_x_ticks)
        axs[1].set_xticks(int_x_ticks)
        axs[0].set_xlabel(versus_config)
        axs[1].set_xlabel(versus_config)

    axs[0].legend()
    axs[1].legend()
    plt.suptitle(plot_title)
    plt.show()


def plot_all_data(
    data: dict,
    breakup_config: str,
    compare_config: str,
    versus_config: str,
    vis_metric: str,
):
    """Plot all the data generated by all runs

    Args:
        data (dict): Data summarizing all runs. 
                     Keys are the config values and values are the metrics
        breakup_config (str): Make a different plot for each value of this config
        compare_config (str): Compare across values of this config on one plot
        versus_config (str): Use this config as the x axis for visualization
        vis_metric (str): Show the value of this metric. 
    """
    broken_ups = break_up_by_key_elements(data=data, metric=breakup_config)
    for breakup_config_key, broken_up_value in broken_ups.items():
        plot_one_data(
            broken_up_value,
            compare_config=compare_config,
            versus_config=versus_config,
            vis_metric=vis_metric,
            plot_title=f"{breakup_config}: {breakup_config_key}",
        )


def compute_mean_and_std_for_dict(dict_of_metrics):
    aggregate_metrics_dict = {}
    for metric_name, metrics in dict_of_metrics.items():
        mean = np.mean(metrics)
        std = np.std(metrics)
        aggregate_metrics_dict[metric_name] = {MEAN_KEY: mean, STD_KEY: std}
    return aggregate_metrics_dict


def vis_from_json(
    savepath,
    breakup_config="mapf_solver_cls",
    versus_config="num_agents",
    compare_config="map_file",
    vis_metric="runtime",
):
    with open(savepath, "r") as f:
        data = json.load(f)
    values = data.values()
    keys = [tuple(x[1:-1].split(", ")) for x in data.keys()]
    keys = [tuple((eval(y) for y in x)) for x in keys]
    data = {k: v for k, v in zip(keys, values)}
    plot_all_data(
        data,
        breakup_config=breakup_config,
        versus_config=versus_config,
        compare_config=compare_config,
        vis_metric=vis_metric,
    )


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
    task_factory_func=RandomTaskFactory,
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
        task_factory=task_factory_func(map_instance, num_agents),
        task_allocator=task_allocator_cls(map_instance),
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
    map_folder=Path(BENCHMARK_DIR, "custom"),
    map_glob="*",
    nums_agents=list([2, 3, 4, 6, 8, 10]),
    task_freq_scaler=0.5,
    task_allocator_classes=(
        RandomTaskAllocator,
        RandomTaskAllocator_IdleKmeans,
        RandomTaskAllocator_IdleCurrent,
        LinearSumTaskAllocator,
        LinearSumAllocator_IdleKmeans,
        LinearSumAllocator_IdleCurrent,
    ),
    mapf_solver_classes=(CBSSolver,),
    n_maps=None,
    max_timesteps=100,
    timeout_seconds=5,
    num_random_trials=3,
    save_folder=VIS_DIR,
    verbose=True,
):
    savepath = Path(
        save_folder,
        f"results_task_freq_scaler:{task_freq_scaler}_timeout:{timeout_seconds}.json",
    )

    task_factory_funcs = (
        lambda map_instance, num_agents: RandomTaskFactory(
            map_instance,
            max_tasks_per_timestep=1,
            per_task_prob=task_freq_scaler
            * num_agents
            / map_instance.get_manhattan_size(),
        ),
    )
    map_files = sorted(Path(map_folder).glob(map_glob))
    if n_maps is not None:
        map_files = np.random.choice(map_files, size=n_maps, replace=False)

    config_tuples = list(
        itertools.product(
            map_files,
            nums_agents,
            task_factory_funcs,
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
            "task_factory_func": t[2],
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
        save_to_json(results_dict=results_dict, savepath=savepath)
    return savepath


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis-existing-json", help="Path to existing json")
    parser.add_argument("--vis-breakup-config", default="map_file")
    parser.add_argument("--vis-versus-config", default="num_agents")
    parser.add_argument("--vis-compare-config", default="task_allocator_cls")
    parser.add_argument("--timeout-seconds", default=30, type=int)
    parser.add_argument(
        "--task-freq-scaler",
        default=0.5,
        type=float,
        help="Multiplier on number of agents / map size",
    )
    parser.add_argument("--maps-folders", default=Path(BENCHMARK_DIR, "custom"))
    parser.add_argument(
        "--vis-metric",
        default="timesteps_to_task_start",
        choices=(
            "timesteps_to_task_start",
            "runtime",
            "pathlength",
            "idle_timesteps_before_task_assignment",
            "idle_timesteps_before_task_pickup",
            "total_timesteps_until_task_pickup",
        ),
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.vis_existing_json is not None:
        vis_from_json(
            savepath=args.vis_existing_json,
            breakup_config=args.vis_breakup_config,
            compare_config=args.vis_compare_config,
            versus_config=args.vis_versus_config,
            vis_metric=args.vis_metric,
        )
    else:
        savepath = multirun_experiment_runner(
            timeout_seconds=args.timeout_seconds, task_freq_scaler=args.task_freq_scaler
        )
        print(f"visualizing {savepath}")
        vis_from_json(
            savepath=savepath,
            breakup_config=args.vis_breakup_config,
            compare_config=args.vis_compare_config,
            versus_config=args.vis_versus_config,
            vis_metric=args.vis_metric,
        )
