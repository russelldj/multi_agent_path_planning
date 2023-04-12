from multi_agent_path_planning.config import BENCHMARK_DIR, MEAN_KEY, STD_KEY, VIS_DIR
from pathlib import Path
from multi_agent_path_planning.lifelong_MAPF.lifelong_MAPF import (
    lifelong_MAPF_experiment,
)
from multi_agent_path_planning.lifelong_MAPF.datastuctures import Agent, Map, TaskSet
from multi_agent_path_planning.lifelong_MAPF.helpers import *
from multi_agent_path_planning.lifelong_MAPF.mapf_solver import CBSSolver
from multi_agent_path_planning.lifelong_MAPF.task_allocator import RandomTaskAllocator
from multi_agent_path_planning.lifelong_MAPF.task_factory import RandomTaskFactory
from multi_agent_path_planning.lifelong_MAPF.dynamics_simulator import (
    BaseDynamicsSimulator,
)
import numpy as np
from call_function_with_timeout import SetTimeout
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
from multiprocessing import Process
import os


def plot_metrics(
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
    nums_agents=(2,),
    task_factory_cls=RandomTaskFactory,
    task_allocator_cls=RandomTaskAllocator,
    mapf_solver_cls=CBSSolver,
    max_timesteps=100,
    num_random_trials=10,
    n_maps=10,
    verbose=True,
):
    map_files = sorted(map_folder.glob(map_glob))[:n_maps]
    results_dict = {}
    for num_agents in nums_agents:
        results_dict[num_agents] = {"fracs_successful": [], "means_and_stds": []}
        for map_file in tqdm(map_files):
            metrics_list = []
            for _ in range(num_random_trials):
                metrics = singlerun_experiment_runner(
                    map_file=map_file,
                    num_agents=num_agents,
                    task_factory_cls=task_factory_cls,
                    task_allocator_cls=task_allocator_cls,
                    mapf_solver_cls=mapf_solver_cls,
                    max_timesteps=max_timesteps,
                    verbose=verbose,
                )
                metrics_list.append(metrics)
            # TODO figure out why empty dicts are showing up
            successful_metrics = [
                m for m in metrics_list if (m is not None and m != {})
            ]
            frac_successful = len(successful_metrics) / num_random_trials
            successful_metrics = zipunzip_list_of_dicts(successful_metrics)
            mean_and_std_per_metric = compute_mean_and_std_for_dict(successful_metrics)

            results_dict[num_agents]["fracs_successful"].append(frac_successful)
            results_dict[num_agents]["means_and_stds"].append(mean_and_std_per_metric)

    # Plotting
    for num_agents in nums_agents:
        plot_metrics(
            map_files,
            num_agents,
            fracs_successful=results_dict[num_agents]["fracs_successful"],
            means_and_stds=results_dict[num_agents]["means_and_stds"],
            output_folder=Path(VIS_DIR, "evaluation"),
        )


if __name__ == "__main__":
    multirun_experiment_runner()
