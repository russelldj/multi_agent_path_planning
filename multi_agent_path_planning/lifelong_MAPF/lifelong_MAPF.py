import argparse
import typing
import numpy as np

from multi_agent_path_planning.lifelong_MAPF.datastuctures import Agent, Map, TaskSet
from multi_agent_path_planning.lifelong_MAPF.dynamics_simulator import (
    BaseDynamicsSimulator,
)
from multi_agent_path_planning.lifelong_MAPF.helpers import *
from multi_agent_path_planning.lifelong_MAPF.mapf_solver import (
    BaseMAPFSolver,
    SippSolver,
    CBSSolver,
)
from multi_agent_path_planning.lifelong_MAPF.task_allocator import (
    BaseTaskAllocator,
    RandomTaskAllocator,
    LinearSumTaskAllocator,
    TASK_ALLOCATOR_CLASS_DICT,
)
from multi_agent_path_planning.lifelong_MAPF.task_factory import (
    BaseTaskFactory,
    RandomTaskFactory,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input file")
    parser.add_argument("output", help="output file with the schedule")
    parser.add_argument(
        "--allocator",
        choices=TASK_ALLOCATOR_CLASS_DICT.keys(),
        help="which allocator to use",
        default="random_random",
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="warning",
        help="Provide logging level. Example --loglevel debug, default=warning",
        choices=logging._nameToLevel.keys(),
    )
    parser.add_argument("--random-seed", type=int, help="Optional random seed")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    logging.basicConfig(level=args.loglevel.upper())

    logging.info(args.input)

    world_map = Map(args.input)

    allocator = TASK_ALLOCATOR_CLASS_DICT[args.allocator](world_map)

    output, metrics = lifelong_MAPF_experiment(
        map_instance=world_map,
        initial_agents=make_agent_set(args.input),
        task_factory=RandomTaskFactory(
            world_map,
            start_timestep=10,
            end_timestep=50,
            max_tasks=10,
            per_task_prob=0.2,
        ),
        task_allocator=allocator,
        mapf_solver=CBSSolver(),
        # mapf_solver=SippSolver(),
        dynamics_simulator=BaseDynamicsSimulator(),
    )

    with open(args.output, "w") as output_yaml:
        yaml.safe_dump(output, output_yaml)

    logging.info("the end")


def lifelong_MAPF_experiment(
    map_instance: Map,
    initial_agents: typing.Dict[int, Agent],
    task_factory: BaseTaskFactory,
    task_allocator: BaseTaskAllocator,
    mapf_solver: BaseMAPFSolver,
    dynamics_simulator: BaseDynamicsSimulator = BaseDynamicsSimulator(),
    max_timesteps: int = 100,
    verbose: bool = False,
):
    """
    Arguments:
        map_instance: The obstacles and extent
        initial_agents: a dict mapping from agent IDs to the location
        task_factory: Creates the tasks
        task_allocator: Allocates the tasks
        mapf_solver: Solves a MAPF instance,
        max_timesteps: How many timesteps to run the simulation for
    """
    # The set of tasks which need to be exected
    # It should be a List[Task]
    open_tasks = TaskSet()

    # This is all agents
    agents = initial_agents
    for agent in agents.tolist():
        agent.verbose = verbose

    output = {}
    active_task_list = []
    open_task_list = []

    # Run for a fixed number of timesteps
    for timestep in range(max_timesteps):
        logging.info(f"========== Timestep: {timestep} ==========")

        # Ask the task factory for new task
        new_tasks, no_new_tasks = task_factory.produce_tasks(agents, timestep=timestep)
        # Add them to the existing list
        open_tasks.add_tasks(new_tasks)

        logging.info(f"Number of Open Tasks: {open_tasks.__len__()}")

        # If there are no current tasks and the factory says there won't be any more
        # and all the agents are at the goal, break
        if len(open_tasks) == 0 and no_new_tasks and agents.all_at_goals():
            logging.info("Jobs Done")
            break

        # Assign the open tasks to the open agents
        agents = task_allocator.allocate_tasks(open_tasks, agents)

        # Save active tasks
        for agent in agents.agents:
            if agent.task is not None:
                task_dict = agent.task.get_dict()
                task_dict["t"] = timestep
                active_task_list.append(task_dict)
        output["active_tasks"] = active_task_list

        # Save open tasks
        for open_task in open_tasks.task_dict.values():
            task_dict = open_task.get_dict()
            task_dict["t"] = timestep
            open_task_list.append(task_dict)
        output["open_tasks"] = open_task_list

        # print("AGENTS")
        # for agent in agents.get_agent_dict():
        #    print(agent)
        # print("OBSTACLES")
        # for obs in map_instance.obstacles:
        #    print(obs)

        # Plan all the required paths
        agents = mapf_solver.solve_MAPF_instance(
            map_instance=map_instance, agents=agents, timestep=timestep,
        )
        # Step the simulation one step and record the paths
        agents = dynamics_simulator.step_world(
            agents=agents, timestep=timestep, verbose=verbose
        )
        open_tasks.step_idle()

    # Save tasks one more time to match timestep of agents
    for agent in agents.agents:
        if agent.task is not None:
            task_dict = agent.task.get_dict()
            task_dict["t"] = timestep + 1
            active_task_list.append(task_dict)
    output["active_tasks"] = active_task_list
    for open_task in open_tasks.task_dict.values():
        task_dict = open_task.get_dict()
        task_dict["t"] = timestep + 1
        open_task_list.append(task_dict)
    output["open_tasks"] = open_task_list

    metrics = agents.report_metrics()

    # Combine visualization data
    output.update(agents.get_executed_paths())
    return output, metrics


if __name__ == "__main__":
    paths = main()
