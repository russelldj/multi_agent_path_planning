import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
import typing

from multi_agent_path_planning.lifelong_MAPF.datastuctures import (
    AgentSet,
    TaskSet,
    Location,
    Agent,
)
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


def find_closest_list_index(loc: Location, list):
    best_dist = np.inf
    best_i = 0
    element = loc.as_ij()
    for i, item in enumerate(list):
        dist = np.linalg.norm(np.array(element) - np.array(item))
        if dist < best_dist:
            best_i = i
            best_dist = dist
    return best_i


def pick_idle_goals_kmeans(map_instance, agents: AgentSet):
    idle_agents = []
    for agent in agents.tolist():
        if agent.task is None:
            idle_agents.append(agent)

    # Get obstacle-free map space
    free_spaces = np.flip(map_instance.unoccupied_inds, axis=1).tolist()

    # Partition space based on obstacle map only
    # kmeans = KMeans(n_clusters=n_agents, random_state=0, n_init="auto").fit(free_spaces)
    if len(idle_agents) == 0:
        return

    kmeans = KMeans(n_clusters=len(idle_agents), random_state=0, n_init="auto").fit(
        free_spaces
    )
    idle_locs = np.rint(kmeans.cluster_centers_)

    # Remove agent goals from available free space
    for agent in agents.tolist():
        if agent.goal is not None:
            closest_i = find_closest_list_index(Location(agent.goal), free_spaces)
            free_spaces.pop(closest_i)

    # Make sure rounded positions are in free space
    idle_goals = [Location(loc) for loc in idle_locs]

    # Assign idle agents using linear sum assignment
    distance_matrix = np.zeros((len(idle_goals), len(idle_agents)))

    if distance_matrix.size > 0:
        for i, idle_goal in enumerate(idle_goals):
            for j, idle_agent in enumerate(idle_agents):
                diff = np.array(Location(idle_goal).as_ij()) - np.array(
                    Location(idle_agent.loc).as_ij()
                )
                dist = np.sum(np.abs(diff))
                distance_matrix[i, j] = dist
    idle_goal_inds, idle_agent_inds = linear_sum_assignment(distance_matrix)

    for idle_goal_ind, idle_agent_ind in zip(idle_goal_inds, idle_agent_inds):
        idle_agent = idle_agents[idle_agent_ind]
        idle_goal = idle_goals[idle_goal_ind]
        idle_agent.goal = idle_goal


class BaseTaskAllocator:
    def __init__(self, map_instance, assign_unallocated_w_kmeans=True) -> None:
        self.map_instance = map_instance
        self.assign_unallocated_w_kmeans = assign_unallocated_w_kmeans

    def allocate_tasks(self, tasks: TaskSet, agents: AgentSet) -> AgentSet:
        """
        Arguments:
            tasks: The open tasks
            agents:

        Returns:
            Agents updated with assignments
        """
        return agents

    @classmethod
    def get_name(cls):
        return "base"

    def set_tasks(self, agents, tasks):
        # Assign each agent a task
        for agent, task in zip(agents, tasks):
            # TODO: make this more elegent, we dont want to assign tasks where the agent is on top of the start, unless we rework some of the initilization stuff, it creates issues with the planner which assumes there is a path required
            logging.info(f"Agent : {agent.get_id()} has been allocated a task!")
            agent.set_task(task)


class RandomTaskAllocator(BaseTaskAllocator):
    def allocate_tasks(self, tasks: TaskSet, agents: AgentSet) -> AgentSet:
        """Randomly match task with available robots

        Args:
            tasks (typing.List[Task]): The list of tasks which are open
            agents (AgentSet): The agents which may or may not be free to recive the task

        Returns:
            AgentSet: The agents are updated with their new task
        """
        # Parse which agents are not tasked yet
        untasked_agents = agents.get_unallocated_agents()

        # Sample the tasks to be assigned this timestep
        sampled_tasks = tasks.pop_n_random_tasks(min(len(untasked_agents), len(tasks)))

        # Sample the agents to associate with
        sampled_agents = untasked_agents.get_n_random_agents(len(sampled_tasks))
        self.set_tasks(agents=sampled_agents, tasks=sampled_tasks)

        if self.assign_unallocated_w_kmeans:
            pick_idle_goals_kmeans(map_instance=self.map_instance, agents=agents)

        # Return the agents which were updated by reference
        return agents

    @classmethod
    def get_name(cls):
        return "random"


class LinearSumTaskAllocator(BaseTaskAllocator):
    def allocate_tasks(self, tasks: TaskSet, agents: AgentSet, vis=False) -> AgentSet:
        # Parse which agents are not tasked yet
        untasked_agents = agents.get_unallocated_agents()
        task_list = tasks.task_list()
        distance_matrix = np.zeros((len(tasks), len(untasked_agents)))

        if distance_matrix.size > 0:
            for i, task in enumerate(task_list):
                for j, agent in enumerate(untasked_agents.agents):
                    dist = agent.loc.manhatan_dist(task.start)
                    distance_matrix[i, j] = dist

        if distance_matrix.size > 0:
            task_inds, agent_inds = linear_sum_assignment(distance_matrix)
            assigned_tasks = []
            assigned_agents = [
                untasked_agents.tolist()[agent_ind] for agent_ind in agent_inds
            ]
            for task_ind in task_inds:
                task = task_list[task_ind]
                tasks.pop_task(task)
                assigned_tasks.append(task)
            if vis:
                plt.imshow(distance_matrix)
                plt.colorbar()
                plt.xlabel("Agent")
                plt.ylabel("Task")
                plt.title(f"task_inds: {task_inds}, agent_inds: {agent_inds}")
                plt.pause(0.5)
                plt.close()

            self.set_tasks(agents=assigned_agents, tasks=assigned_tasks)

        if self.assign_unallocated_w_kmeans:
            pick_idle_goals_kmeans(map_instance=self.map_instance, agents=agents)

        return agents


TASK_ALLOCATOR_CLASS_DICT = {
    "random": RandomTaskAllocator,
    "linear_sum": LinearSumTaskAllocator,
}

