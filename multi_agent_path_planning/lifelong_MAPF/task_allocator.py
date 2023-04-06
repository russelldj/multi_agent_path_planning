import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from multi_agent_path_planning.lifelong_MAPF.datastuctures import AgentSet, TaskSet
import matplotlib.pyplot as plt


class BaseTaskAllocator:
    """
    Def
    """

    def allocate_tasks(self, tasks: TaskSet, agents: AgentSet) -> AgentSet:
        """
        Arguments:
            tasks: The open tasks
            agents:

        Returns:
            Agents updated with assignments
        """
        return agents

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

        # Return the agents which were updated by reference
        return agents


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

        return agents


TASK_ALLOCATOR_CLASS_DICT = {
    "random": RandomTaskAllocator,
    "linear_sum": LinearSumTaskAllocator,
}

