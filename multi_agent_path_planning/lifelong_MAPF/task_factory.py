import logging
from multi_agent_path_planning.lifelong_MAPF.datastuctures import Map, Task
import numpy as np


class BaseTaskFactory:
    """
    Def
    """

    # encorporate the map?
    # total number of agents
    # timestep = 0

    def produce_tasks(self, timestep: int = None):
        """
        Args:
            timestep: The current simulation timestep
        Returns:
            tasks: A list of Tasks
            complete: Is the factory done producing tasks
        """
        # only place tasks in free space

        return [], True


class RandomTaskFactory:
    def __init__(
        self,
        world_map: Map,
        max_tasks_per_timestep=1,
        max_timestep: int = None,
        max_tasks: int = None,
        per_task_prob: float = 1,
    ) -> None:
        """Initalize a random task generator

        Args:
            world_map (Map): The map of the world
            max_tasks_per_timestep (int, optional): At most how many tasks to produce per timestep. Defaults to 1.
            max_timestep (_type_, optional): The timestep to stop producing tasks at. Defaults to None.
            max_tasks: maximum number of tasks to generate
            per_task_prob: the chance of each task being generated
        """
        self.world_map = world_map
        self.max_tasks_per_timestep = max_tasks_per_timestep
        self.max_timestep = max_timestep
        self.next_task_id = 0
        self.per_task_prob = per_task_prob
        self.max_tasks = max_tasks
        self.n_created_tasks = 0

    def produce_tasks(self, timestep: int = None):
        """
        Args:
            timestep: The current simulation timestep
        Returns:
            tasks: A list of Tasks
            complete: Is the factory done producing tasks
        """
        if self.max_timestep is not None and self.max_timestep < timestep:
            return [], True

        n_tasks = np.sum(
            [
                np.random.random() <= self.per_task_prob
                for _ in range(self.max_tasks_per_timestep)
            ]
        )

        task_list = []
        for _ in range(n_tasks):
            if self.n_created_tasks >= self.max_tasks:
                break
            start, goal = self.world_map.get_random_unoccupied_locs(2)
            logging.info(f"New Task Start: {start} New Task Goal: {goal}")
            new_task = Task(
                start=start, goal=goal, timestep=timestep, task_id=self.next_task_id
            )
            task_list.append(new_task)
            self.n_created_tasks += 1
            self.next_task_id += 1

        return task_list, False
