import typing
import multi_agent_path_planning
import matplotlib.pyplot as plt
import numpy as np
import yaml

import logging


class Location:
    def __init__(self, loc):
        """Reduce ambiguity about i,j vs. x,y convention

        Args:
            loc (iterable or Location): assumed to be in i,j order
        """
        # TODO make this more robust
        try:
            self.ij_loc = tuple(loc)
        except TypeError:
            self.ij_loc = loc.ij_loc

    def __repr__(self) -> str:
        return f"loc: i={self.i()}, j={self.j()}"

    @staticmethod
    def from_xy(xy_loc):
        return Location((xy_loc[1], xy_loc[0]))

    def as_ij(self):
        return (self.i(), self.j())

    def as_xy(self):
        return (self.x(), self.y())

    def __eq__(self, __value: object) -> bool:
        try:
            return self.ij_loc == __value.ij_loc
        except:
            return False

    def x(self):
        return self.ij_loc[1]

    def y(self):
        return self.ij_loc[0]

    def i(self):
        return self.ij_loc[0]

    def j(self):
        return self.ij_loc[1]

    def manhatan_dist(self, other):
        diff = np.array(self.as_ij()) - np.array(other.as_ij())
        dist = np.sum(np.abs(diff))
        return dist


class Task:
    def __init__(self, start, goal, timestep, task_id):
        self.start = Location(start)
        self.goal = Location(goal)
        self.timestep = timestep
        self.task_id = task_id
        self.n_steps_idle = 0

    def step_idle(self):
        self.n_steps_idle += 1

    def get_dict(self):
        return {
            "task_id": int(self.task_id),
            "start": {"x": int(self.start.x()), "y": int(self.start.y())},
            "goal": {"x": int(self.goal.x()), "y": int(self.goal.y())},
            "t": int(self.timestep),
        }


class TaskSet:
    """An unordered set of tasks"""

    def __init__(self, task_iterable: typing.Iterable = ()) -> None:
        """An unordered set of tasks

        Args:
            task_iterable (typing.Iterable, optional): The individual tasks. Defaults to ().
        """
        self.task_dict = {i: task_iterable[i] for i in range(len(task_iterable))}
        self.next_key = len(task_iterable)

    def __len__(self) -> int:
        """The number of tasks

        Returns:
            int: the number of tasks
        """
        return len(self.task_dict)

    def step_idle(self):
        [x.step_idle() for x in self.task_dict.values()]

    def task_list(self):
        return list(self.task_dict.values())

    def add_tasks(self, task_iterable: typing.Iterable[Task]):
        """Add new tasks to the set

        Args:
            task_iterable (typing.Iterable[Task]): The tasks
        """
        new_task_dict = {
            i + self.next_key: task_iterable[i] for i in range(len(task_iterable))
        }
        self.next_key += len(task_iterable)
        self.task_dict.update(new_task_dict)

    def pop_n_random_tasks(self, n_tasks: int) -> typing.List[Task]:
        """Select, remove, and return n tasks

        Args:
            n_tasks (int): Number to remove

        Returns:
            typing.List[Task]: The tasks which were popped
        """
        keys = list(self.task_dict.keys())
        chosen_keys = np.random.choice(keys, size=n_tasks, replace=False)
        tasks = [self.task_dict.pop(k) for k in chosen_keys]
        return tasks

    def pop_task(self, task: Task):
        try:
            self.task_dict.pop(task.task_id)
        except KeyError:
            print(self.task_dict)
            breakpoint()


class PathNode:
    def __init__(self, loc, timestep):
        self.loc = Location(loc)
        self.timestep = timestep

    def __repr__(self):
        return f"{self.loc}, t: {self.timestep}"

    def get_loc(self) -> Location:
        return self.loc

    def get_time(self):
        return self.timestep


class Path:
    def __init__(self, initial_pathnodes=None):
        """A path class

        Args:
            initial_pathnodes (_type_, optional): Note that this is defaulted to None
            because storing an empty list as a default value is dangerous. If you append
            to it, that will be reflected any other time the default value is used. This is
            because default mutable values are stored in a global namespace.
        """
        if initial_pathnodes == None:
            self.pathnodes = []
        else:
            self.pathnodes = initial_pathnodes

    def __len__(self):
        return len(self.pathnodes)

    def __repr__(self) -> str:
        return str([str(x) for x in self.pathnodes])

    def get_path(self):
        return self.pathnodes

    def add_pathnode(self, pathnode: PathNode):
        self.pathnodes.append(pathnode)

    def pop_pathnode(self):
        if len(self.pathnodes) > 0:
            temp = self.pathnodes.pop(0)
            return temp
        else:
            logging.error("Popped from empty Path")
            exit()

    def get_len(self):
        if len(self.pathnodes) < 2:
            return 0
        return np.sum(
            [
                self.pathnodes[i].get_loc() != self.pathnodes[i + 1].get_loc()
                for i in range(len(self.pathnodes) - 1)
            ]
        )


class Agent:
    def __init__(self, loc, ID, goal=None, task: Task = None, verbose=False):
        """_summary_

        Args:
            loc (_type_): _description_
            ID (_type_): _description_
            goal (_type_, optional): _description_. Defaults to None.
            task (Task, optional): _description_. Defaults to None.
        """
        self.loc = Location(loc)
        self.verbose = verbose
        self.ID = ID

        self.goal = goal
        self.task = task

        self.planned_path = None
        self.executed_path = Path()
        self.executed_path.add_pathnode(PathNode(self.loc, 0))

        self.task_ids = []

        self.n_completed_task = 0
        self.timesteps_to_task_start = 0
        self.idle_timesteps = 0
        self.timestep = 1

        self.logged_timesteps_to_task_start = (
            []
        )  # how long it took to reach the task starts
        self.idle_timesteps_before_task_assignment = (
            []
        )  # How long each task sat idle before being picked up

        self.log_task_id()

    def __repr__(self) -> str:
        return f"ID: {self.ID}, loc: {self.loc}, TODO"

    def get_id(self):
        return self.ID

    def get_loc(self) -> Location:
        return self.loc

    def get_goal(self):
        return self.goal

    def get_planned_path(self):
        return self.planned_path

    def set_task(self, task: Task):
        self.task = task
        self.goal = self.task.start
        self.idle_timesteps_before_task_assignment.append(task.n_steps_idle)

    def get_executed_path(self):
        return self.executed_path

    def get_metrics(self):
        metrics = {
            "pathlength": int(self.executed_path.get_len()),
            "n_completed_tasks": int(self.n_completed_task),
            "idle_timesteps": int(self.idle_timesteps),
            "timesteps_to_task_start": self.logged_timesteps_to_task_start,
            "idle_timesteps_before_task_assignment": self.idle_timesteps_before_task_assignment,
            "total_timesteps_until_task_pickup": [
                i + t
                for i, t in zip(
                    self.idle_timesteps_before_task_assignment,
                    self.logged_timesteps_to_task_start,
                )
            ],
        }
        return metrics

    def get_as_dict(self):
        # {'start': [0, 0], 'goal': [2, 0], 'name': 'agent0'}
        return {
            "start": list(self.loc.as_xy()),
            "goal": list(self.goal.as_xy()),
            "task": self.task
            if self.goal is not None
            else None,  # There is no goal set
            "name": str(self.ID),
        }

    def is_allocated(self):
        return self.task is not None

    def is_going_to_task_start(self):
        return self.task is not None and self.goal == self.task.start

    def is_going_to_task_goal(self):
        return self.task is not None and self.goal == self.task.goal

    def is_at_task_start(self):
        return self.task is not None and self.loc == self.task.start

    def is_at_task_goal(self):
        return self.task is not None and self.loc == self.task.goal

    def is_at_agent_goal(self):
        return self.goal is not None and self.loc == self.goal

    def set_planned_path_from_plan(self, plan):
        if self.verbose:
            logging.info(f"Updating plan by adding nodes {self.ID}")
        temp_path = Path()
        for node in plan[self.ID][1:]:
            temp_loc = Location.from_xy((node["x"], node["y"]))
            temp_time = node["t"]
            if self.verbose:
                logging.info(f" adding node {temp_loc} {temp_time}")
            temp_path.add_pathnode(PathNode(temp_loc, temp_time))

        self.planned_path = temp_path

    def needs_new_plan(self):
        return self.goal is not None and (
            self.planned_path is None or len(self.planned_path) == 0
        )

    def log_task_id(self):
        if self.task is not None:
            self.task_ids.append(self.task.task_id)
        else:
            self.task_ids.append(None)

    def soft_simulation_timestep_update(self):
        # if the agent has no plan is taskless
        if self.verbose:
            logging.info(f"Dynamics for agent {self.ID}")
        if self.planned_path is None or len(self.planned_path) == 0:
            if self.verbose:
                logging.info("     Agent stationary")
                logging.info(f"     current loc {self.loc}")
            self.executed_path.add_pathnode(PathNode(self.loc, self.timestep))
            self.log_task_id()
            self.timestep += 1
            self.idle_timesteps += 1
        else:
            if self.verbose:
                logging.info("     Agent on the move")
                logging.info(f"     current loc {self.loc}")
            self.loc = self.planned_path.pop_pathnode().get_loc()
            if self.verbose:
                logging.info(f"     next loc {self.loc}")
            self.executed_path.add_pathnode(PathNode(self.loc, self.timestep))
            self.log_task_id()
            self.timestep += 1

            # Currently assigned to a task start
            if self.is_going_to_task_start():
                self.timesteps_to_task_start += 1

            # if path is exausted (goal reached)
            if len(self.planned_path.pathnodes) == 0:
                # At the start of a task
                if self.is_at_task_start():
                    self.goal = self.task.goal
                    self.planned_path = None
                    if self.timesteps_to_task_start is not None:
                        self.logged_timesteps_to_task_start.append(
                            self.timesteps_to_task_start
                        )
                    self.timesteps_to_task_start = 0
                # Reached a task goal
                elif self.is_at_task_goal():
                    self.goal = None
                    self.task = None
                    self.planned_path = None
                    self.n_completed_task += 1
                # At an arbitrary (non-task) goal
                elif self.is_at_agent_goal():
                    self.goal = None
                    self.planned_path = None
                    # Do not reset the task as we may have been
                    # redirected temporarily
                else:
                    # Agent must have been reassigned a nearby location in the
                    # MAPF solver and therefore needs to be reassigned a path to get
                    # to its goal
                    self.planned_path = None


class AgentSet:
    def __init__(self, agents: typing.List[Agent]):
        self.agents = agents

    def __len__(self):
        return len(self.agents)

    def __repr__(self) -> str:
        return str(self.agents)

    def get_executed_paths(self):
        schedule = {}

        for agent in self.agents:
            temp_id = agent.get_id()
            temp_list = []

            for i, path_node in enumerate(agent.get_executed_path().get_path()):
                temp = {}
                temp["x"] = path_node.get_loc().x()
                temp["y"] = path_node.get_loc().y()
                temp["t"] = path_node.get_time()
                temp["task_id"] = agent.task_ids[i]
                temp_list.append(temp)
            schedule[temp_id] = temp_list

        output = {}
        output["schedule"] = schedule

        return output

    def tolist(self):
        return self.agents

    def get_unallocated_agents(self):
        return AgentSet([agent for agent in self.agents if not agent.is_allocated()])

    def get_n_random_agents(self, n_agents):
        sampled_agents = np.random.choice(self.agents, size=n_agents).tolist()
        return sampled_agents

    def get_agent_from_id(self, search_id):
        # search for agent by self.ID
        for index, agent in enumerate(self.agents):
            if agent.get_id() == search_id:
                return index
        logging.warn("agent ID does not exist in agent list")
        return False

    def get_agent_dict(self):
        return [agent.get_as_dict() for agent in self.agents]

    def all_at_goals(self):
        return np.all([a.goal is None for a in self.agents])

    def any_agent_needs_new_plan(self):
        return np.any([a.needs_new_plan() for a in self.agents])

    def report_metrics(self, show_violin=False, vis=False):
        all_metrics = [a.get_metrics() for a in self.agents]
        # Flatten so each so it's a dict of metric names: list of all values
        all_metrics = {k: [m[k] for m in all_metrics] for k in all_metrics[0]}
        if vis:
            n_agents = len(all_metrics[list(all_metrics.keys())[0]])
            x = np.arange(n_agents)  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0

            fig, ax = plt.subplots(layout="constrained")

            for attribute, measurement in all_metrics.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                ax.bar_label(rects, padding=3)
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            ax.set_title("Metrics by agent")
            ax.set_xticks(x + width, x)
            ax.legend()
            plt.show()

            # Show the aggregate metrics
            f, axs = plt.subplots(1, len(all_metrics.keys()))
            plt.suptitle("Aggregate metric violin plots")
            for i, (k, metrics) in enumerate(all_metrics.items()):
                if show_violin:
                    axs[i].violinplot(metrics)
                else:
                    axs[i].boxplot(metrics)
                axs[i].set_title(k)
            plt.legend()
            plt.show()
        return all_metrics


class Map:
    def __init__(self, map, vis=False):
        with open(map, "r") as map_file:
            try:
                self.map_dict = yaml.load(map_file, Loader=yaml.FullLoader)["map"]
            except yaml.YAMLError as exc:
                logging.error(exc)
        self.map_np = np.ones(self.map_dict["dimensions"]).astype(bool)
        self.obstacles = self.map_dict["obstacles"]
        for obstacle in self.obstacles:
            # Obstacles are in the x, y convention
            self.map_np[obstacle[1], obstacle[0]] = False
        self.unoccupied_inds = np.stack(np.where(self.map_np), axis=0).T
        if vis:
            plt.imshow(self.map_np)
            plt.show()

    def get_map_dict(self):
        return self.map_dict

    def get_dim(self):
        # TODO make sure this right order
        return self.map_np.T.shape

    def get_obstacles(self):
        return self.obstacles

    def check_ocupied(self, loc: Location):
        return self.map_np[loc.i(), loc.j()]

    def get_random_unoccupied_locs(
        self, n_samples, with_replacement=False
    ) -> typing.List[Location]:
        selected_inds = np.random.choice(
            self.unoccupied_inds.shape[0], n_samples, replace=with_replacement
        )

        selected_locs = self.unoccupied_inds[selected_inds]
        # Convert to Location datatype
        selected_locs = [Location(loc) for loc in selected_locs]
        return selected_locs
