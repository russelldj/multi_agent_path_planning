import typing

from multi_agent_path_planning.centralized.sipp.graph_generation import SippGraph, State
from multi_agent_path_planning.centralized.sipp.sipp import SippPlanner
from multi_agent_path_planning.lifelong_MAPF.datastuctures import (
    AgentSet,
    Map,
    Path,
    Agent,
    Location,
)
from multi_agent_path_planning.lifelong_MAPF.helpers import make_map_dict_dynamic_obs
from multi_agent_path_planning.centralized.cbs.cbs import CBS, Environment
from multi_agent_path_planning.lifelong_MAPF.task_allocator import (
    find_closest_list_index,
)
import numpy as np
import logging
from copy import copy
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import random


class BaseMAPFSolver:
    """
    Def
    """

    def solve_MAPF_instance(
        self, map_instance: Map, agents: AgentSet, timestep: int,
    ) -> typing.List[Path]:
        """
        Arguments:
            map: The map representation
            assignments: The assignments that must be completed
            planned_paths: The already planned paths
            timestep: The simulation timestep
        """
        return agents

    @classmethod
    def get_name(cls):
        return "base"


class SippSolver:
    """
    Def
    """

    def solve_MAPF_instance(
        self, map_instance: Map, agents: AgentSet, timestep: int,
    ) -> typing.List[Path]:
        """
        Arguments:
            map: The map representation
            assignments: The assignments that must be completed
            planned_paths: The already planned paths
            timestep: The simulation timestep
        """
        temp_map = make_map_dict_dynamic_obs(
            map_instance=map_instance, agents=agents, timestep=timestep
        )

        if "agents" not in temp_map:
            logging.info(f"No Agent Replans at Timestep :{timestep}")
            return agents

        if len(temp_map["agents"]) == 0:
            logging.info(f"No Agent Replans at Timestep :{timestep}")
            return agents

        for agent in range(len(temp_map["agents"])):
            id_agent = temp_map["agents"][agent]["name"]
            logging.info(f"setting plan for the following agent {id_agent}")
            sipp_planner = SippPlanner(temp_map, agent)

            if sipp_planner.compute_plan():
                plan = sipp_planner.get_plan()
                logging.info(f"Plan for agent: {id_agent}, {plan}")
                temp_map["dynamic_obstacles"].update(plan)
                # update agent
                agents.agents[
                    agents.get_agent_from_id(id_agent)
                ].set_planned_path_from_plan(plan)
            else:
                logging.warn(
                    f"Plan not found for agent : {id_agent} at timestep: {timestep}"
                )

        return agents


class CBSSolver:
    """
    Def
    """

    def __init__(self):
        self.idle_goals = []

    def fixup_goals(self, map_instance: Map, agent_list: typing.List[dict]):
        """Some goals may be unset, others may be duplicates

        Args:
            map_instance (Map): _description_
            agents (AgentSet): set of agents

        Returns:
            _type_: _description_
        """

        # These are the initial goals which may be None
        initial_goals = [a["goal"] for a in agent_list]
        # THESE LOCS ARE IN XY since they were need to be in the same convention as
        # what the solver is going to take
        freespace_locs_xy = np.flip(map_instance.unoccupied_inds, axis=1).tolist()
        # freespace_locs_xy = map_instance.unoccupied_inds.tolist()

        # We randomize the allocation order to avoid bias
        permutation = np.random.permutation(len(initial_goals))
        # Run through the goals and make sure each one is uniuqe
        for i in permutation:
            initial_goal = initial_goals[i]
            # Pick randomly if there is no goal yet
            if initial_goal is None:
                initial_goal = random.choice(freespace_locs_xy)

            ind = find_closest_list_index(Location(initial_goal), freespace_locs_xy)
            updated_goal = freespace_locs_xy.pop(ind)
            # Set goal
            agent_list[i]["goal"] = updated_goal

        return agent_list

    def solve_MAPF_instance(
        self, map_instance: Map, agents: AgentSet, timestep: int,
    ) -> typing.List[Path]:
        """
        Arguments:
            map: The map representation
            assignments: The assignments that must be completed
            planned_paths: The already planned paths
            timestep: The simulation timestep
        """
        # No new plans need to be added
        # if not agents.any_agent_needs_new_plan():
        #     return agents

        # Get the dimensions, agents, and obstacles in the expected format
        dimension = map_instance.get_dim()
        agent_list = agents.get_agent_dict()
        obstacles = map_instance.get_obstacles()

        # Make sure there are no errors in the agent list
        agent_list = self.fixup_goals(map_instance=map_instance, agent_list=agent_list)

        # Create an environment and solver
        env = Environment(dimension, agent_list, obstacles)
        cbs = CBS(env)
        # Solve the CBS instance

        starts = [tuple(a["start"]) for a in agent_list]
        goals = [tuple(a["goal"]) for a in agent_list]
        logging.info(f"\nstarts: {starts},\ngoals: {goals},\nobstacles: {obstacles}")

        if np.any([g in obstacles for g in goals]):
            logging.error("Goal in obstacles")
            breakpoint()

        if len(np.unique(goals, axis=1)) != len(goals):
            logging.error("Duplicate goals")
            breakpoint()

        logging.info("solving..")
        solution = cbs.search()
        logging.info("solved!")

        # Set the paths for each agent
        for agent_id in solution.keys():
            agents.agents[
                agents.get_agent_from_id(agent_id)
            ].set_planned_path_from_plan(solution)

        return agents

    @classmethod
    def get_name(cls):
        return "base"
