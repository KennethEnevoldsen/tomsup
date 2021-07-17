"""
This script contains utility functions used in the tomsup package
"""
import json
import os
from typing import List, Optional, Union

import pandas as pd

from tomsup.agent import AgentGroup
from tomsup.agent import Agent


def valid_agents() -> dict:
    """
    Returns:
        dict: A dictionary of valid agent, where each key is an valid agent shorthand.
            Each dictionary key (e.g. shorthand) refers to a new dictionary containing,
            additional information, such as a short description of the strategy, and a reference.
            See examples for more.
    Examples:
        >>> output = valid_agents()
        >>> isinstance(output, dict)
        True
        >>> output.keys()
        dict_keys(['RB', 'WSLS'])
        >>> output['RB']
        {'name': 'Random Bias',
        'shorthand': 'RB',
        'example': 'RB(bias = 0.5)',
        'reference': 'Devaine, et al. (2017)',
        'strategy': 'Chooses 1 randomly based on a probability or bias'}
    """
    this_dir, this_filename = os.path.split(__file__)
    path = os.path.join(this_dir, "agent_info.json")
    with open(path, "r") as fp:
        agent_dict = json.load(fp)
    return agent_dict


def create_agents(
    agents: Union[str, list], start_params: Optional[List[dict]] = None, **kwargs
) -> Union[AgentGroup, Agent]:
    """
    Args:
        agents (Union[str, list]): A str denoting an agent or a list of agents. To see option use `valid_agents()`.
        start_params (Optional[List[dict]], optional): The starting parameters of the agent.
            Defaults to None, indicating the default starting parameters.

    Returns:
        Union[AgentGroup, Agent]: If only one agent is specified returns an Agent otherwise returns an AgentGroup.

    Examples:
        >>> group = create_agents(agents = ['RB']*3)
        >>> isinstance(group, AgentGroup)
        True
        >>> group.get_names()
        ['RB_0', 'RB_1', 'RB_2']
        >>> RB_agent = group.get_agent('RB_0')
        >>> RB_agent.get_strategy()
        'RB'
    """
    if isinstance(agents, str):
        if (start_params is not None) and ("save_history" in start_params):
            save_history = start_params["save_history"]
        else:
            save_history = False
        return Agent(strategy=agents, save_history=save_history, **kwargs)
    return AgentGroup(agents=agents, start_params=start_params)
