"""
This script contains utility functions
"""
import pandas as pd
from tomsup.agent import AgentGroup
import json
import os

def valid_agents():
    """
    prints dictionary of valid agent, where each key is an valid agent shorthand.
    Each dictionary key (e.g. shorthand) refers to a new dictionary containing, 
    additional information, such as a short description of the strategy, 
    a code exampel and a reference. See examples for more. 

    TODO: fix output of examples

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
    with open(path, 'r') as fp:
        agent_dict = json.load(fp)
    return agent_dict


def create_agents(agents, start_params = None):
    """
    Given a list of agents and their starting parameters returns an
    object agent_group.
    A wrapper for accessing the Agent_group class.

    a wrapper function for the class Agent_group()

    Examples:
    >>> group = create_agents(agents = ['RB']*3)
    >>> isinstance(group, AgentGroup)
    True
    >>> group.get_names()
    ['RB_0', 'RB_1', 'RB_2']
    >>> RB_agent = group.get_agent('RB_0')
    >>> RB_agent.get_strategy()
    'RB'
    >>> group.set_env(env = 'round_robin')
    """
    return AgentGroup(agents = agents, start_params = start_params)



# %% Run test
if __name__ == "__main__":
  import doctest
  doctest.testmod(verbose=True)