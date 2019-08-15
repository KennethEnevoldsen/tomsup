import pandas as pd

def valid_agents():
    """
    prints a list of valid agents
    """
    RB_d = {'agent': 'Random Bias',
     'shorthand': 'RB',
     'example': 'RB(bias = 0.5)', 
     'reference': 'Devaine, et al. (2017)', 
     'strategy': 'Chooses 1 randomly based on a probability or bias'}
    WSLS_d = {
     'agent': 'Win-stay, lose-switch',
     'shorthand': 'WSLS',
     'example': 'WSLS()', 
     'reference': 'Nowak & Sigmund (1993)', 
     'strategy': 'If it win it chooses the same strategy and if it loose it change to another'}
    agent_dicts = [RB_d, WSLS_d]
    print(pd.DataFrame(agent_dict).T)


def create_agents(agents, start_params):
    """
    
    TODO: create a create_agent() function (for single agents) (eller lav den så den virker på begge måder)
    agents is a list of agents e.g. ['RB', 'WSLS', '0-tom']

    a wrapper function for the class Agent_group()

    Examples:
    >>> create_agents(['RB']*3)
    """
