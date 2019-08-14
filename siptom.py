"""
This is a python implementation of sirtom


#TODO
    - X add WSLS
    - add 0-tom
    - add k-tom
    - X add create_agents() (agent list)
    - X add pairing
    - add compete to pairing 
    - X create a 1v1 compete function

"""


#%% import modules
import numpy as np
import pandas as pd
from warnings import warn
from itertools import combinations
import random




#%% Payoff Matrix
class Payoff_matrix():
    """
    Example:
    >>> p_matrix = Payoff_matrix(name="penny_competitive")
    >>> p_matrix.matrix
    array([[[ 1, -1],
        [-1,  1]],

       [[-1,  1],
        [ 1, -1]]])
    >>> p_matrix.outcome(1,1, agent = "p1")
    -1

    TODO: 
    - add method to class payoff matrix to fetch value
    - make a nicer print
    """
    def __init__(self, name, predefined=None):
        """
        TODO:
        # add the remaining payoff matrices
        """
        self.name = name
        if name == "penny_competitive":
            self.matrix = np.array(([(1, -1), (-1, 1)],
                                    [(-1, 1), (1, -1)]))
        if predefined:
            matrix = np.array(predefined)
            if matrix.shape == (2, 2, 2):
                self.matrix = np.array(predefined)
            else:
                raise Exception("Predefined should be a valid matrix where matrix.shape == (2, 2, 2), e.g. a 2x2x2 matrix")

    def outcome(self, action_agent0, action_agent1, agent = "p0"):
        """
        assumes action_agent0 and action_agent1 to be integers
        agent is either 'p0' or 'p1' indicating whether the agent is player one or two
        """
        if agent == "p0":
            return self.matrix[0, action_agent0, action_agent1]
        elif agent == "p1":
            return self.matrix[1, action_agent0, action_agent1]
        else:
            raise TypeError("Please input valid agent, valid options include 'p0' and 'p1'.")

    def __call__(self):
        return self.matrix


#%% Agents
class Agent():
    """

    TODO:
    make a create_agent(strategy = "RB") i stedet for at bruge super class'en

    Examples:
    >>> sirRB = Agent("RB", bias = 0.7)
    >>> isinstance(sirRB, RB)
    True
    >>> sirRB.bias
    0.7
    """
    def __init__(self, strategy = None, save_history = False, **kwargs):
        """
        Valid strategies include:
            'WSLS': Win-stay, lose-switch
            'RB': Random bias

        kwargs is specifications passed to the strategy, see examples in class docstring

        TODO:
        - add warning for invalid kwargs
        - make agent callable with strategy
        - add self the ability to add class
        - add remaining strategies:
            - WSLS
            - ToM   
        """
        self.choice = None                                  # the last choice of the agent
        self._start_params = None                           # starting parameters of the agent
        if save_history:
            self.history = pd.DataFrame()                  # history of choices
        else:
            self.history = None
        self.op_choice = None                               # Opponent's choice

        if strategy == "RB":
            self.__class__ = RB
            self.__init__(**kwargs)
        elif strategy == "WSLS":
            self.__class__ = WSLS
            self.__init__(**kwargs)


    def reset(self):
        if self._start_params:
            self.__init__(**self._start_params)
        else:
            self.__init__()


    def get_strategy():
        return self.strategy


    def get_history(self, key = None, format = 'df'):
        """
        Return the agents history. This include only information relevant to the agent
        e.g. for a Random bias (RB) agent only its choice is saved,m while opponent choice is not. 
        Possible keys (dependent on agent):
            'choices'
            'op_choices'
            ...

        Key (list | str): The desired history to be returned
        format (str): The desired fomat the history is return in possible option include;
            'dict': for dictionary
            'df': for pandas dataframe
            'list': for a list. Only valid if key is a str
        """
        if self.get_history is None:
            warn("save_history is unspecified or set to False. Consequently None is returned." + 
                 " Set save_history = True if you want to save agent history", Warning)
            return None
        if key is None:
            _return = self.history
        elif isinstance(key, list) or isinstance(key, str):
            _return = self.history[key]
        else:
            raise Exception("Please input valid key. Key should be either a list or a string." +
                            " Alternatively, key can be left unspecified.")
        if format == "df":
            return _return
        elif format == "dict":
            return dict(_return)
        elif format == "list":
            if isinstance(key, str):
                return list(_return)
            else:
                raise Exception("Format cannot be 'list' if key is given as a list or left unspecified." +
                                " Key should be given as a string e.g. key = 'choice'.")
        else:
            raise Exception("Please input valid format e.g. 'df' or leave format unspecified")


    def _add_to_history(self, **kwargs):
        if self.history is None:
            pass
        elif self.history.empty:
            self.history = pd.DataFrame(data = kwargs, index=[0])
        else:
            self.history = self.history.append(kwargs, ignore_index=True)


    def get_start_params(self):
        return self._start_params



class RB(Agent):
    """
    'RB': Random bias agent

    Examples:
    >>> sirRB = RB(bias = 1, save_history = True) #bias = 1 indicates that probability of choosing 1 is 100%
    >>> sirRB.compete()
    1
    >>> sirRB.get_start_params()
    {'bias': 1, 'save_history': True}
    >>> sirRB.compete()
    1
    >>> sirRB.get_history(key = 'choice',format = "list")
    [1, 1]
    """
    def __init__(self, bias = 0.5, **kwargs):
        self.bias = bias
        self.strategy = "RB"
        super().__init__(**kwargs)
        self._start_params = {'bias': bias, **kwargs}

    def compete(self, **kwargs):
        self.choice = np.random.binomial(1, self.bias)
        self._add_to_history(choice = self.choice)
        return self.choice


class WSLS(Agent):
    """
    'WSLS': Win-stay, lose-switch

    Examples:
    >>> sirWSLS = WSLS()
    >>> sirWSLS.choice = 0  # Manually setting choice
    >>> sirWSLS.compete(outcome = 1)
    0
    """
    def __init__(self, **kwargs):
        self.strategy = "WSLS"
        super().__init__(**kwargs)
        self._start_params = {**kwargs}


    def compete(self, outcome = None, **kwargs):
        if self.choice is None: # if a choice haven't been made: Choose randomly (0 or 1)
            self.choice = np.random.binomial(1, 0.5)
        else:  # if a choice have been made:
            if outcome is None:
                raise TypeError("compete() missing 1 required positional argument: 'outcome'")
            elif outcome == -1:
                self.choice = 1-self.choice
        self._add_to_history(choice = self.choice)
        return self.choice

#%%
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


class Agent_group():
    """

    Examples:
    >>> round_table = Agent_group(agents = ['RB'], parameters = [{'bias': 1}])
    >>> round_table.agent_names
    """
    def __init__(self, agents, parameters):
        # TODO: add option to set payoff matrix and env here
        self.agents = agents
        self.params = parameters
        self.environment = None
            # create unique agent ID's, e.g. a list of 3 RB agent becomes [RB_0, RB_1, RB_2]
        self.agent_names = [agent + '_' + str(idx) for agent in set(agents) for idx in range(agents.count(agent))]
        self._agents = {name: Agent(agent, param) for name, agent, param in zip(self.agent_names, agents, parameters)}

    def get_environment(self):
        if self.environment:
            return self.environment
        else:
            raise Exception('Environment in not set, use set_env() to set environment')

    def get_names(self):
        return self.agent_names

    def get_agent(self, agent):
        if agent in agent_names:
            return self._agents[agent]
        else:
            raise Exception('agent is not in agent names, to get a list of agent names, use get_names()')

    def set_env(self, env):
        """
        set environment of agent group

        env (str): desired environment
        """
        self.environment = env.lower()
        if self.environment in ('rr', 'round_robin', 'round robin'):
            self.pairing = list(combinations(self.agent_names, 2))
        if self.environment in ['random pairs', 'random']:
            L = self.agent_names[:]
            if len(L) % 2 != 0:
                 raise Exception('List is agent in Agent_group should be even if environment is set to random pairs.' + 
                                 "Otherwise one would be lonely and we can't have that.")
            random.shuffle(L)
            self.pairing = list(zip(L[:len(L)//2], L[len(L)//2:]))
        else:
            raise TypeError(f"{env} is not a valid environment.")

    def compete(n_rounds = 10, n_sim = 1, p_matrix = None, reset_agent = True, env = None):
        # TODO: add check to payoffmatrix is already set
        #(agent_0, agent_1, p_matrix, n_rounds = 1, n_sim = None, reset_agent = True, return_val = 'df')
        if self.environment is None and env is None:
            raise TypeError('No env was specified, either specify environment using set_env() or by specifying env for compete()')

        for pair in self.pairing:
            res = compete(self._agents[pair[0]], self._agents[pair[1]], n_rounds, n_sim, reset_agent, return_val = 'df')
            res['agent0'] = pair[0]
            res['agent1'] = pair[1]
            
        if self.environment is None and env is None:
            raise TypeError('No env was specified, either specify environment using set_env() or by specifying env for compete()')

        

('t' or None) is None
tmp = Agent_group(agents = ['RB']*3, parameters = [{'bias': 1}]*3)
tmp.set_env('rr')
tmp.pairing
tmp._agents['RB_0']
tmp.get_environment()

L = list(range(10))


def compete(agent_0, agent_1, p_matrix, n_rounds = 1, n_sim = None, reset_agent = True, return_val = 'df'):
    """
    agent_0 and agent_1 (Agent): objects of class Agent which should compete
    p_matrix (Payoff_matrix | str): The Payoff_matrix in which the agents compete or the name of the payoff matrix
    n_rounds (int): Number of rounds the agents should compete
    return_val ('df' | 'list')
    n_sim (int)

    Examples:
    >>> sirRB = RB(bias = 0.7)
    >>> sirWSLS = WSLS()
    >>> result = compete(sirRB, sirWSLS, p_matrix = "penny_competitive", n_rounds = 10)
    >>> type(result)
    pandas.core.frame.DataFrame
    >>> result.columns
    Index(['round', 'action_agent0', 'action_agent1'], dtype='object')
    """

    if isinstance(p_matrix, str):
        p_matrix = Payoff_matrix(name = p_matrix)
    if reset_agent:
        agent_0.reset()
        agent_1.reset()

    if n_sim:
        if reset_agent is False: # make sure people don't do things they regret
            warn("n_sim is set and reset_agent is False, the agent will maintain their knowledge across" + 
                 "simulations. Is this the intended outcome?", Warning)
        for sim in range(int(n_sim)):
            res = compete(agent_0, agent_1, p_matrix, n_rounds, None, reset_agent, return_val = 'list')
            [tup + (sim,) for tup in res] #add n_sim til list
            if reset_agent:
                agent_0.reset()
                agent_1.reset()
    else:      
        outcome = [None, None]
        result = []
        for i in range(n_rounds):
            a_0 = agent_0.compete(p_matrix = p_matrix, outcome = outcome[0]) #a for action
            a_1 = agent_1.compete(p_matrix = p_matrix, outcome = outcome[1])
            
            outcome = p_matrix.matrix[:, a_0, a_1] #[agent_0, agent_1]
            result.append((i, a_0, a_1))
        if return_val == 'df':
            return pd.DataFrame(result, columns = ['round', 'action_agent0', 'action_agent1'])

    if return_val == 'list':
        return result
    elif return_val == 'df':
        return pd.DataFrame(result, columns = ['n_sim','round', 'action_agent0', 'action_agent1'])
    else:
        raise TypeError("Invalid return_val, please use either 'df' or 'list'")


L = compete(sirRB, sirWSLS, p_matrix = "penny_competitive", n_rounds = 10, return_val = 'list')
list(zip(L, (1,)*10))
(1,)*10
for
L1 = compete(sirRB, sirWSLS, p_matrix = "penny_competitive", n_rounds = 10, return_val = 'df')

for i in range(10):
    player_c = 
    SirToM.compete(c_op = player_c, dsdasds)

### dictionary (should be fetchable from somewhere) - should also be the one valid_agents fetch from
# create dict for payoff matrices 
agent_dict = {'RB': {'name': 'Random Bias',
     'shorthand': 'RB',
     'example': 'RB(bias = 0.5)', 
     'reference': 'Devaine, et al. (2017)', 
     'strategy': 'Chooses 1 randomly based on a probability or bias',
     'function': RB}, 
     'WSLS': {
     'agent': 'Win-stay, lose-switch',
     'shorthand': 'WSLS',
     'example': 'WSLS()', 
     'reference': 'Nowak & Sigmund (1993)', 
     'strategy': 'If it win it chooses the same option again, if it lose it change to another',
     'function': WSLS}}



# %% Run test
if __name__ == "__main__":
  import doctest
  doctest.testmod(verbose=True)