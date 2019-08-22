"""
dogstring
"""
#%%
from tomsup.payoffmatrix import PayoffMatrix
import pandas as pd
import numpy as np
from warnings import warn
from itertools import combinations

###################
###___ AGENT ___###
###################

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
        elif strategy is None:
            self.strategy = None


    def reset(self):
        if self._start_params:
            self.__init__(**self._start_params)
        else:
            self.__init__()

    def _add_to_history(self, **kwargs):
        if self.history is None:
            pass
        elif self.history.empty:
            self.history = pd.DataFrame(data = kwargs, index=[0])
        else:
            self.history = self.history.append(kwargs, ignore_index=True)


    # Get
    def get_start_params(self):
        return self._start_params

 
    def get_strategy(self):
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


class RB(Agent):
    """
    'RB': Random bias agent
    bias (0 <= float <= 1): Is the probability of the agent to choose 1
    var: The sampling variance of the bias, default to 0. The variance is measured in standard deviation of logit(bias) (logodds).
    
    Given a non-zero variance, the bias sampled from a normal distribution with a mean of the bias in logodds and a SD of the 
    var. It is then transformed into a 0-1 scale using the inverse logit function.

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
    def __init__(self, bias = 0.5, var = 0, **kwargs):
        self._start_params = {'bias': bias, 'var': var, **kwargs}
        self.bias = inv_logit(np.random.normal(logit(bias), scale=var))
        self.var = var
        self.strategy = "RB"
        super().__init__(**kwargs)


    def compete(self, **kwargs):
        self.choice = np.random.binomial(1, self.bias)
        self._add_to_history(choice = self.choice)
        return self.choice


    def get_bias(self):
        return self.bias


class WSLS(Agent):
    """
    'WSLS': Win-stay, lose-switch

    Examples:
    >>> sirWSLS = WSLS()
    >>> sirWSLS.choice = 0  # Manually setting choice
    >>> sirWSLS.compete(payoff = 1)
    0
    >>> sirWSLS.choice = 1  # Manually setting choice
    >>> sirWSLS.compete(payoff = -1)
    0
    """
    def __init__(self, **kwargs):
        self.strategy = "WSLS"
        super().__init__(**kwargs)
        self._start_params = {**kwargs}


    def compete(self, payoff = None, **kwargs):
        if self.choice is None: # if a choice haven't been made: Choose randomly (0 or 1)
            self.choice = np.random.binomial(1, 0.5)
        else:  # if a choice have been made:
            if payoff is None:
                raise TypeError("compete() missing 1 required positional argument: 'payoff'")
            elif payoff == -1:
                self.choice = 1-self.choice
        self._add_to_history(choice = self.choice)
        return self.choice


#########################
###___ AGENT GROUP ___###
#########################

class AgentGroup():
    """

    Examples:
    >>> round_table = AgentGroup(agents = ['RB']*2, start_params = [{'bias': 1}]*2)
    >>> round_table.agent_names
    ['RB_0', 'RB_1']
    >>> RB_0 = round_table.get_agent('RB_0') # extract an agent
    >>> RB_0.bias == 1 # should naturally be 1, as we specified it 
    True
    >>> round_table.set_env('round_robin')
    >>> result = round_table.compete(p_matrix = "penny_competitive", n_rounds = 100, n_sim = 10)
    Currently the pair, ('RB_0', 'RB_1'), is competing for 10 simulations, each containg 100 rounds.
        Running simulation 1 out of 10
        Running simulation 2 out of 10
        Running simulation 3 out of 10
        Running simulation 4 out of 10
        Running simulation 5 out of 10
        Running simulation 6 out of 10
        Running simulation 7 out of 10
        Running simulation 8 out of 10
        Running simulation 9 out of 10
        Running simulation 10 out of 10
    Simulation complete
    >>> result.shape[0] == 10*100 # As there is 10 simulations each containing 100 round
    True
    >>> result['payoff_agent0'].mean() == 1  # Given that both agents have always choose 1, it is clear that agent0 always win, when playing the competitive pennygame   
    True 
    """
    def __init__(self, agents, start_params = None):
        # TODO: add option to set payoff matrix and env here
        self.agents = agents
        if start_params is None:
            start_params = [{}] * len(agents)
        self.start_params = start_params
        self.environment = None
        self.pairing = None
            # create unique agent ID's, e.g. a list of 3 RB agent becomes [RB_0, RB_1, RB_2]
        self.agent_names = [agent + '_' + str(idx) for agent in set(agents) for idx in range(agents.count(agent))]
        self._agents = {name: Agent(agent, **param) for name, agent, param in zip(self.agent_names, agents, start_params)}

    def get_environment_name(self):
        if self.environment:
            return self.environment
        else:
            raise Exception('Environment in not set, use set_env() to set environment')

    def get_environment(self):
        if self.pairing:
            return self.pairing
        else:
            raise Exception('Environment in not set, use set_env() to set environment')

    def get_names(self):
        return self.agent_names

    def get_agent(self, agent):
        if agent in self.agent_names:
            return self._agents[agent]
        else:
            raise Exception('agent is not in agent names, to get a list of agent names, use get_names()')

    def set_env(self, env):
        """
        set environment of agent group

        env (str): desired environment
        """
        self.environment = env.lower()
        if self.environment == 'round_robin':
            self.pairing = list(combinations(self.agent_names, 2))
        elif self.environment == 'random_pairs':
            L = self.agent_names[:]
            if len(L) % 2 != 0:
                 raise Exception('List is agent in Agent_group should be even if environment is set to random pairs.' + 
                                 "Otherwise one would be lonely and we can't have that.")
            random.shuffle(L)
            self.pairing = list(zip(L[:len(L)//2], L[len(L)//2:]))
        else:
            raise TypeError(f"{env} is not a valid environment.")

    def compete(self, p_matrix, n_rounds = 10, n_sim = 1, reset_agent = True, env = None, silent = False):
        # TODO: add check to payoffmatrix is already set
        #(agent_0, agent_1, p_matrix, n_rounds = 1, n_sim = None, reset_agent = True, return_val = 'df')
        if self.environment is None and env is None:
            raise TypeError('No env was specified, either specify environment using set_env() or by specifying env for compete()')

        result = []
        for pair in self.pairing:
            if not silent:
                print(f"Currently the pair, {pair}, is competing for {n_sim} simulations, each containg {n_rounds} rounds.")
            res = compete(self._agents[pair[0]], self._agents[pair[1]], p_matrix = p_matrix, n_rounds = n_rounds, 
                          n_sim = n_sim, reset_agent = reset_agent, return_val = 'df', silent = silent) 
            res['agent0'] = pair[0]
            res['agent1'] = pair[1]
            result.append(res)
        
        if not silent:
            print("Simulation complete")
        return pd.concat(result) #concatenate into one df
        


###################
###___ UTILS ___###
###################

def compete(agent_0, agent_1, p_matrix, n_rounds = 1, n_sim = None, reset_agent = True, return_val = 'df', silent = True):
    """
    agent_0 and agent_1 (Agent): objects of class Agent which should compete
    p_matrix (PayoffMatrix | str): The PayoffMatrix in which the agents compete or the name of the payoff matrix
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
    Index(['round', 'action_agent0', 'action_agent1', 'payoff_agent0',
       'payoff_agent1'],
      dtype='object')
    >>> result = compete(sirRB, sirWSLS, p_matrix = "penny_competitive", n_rounds = 10, n_sim = 3, return_val = 'list')
    >>> len(result) == 3*10
    True
    >>> result = compete(sirRB, sirWSLS, p_matrix = "penny_competitive", n_rounds = 100, n_sim = 3, return_val = 'df', silent = False)
        Running simulation 1 out of 3
        Running simulation 2 out of 3
        Running simulation 3 out of 3
    >>> result['payoff_agent1'].mean() > 0  # We see that the WSLS() on average win more than it lose vs. the biased agent (RB)
    True
    """

    if isinstance(p_matrix, str):
        p_matrix = PayoffMatrix(name = p_matrix)
    if reset_agent:
        agent_0.reset()
        agent_1.reset()

    if n_sim:
        if reset_agent is False: # make sure people don't do things they regret
            warn("n_sim is set and reset_agent is False, the agent will maintain their knowledge across" + 
                 "simulations. Is this the intended outcome?", Warning)
        result = []
        for sim in range(int(n_sim)):
            if not silent:
                print(f"\tRunning simulation {sim+1} out of {n_sim}")
            res = compete(agent_0, agent_1, p_matrix, n_rounds, None, reset_agent, return_val = 'list')
            result += [(sim,) + tup for tup in res] #add n_sim til list and 'append' to results
            if reset_agent:
                agent_0.reset()
                agent_1.reset()

    else:      
        payoff = [None, None]
        result = []
        for i in range(n_rounds):
            a_0 = agent_0.compete(p_matrix = p_matrix, payoff = payoff[0]) #a for action
            a_1 = agent_1.compete(p_matrix = p_matrix, payoff = payoff[1])
            
            payoff = p_matrix()[:, a_0, a_1] #[agent_0, agent_1]
            result.append((i, a_0, a_1, payoff[0], payoff[1]))
        if return_val == 'df':
            return pd.DataFrame(result, columns = ['round', 'action_agent0', 'action_agent1', 'payoff_agent0', 'payoff_agent1'])

    if return_val == 'list':
        return result
    elif return_val == 'df':
        return pd.DataFrame(result, columns = ['n_sim','round', 'action_agent0', 'action_agent1', 'payoff_agent0', 'payoff_agent1'])
    else:
        raise TypeError("Invalid return_val, please use either 'df' or 'list'")

def logit(p):
    return np.log(p) - np.log(1 - p)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))


if __name__ == "__main__":
  import doctest
  doctest.testmod(verbose=True)


#%%