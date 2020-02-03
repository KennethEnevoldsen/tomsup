"""
docstring
"""
#%%
from tomsup.payoffmatrix import PayoffMatrix
import pandas as pd
import numpy as np
from warnings import warn
from itertools import combinations
from tomsup.ktom_functions import k_tom, init_k_tom
from scipy.special import expit as inv_logit
from scipy.special import logit

###################
###___ AGENT ___###
###################

class Agent():
    """
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
        if save_history:
            self.history = pd.DataFrame()                  # history of choices
        else:
            self.history = None
        if strategy:
            if 'TOM' in strategy.upper():
                k = strategy.split('-')[0]
                kwargs['level'] = int(k)
                strategy = strategy.split('-')[1].upper()
            kwargs['save_history'] = save_history
            self.__class__ = eval(strategy)
            self.__init__(**kwargs)



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


    ### Getters
    def get_start_params(self):
        return self._start_params

 
    def get_strategy(self):
        return self.strategy


    def get_choice(self):
        return self.choice


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
    >>> sigmund = WSLS()
    >>> sigmund.choice = 0  # Manually setting choice
    >>> penny = PayoffMatrix(name = "penny_competitive")
    >>> sigmund.compete(op_choice = 1, p_matrix = penny)
    0
    >>> sigmund.choice = 1  # Manually setting choice
    >>> sigmund.compete(payoff = -1)
    0
    """
    def __init__(self, prob_stay = 1, prob_switch = 1, **kwargs):
        self.strategy = "WSLS"
        self.prob_switch = prob_switch
        self.prob_stay = prob_stay
        super().__init__(**kwargs)
        self._start_params = {'prob_stay': prob_stay, 'prob_switch':  prob_switch, **kwargs}

    def compete(self, op_choice, p_matrix, agent, **kwargs):
        if self.choice is None: # if a choice haven't been made: Choose randomly (0 or 1)
            self.choice = np.random.binomial(1, 0.5)
        else:  # if a choice have been made:
            if op_choice is None:
                raise TypeError("compete() missing 1 required positional argument: 'op_choice',"
                                " which should be given for all round except the first.")
            payoff = p_matrix.payoff(self.choice, op_choice, agent)  # calculate payoff of last round
            if payoff < p_matrix().mean(): # if you lost change action (e.g. if payoff is less the mean outcome)
                switch = np.random.binomial(1, self.prob_switch)
                self.choice = switch * (1-self.choice) + (1-switch) * self.choice
            else:  # if you won
                stay = np.random.binomial(1, self.prob_stay)
                self.choice = stay * self.choice + (1-stay) * (1-self.choice)

        self._add_to_history(choice = self.choice)
        return self.choice


class TFT(Agent):
    """
    'TFT': Tit-for-Tat

    Examples:
    >>> shelling = TFT(copy_prob = 1)
    >>> shelling.choice = 1  # manually setting the first choice
    >>>
    """
    def __init__(self, copy_prob = 1, **kwargs):
        self.strategy = "TFT"
        self.copy_prob = copy_prob
        super().__init__(**kwargs)
        self._start_params = {'copy_prob': copy_prob, **kwargs}

    def compete(self, op_choice = None, p_matrix = "prisoners_dilemma", silent = False, **kwargs):
        """
        choice_op (0 <= int <= 1): The choice of the oppenent given af a 1 or a 0
        copy_prop (0 <= float <= 1): The probability of the TFT agent to copy the action of its opponent, hereby introducing noise to
        the original TFT strategy by Shelling (1981).
        """
        if p_matrix != "prisoners_dilemma" and silent is False:
            warn("Tit-for-Tat is designed for the prisoners dilemma"
            " and might not perform well with other payoff matrices.", Warning)
        
        # If a choice haven't been made: Cooperate
        if self.choice is None: 
            self.choice = 1 #assumes 1 to be cooperate
        else:  # if a choice have been made
            if op_choice is None:
                raise TypeError("compete() missing 1 required positional argument: 'op_choice',"
                                " which should be given for all round except the first.")

            self.op_choice = op_choice
            copy = np.random.binomial(1, self.copy_prob)
            #Calculate resulting choice
            self.choice = copy * op_choice + (1 - copy) * (1 - op_choice)
        self._add_to_history(choice = self.choice, choice_op = op_choice)
        return self.choice


class QL(Agent): 
    """
    'QL': Q-learning model by Watkinns (1992)
    """
    def __init__(self, learning_rate = 0.5, b_temp = 0.001, expec_val = [0.5, 0.5],  **kwargs):
        self.strategy = "QL"
        self.learning_rate = learning_rate
        self.expec_val = expec_val
        self.b_temp = b_temp
        super().__init__(**kwargs)
        self._start_params = {'learning_rate': learning_rate,'b_temp': b_temp, 'expec_val': expec_val, **kwargs}

    def compete(self, p_matrix, agent, op_choice = None, **kwargs):
        
        if self.choice and op_choice: # if not first round
            #Calculate whether or not last round was a victory
            payoff = p_matrix.payoff(self.choice, op_choice, agent) #calculate payoff from last round
            if payoff > p_matrix().mean(): #if you won last round
                reward = 1 #Save a win
            else: #and if you lost
                reward = 0 #Save a loss
            #Update perceived values of options. Only the last chosen option is updated
            self.expec_val[self.choice] = self.expec_val[self.choice] + self.learning_rate * (reward - self.expec_val[self.choice])
        elif self.choice and op_choice is None:
            raise TypeError("compete() missing 1 required positional argument: 'op_choice',"
                            " which should be given for all rounds except the first.")
        
        #softmax
        p_self = np.exp(self.expec_val[1] / self.b_temp) / sum(np.exp(np.array(self.expec_val)/self.b_temp))

        #Make choice
        self.choice = np.random.binomial(1, p_self) 

        self._add_to_history(choice = self.choice, expected_value0 = self.expec_val[0], expected_value1 = self.expec_val[1])
        return self.choice
        
    #Define getters
    def get_expected_values(self):
        return self.expec_val

    def get_learning_rate(self):
        return self.learning_rate


class TOM(Agent):
    """
    'TOM': Theory of Mind agent

    Examples:
    >>> Devaine = TOM(level = 0, volatility = -2, b_temp = -1)
    >>> Devaine = TOM(level = 2, volatility = -2, b_temp = -1)
    >>> Devaine = TOM(level = 2, volatility = -2, b_temp = -1, dilution = 0.5, bias = 0.3)
    """
    def __init__(self, level = 0, volatility = -2, b_temp = -1, bias = 0, dilution = None,  **kwargs):
        if level > 5:
            warn("It is quite computationally expensive to run a TOM with a level > 5." + 
                 " Make sure this is your intention.", Warning)


        self.volatility = volatility
        self.b_temp = b_temp
        self.bias = bias
        self.dilution = dilution
        self.level = level
        self.strategy = str(level) + '-TOM'

        priors = 'default' if 'priors' not in kwargs else kwargs['priors']

        params = {'volatility': volatility, 'b_temp': b_temp} 
        if dilution is not None:
            params['dilution'] = dilution
        if bias is not None:
            params['bias'] = bias
        
        self.params = params
        self.internal = init_k_tom(params, level, priors)

        super().__init__(**kwargs)

        self._start_params = {'volatility': volatility, 'level': level,'b_temp': b_temp, 
                              'bias': bias, 'dilution': dilution, **kwargs}


    def compete(self, p_matrix, agent, op_choice = None, **kwargs):
        """
        
        """
        self.op_choice = op_choice
        self.choice, self.internal = k_tom(
                                        self.internal,
                                        self.params,
                                        self.choice,
                                        op_choice,
                                        self.level,
                                        agent,
                                        p_matrix,
                                        **kwargs)
        self._add_to_history(choice = self.choice, internal_states = self.internal)
        return self.choice


    # Define getters
    def get_volatility(self):
        return self.volatility

    def get_behav_temperature(self):
        return self.b_temp

    def get_bias(self):
        if self.bias is None:
             print("TOM does not have a bias.")
        return self.bias
    
    def get_dilution(self):
        if self.get_dilution is None:
            print("TOM does not have a dilution parameter.")
        return self.get_dilution

    def get_level(self):
        return self.level

    def get_internal_states(self):
        return self.internal

    def get_parameters(self):
        return self.params

    def print_internal(self, keys_to_print = None, silent = False, print_as_str = False):
        """
        keys_to_print (list) is the keys which you desire to print. If key is None all keys will be printed.
        """

        # Convert all elements to string
        if keys_to_print:
            keys_to_print = [str(key) for key in keys_to_print]
            # if sum(key not in keys_to_print for key in ['opponent_states', 'own_states']) != 0:
            #     if not silent:
            #         print("You haven't added 'opponent_states' and/or 'own_states' to keys_to_print." + 
            #               " If any of the keys_to_print is within one")

        def _print_internal(d, n = 0, keys_to_print = None):
            for key in d:
                p_key = str(key) + '-ToM' if isinstance(key, int) else key
                p_str = '|   '*n + str(p_key)
                # print('|---'*n, key, sep = "", end = '')
                if isinstance(d[key], dict) is False:
                    x = d[key].tolist() if isinstance(d[key], np.ndarray) else d[key]
                    p_str = p_str + ":" + " "*(12-len(p_key)) + str(x)
                if keys_to_print is None or str(key) in keys_to_print:
                    #continue
                    print(p_str)

                if isinstance(d[key], dict):
                    _print_internal(d[key], n+1, keys_to_print)

        if print_as_str:
            return  str(self.internal)
        _print_internal(self.internal, n = 0, keys_to_print = keys_to_print)



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
        if start_params:
            if len(agents) != len(start_params):
                raise ValueError("the length of agents is not equal to the length of starting parameters.")
        else:
            start_params = [{}] * len(agents)
        self.start_params = start_params
        self.environment = None
        self.pairing = None
            # create unique agent ID's, e.g. a list of 3 RB agent becomes [RB_0, RB_1, RB_2]
        self.agent_names = [agent + '_' + str(idx) for agent in set(agents) for idx in range(agents.count(agent))]
        self._agents = {name: Agent(name.split('_')[0], **param) for name, param in zip(self.agent_names, start_params)}

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
        set environment of agent group.

        env (str): The string for the environment you wish to set.
        Valid environment strings include:
            'round_robin': Matches all participant against all others
            'rondom_pairs': Combines the agent in random pairs (the number of agent must be even)
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
            raise TypeError(f"{env} is not a valid environment. Use help() to see valid environments")

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

        
    def __str__(self):
        header = f"<Class AgentGroup, envinment = {self.environment} \n\n" 
        info = "\n".join(("\t | \t".join(str(ii) for ii in i)) for i in list(zip(self.agent_names, self.start_params)))
        return header + info
        


###################
###___ UTILS ___###
###################

def compete(agent_0, agent_1, p_matrix, n_rounds = 1, n_sim = None, reset_agent = True, return_val = 'df', save_history = False, silent = True):
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
            res = compete(agent_0, agent_1, p_matrix, n_rounds, None, reset_agent, return_val = 'list', save_history=save_history)
            result += [(sim,) + tup for tup in res] #add n_sim til list and 'append' to results
            if reset_agent and sim != n_sim-1:
                agent_0.reset()
                agent_1.reset()

    else:      
        c_0, c_1 = None, None
        result = []
        for i in range(n_rounds):
            c_0_prev, c_1_prev = c_0, c_1 
            c_0 = agent_0.compete(p_matrix = p_matrix, agent = 0, op_choice = c_1_prev) #c for choice
            c_1 = agent_1.compete(p_matrix = p_matrix, agent = 1, op_choice = c_0_prev)

            payoff = (p_matrix.payoff(c_0, c_1, agent = 0), 
                      p_matrix.payoff(c_0, c_1, agent = 1))

            if save_history:
                history0 = agent_0.get_history(key = "internal_states", format = "list")[-1]
                history1 = agent_1.get_history(key = "internal_states", format = "list")[-1]
                result.append((i, c_0, c_1, payoff[0], payoff[1], history0, history1))  
            else:
                result.append((i, c_0, c_1, payoff[0], payoff[1]))
        if return_val == 'df':
            if save_history:
                return pd.DataFrame(result, columns = ['round', 'choice_agent0', 'choice_agent1', 'payoff_agent0', 'payoff_agent1', 'history_agent0', 'history_agent1'])
            return pd.DataFrame(result, columns = ['round', 'choice_agent0', 'choice_agent1', 'payoff_agent0', 'payoff_agent1'])

    if return_val == 'list':
        return result
    elif return_val == 'df':
        if save_history:
            return pd.DataFrame(result, columns = ['n_sim', 'round', 'choice_agent0', 'choice_agent1', 'payoff_agent0', 'payoff_agent1', 'history_agent0', 'history_agent1'])
        return pd.DataFrame(result, columns = ['n_sim','round', 'choice_agent0', 'choice_agent1', 'payoff_agent0', 'payoff_agent1'])
    else:
        raise TypeError("Invalid return_val, please use either 'df' or 'list'")


#%%
if __name__ == "__main__":
  import doctest
  doctest.testmod(verbose=True)