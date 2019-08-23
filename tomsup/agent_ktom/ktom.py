#%% Overall
class TOM(Agent):
    """
    'TOM': Theory of Mind agent

    Examples:
    k_tom(
    prev_internal_states,
    params,
    self_choice,
    op_choice,
    level,
    agent,
    p_matrix,
    **kwargs)


    prepare_k_tom(params, level, priors = 'default')





    >>> sirTOM = RB(
        volatility = -2,
        temperature = -10,
        bias = 0,
        dilution = -1,
        save_history = True)
        #volatility = 1 indicates a standard amount of assumed volatility in the opponent
        #temperature = -10 indicates a standard amount of behavioural exploration
        #bias = 0 indicates no bias for either choice
        #dilution = -1 indicates a standard amount of estimation forgetting
    >>> sirTOM.compete()
    1
    >>> sirTOM.get_start_params()
    {'volatility': 1, 'temperature': 1, 'bias': 0, 'dilution': -1, 'save_history': True}
    >>> sirTOM.compete()
    1
    >>> sirTOM.get_history(key = 'choice',format = "list")
    [1, 1]
    """
    def __init__(self, k, volatility = -2, temperature = -10, bias = 0, dilution = -1, **kwargs):
        self.volatility = volatility
        self.temperature = temperature
        self.bias = bias
        self.dilution = dilution
        self.strategy = 'TOM'
        self.internal = self.init_tom() #TODO What's the input
        super().__init__(**kwargs)
        self._start_params = {'volatility':volatility, 'temperature': temperature, 'bias': bias, 'dilution': dilution, **kwargs}

    def compete(self, **kwargs):
        self.choice = TOM_strategy()
        self._add_to_history(choice = self.choice)
        return self.choice

    def init_tom()
