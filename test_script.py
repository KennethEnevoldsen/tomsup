"""
Dette er et test script til at tjekke om pakken fungerer
"""
#%%
import tomsup as ts

## Get a list of valid agents
ts.valid_agents()


#%%
## tomsup - create a single agent
    # There is two ways to setup an agent (these are equivalent)
sirRB = ts.Agent(strategy = "RB", save_history = True, bias = 0.7)
sirRB = ts.RB(bias = 0.7, save_history = True)
isinstance(sirRB, ts.Agent)  # sirRB is an agent 
type(sirRB)  # of supclass RB

choice = sirRB.compete()



print(f"SirRB chose {choice} and his probability for choosing 1 was {sirRB.bias}.")

## Create a group of agents
desired_agent = ['RB']*3 + ['WSLS']*3  # list of desired agent strategies
print(desired_agent)
parameters = [{'bias': 0.7}]*3 + [{}]*3  # list of their starting parameters in dictionary format
print(parameters)

group = ts.create_agents(desired_agent, start_params = parameters)
print(type(group))

# inspect group
print(group.get_names())

## Set environment
group.set_env('round_robin')
print(group.get_environment_name())
print(group.get_environment())

##
out = group.compete(n_rounds = 10, n_sim = 1, p_matrix = 'penny_competitive')

#%%

#%%
# Definig your own agent:

"""
let's say I want to create an reverse win-stay loose-switch agent.
I.e. if it win it changes choice and if it loose it changes (win-switch, lose-stay).
you can create a supclass using the ts.Agent class, for example
"""
import numpy as np

class ReversedWSLS(ts.Agent):
    """
    ReversedWSLS: Win-switch, lose-stay.

    This agent is a reversed win-stay, lose-switch agent, which ...

    Examples:
    >>> Arnault = ReversedWSLS(first_move = 1)
    >>> Arnault.compete(op_choice = None, p_matrix = penny)
    1
    >>> 
    """
    def __init__(self, first_move, **kwargs): #initalize the agent
        self.strategy = "ReversedWSLS"  # set the strategy name

        # set internal parameters
        self.first_move = first_move

        super().__init__(**kwargs)  # pass additional argument the ts.Agent class (could e.g. include 'save_history = True')
        self._start_params = {'first_move': first_move, **kwargs}  # save any starting parameters used when the agent is reset


    def compete(self, op_choice = None, p_matrix):
        if self.choice is None: # if a choice haven't been made: Choose the redifined first move
            self.choice = self.first_move #fetch from self
        else:  # if a choice have been made:
            payoff = p_matrix.payoff(self.choice, op_choice, 0)  # calculate payoff of last round
            if payoff == 1: # if the agent won then switch
                self.choice = 1-self.choice  # save the choice in self (for next round)
                                             # also save any other internal states which you might 
                                             # want the agent to keep for next round in self
        self._add_to_history(choice = self.choice) # save action and (if any) internal states in history
                                                   # note that _add_to_history() is not intented for 
                                                   # later use within the agent
        return self.choice  # return choice

    # define any additional function you wish the class should have
    def get_first_move(self):
        return self.first_move




class WSLS(ts.Agent):
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
        super().__init__(**kwargs)
        self._start_params = {'prob_stay': prob_stay, 'prob_switch':  prob_switch, **kwargs}

    def compete(self, op_choice, p_matrix):
        if self.choice is None: # if a choice haven't been made: Choose randomly (0 or 1)
            self.choice = np.random.binomial(1, 0.5)
        else:  # if a choice have been made:
            if op_choice is None:
                raise TypeError("compete() missing 1 required positional argument: 'op_choice',"
                                " which should be given for all round except the first.")
            payoff = p_matrix.payoff(self.choice, op_choice, 0)  # calculate payoff of last round
            if payoff < p_matrix().mean(): # if you lost change action (e.g. if payoff is less the mean outcome)
                switch = np.random.binomial(1, self.prob_switch)
                self.choice = switch * (1-self.choice) + (1-switch) * self.choice
            else:  # if you won
                stay = np.random.binomial(1, self.prob_stay)
                self.choice = stay * self.choice + (1-stay) * (1-self.choice)

        self._add_to_history(choice = self.choice)
        return self.choice


class TFT(ts.Agent):
    """
    'TFT': Tit-for-Tat

    Examples:
    >>> shelling = ts.TFT(copy_prob = 1)
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
            warn("Tit-for-Tat is designed for the prisoners dilemma" +
            " and might not perform as intended with other payoff matrices.", Warning)
        if self.choice is None: # if a choice haven't been made: Choose randomly (0 or 1)
            self.choice = 1 #assumes 1 to be cooperate
        else:  # if a choice have been made apply the TFT strategy
            if choice_op is None:
                raise TypeError("choice_op is None, but it is not the first round the agent played." +
                "Try resetting the agent, e.g. agent.reset()")
            self.op_choice = op_choice
            copy = np.random.binomial(1, self.copy_prob)
            #Calculate resulting choice
            choice = copy * op_choice + (1 - copy) * (1 - op_choice)
        self._add_to_history(choice = self.choice, choice_op = op_choice)
        return self.choice



    def get_copy_prop(self):
        return self.copy_prob


print(np.nan)

from scipy.special import expit as inv_logit

inv_logit