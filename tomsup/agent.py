"""
docstring
"""
from typing import Callable, List, Optional, Tuple, Union
from warnings import warn
from itertools import combinations
import random

from joblib import Parallel, delayed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tomsup.ktom_functions import k_tom, init_k_tom, inv_logit
from tomsup.payoffmatrix import PayoffMatrix
from tomsup.plot import (
    choice,
    score,
    ResultsDf,
    plot_history,
    plot_p_k,
    plot_p_op_1,
    plot_p_self,
    plot_op_states,
    plot_heatmap,
)

from wasabi import msg


###################
# ___ AGENT ___
###################


class Agent:
    """
    Agent is super (parent) class for creating agents in tomsup.
    """

    def __init__(
        self, strategy: Optional[str] = None, save_history: bool = False, **kwargs
    ):
        """
        Args:
            strategy (Optional[str], optional): The strategy of the agent you wish to create. Defaults to None.
                It is recommended to use create_agents() to create instead of the Agent class
            save_history (bool, optional): Should the history of the agent be saved. Defaults to False.
            kwargs (dict) The specifications passed to the strategy
        """
        self.choice = None  # the last choice of the agent
        if save_history:
            self.history = pd.DataFrame()  # history of choices
        else:
            self.history = None
        if strategy:
            if "TOM" in strategy.upper():
                k = strategy.split("-")[0]
                kwargs["level"] = int(k)
                strategy = strategy.split("-")[1].upper()
            kwargs["save_history"] = save_history
            self.__class__ = eval(strategy)
            self.__init__(**kwargs)

    def reset(self, save_history: Optional[bool] = None):
        """resets the agent to its starting parameters

        Args:
            save_history (Optional[bool], optional): Should the agent history be saved? Defaults to None, which keeps the previous state.
        """
        if self._start_params:
            if save_history is not None:
                self._start_params["save_history"] = save_history
            self.__init__(**self._start_params)
        else:
            self.__init__()

    def _add_to_history(self, **kwargs):
        if self.history is None:
            pass
        elif self.history.empty:
            self.history = pd.DataFrame(data=kwargs, index=[0])
            if self.strategy.split("-")[-1] == "TOM":
                self.history = self.history.append(kwargs, ignore_index=True)
                self.history = self.history.drop([0]).reset_index()
        else:
            self.history = self.history.append(kwargs, ignore_index=True)

    # Getters
    def get_start_params(self) -> dict:
        """
        Returns:
            dict: The starting parameters of the agent.
        """
        return self._start_params

    def get_strategy(self) -> str:
        """
        Returns:
            str: The strategy of the agent
        """
        return self.strategy

    def get_choice(self) -> int:
        """
        Returns:
            int: the agents choice in the previous round
        """
        return self.choice

    def get_history(
        self, key: Optional[str] = None, format: str = "df"
    ) -> Union[dict, pd.DataFrame, list]:
        """Return the agents history. This include only information relevant to
        the agent. E.g. for a Random bias (RB) agent only its choice is saved, while
        opponent choice is not as it used by the agent.

        Args:
            key (Optional[str], optional): The item of interest in the history. Defaults to None, which returns all the entire history.
            format (str, optional): Format of the return, options include "list", "dict" and "df", which is a pandas dataframe. Defaults to "df".

        Returns:
            Union[dict, pd.DataFrame, list]: The history in the specified format
        """

        if self.history is None:
            raise Exception(
                "save_history is unspecified or set to False. \
                 Consequently you can't get history. \
                 Set save_history=True if you want to save agent history"
            )
        if key is None:
            _return = self.history
        elif isinstance(key, (list, str)):
            _return = self.history[key]
        else:
            raise Exception(
                "Please input valid key. Key should be either a list or a string. \
                            Alternatively, key can be left unspecified."
            )
        if format == "df":
            return _return
        if format == "dict":
            return dict(_return)
        if format == "list":
            return list(_return)
        raise Exception(
            "Please input valid format e.g. 'df' or 'list' or \
                leave format unspecified"
        )

    # plotters
    def plot_choice(self, show: bool = True) -> None:
        """Plot the choices of the agent
        Args:
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        df = self.get_history()
        plt.plot(df.index, df["choice"], color="lightblue", linewidth=4)
        plt.xlabel("Round")
        plt.ylabel("Choice")
        plt.ylim(0, 1)
        if show is True:
            plt.show()

    def plot_internal(self, fun: Callable, show: bool = True) -> None:
        """
        Function for plotting internal states of agent

        Args:
            fun (Callable): a function which to use to extract from the internal states dict
            show (bool): Should it run plt.show at the end of plotting? Default to True

        Examples:
            >>> # plotting the est. probability of opp. choosing one over trials
            >>> tom1.plot_internal(fun=lambda internal_states: internal_states["own_states"]["p_op"])
            >>> # plotting the agent belief about its opponents theory of mind level (p_k)
            >>> # probability of sophistication level k=0
            >>> tom2.plot_internal(fun=lambda internal_states: internal_states["own_states"]["p_k"][0])
            >>> # probability of sophistication level k=1
            >>> tom2.plot_internal(fun=lambda internal_states: internal_states["own_states"]["p_k"][1])
        """
        df = self.get_history()
        df["extracted"] = df["internal_states"].apply(fun)
        plt.plot(df.index, df["extracted"], color="lightblue", linewidth=4)
        if show is True:
            plt.show()


class RB(Agent):
    """
    'RB': Random bias agent

    Examples:
        >>> rb = ts.agent.RB(bias = 1, save_history = True)
        >>> rb.compete()
        1
        >>> rb.get_start_params()
        {'bias': 1, 'save_history': True}
        >>> rb.compete()
        1
        >>> rb.get_history(key='choice', format="list")
        [1, 1]
    """

    def __init__(self, bias: float = 0.5, **kwargs) -> Agent:
        """
        Args:
            bias (float, optional): The probability of the agent to choose 1. Defaults to 0.5.
        """
        self._start_params = {"bias": bias, **kwargs}
        self.bias = bias
        self.strategy = "RB"
        super().__init__(**kwargs)

    def compete(self, **kwargs) -> int:
        """
        Returns:
            int: The choice of the agent
        """
        self.choice = np.random.binomial(1, self.bias)
        self._add_to_history(choice=self.choice)

        return self.choice

    def get_bias(self) -> float:
        """
        Returns:
            float: The bias of the agent
        """
        return self.bias


class WSLS(Agent):
    """
    'WSLS': Win-stay, lose-switch is an agent which employ a simple heuristic.
    It simply takes the same choice if it wins and switches when it looses.

    Examples:
        >>> sigmund = WSLS()
        >>> sigmund.choice = 0  # Manually setting choice
        >>> penny = PayoffMatrix(name="penny_competitive")
        >>> sigmund.compete(op_choice=1, p_matrix=penny)
        0
        >>> sigmund.choice = 1  # Manually setting choice
        >>> sigmund.compete(op_choice=0)
        0
    """

    def __init__(self, prob_stay: float = 1, prob_switch: float = 1, **kwargs) -> Agent:
        """
        Args:
            prob_stay (float, optional): he probability to stay if the agent wins. Defaults to 1.
            prob_switch (float, optional): The probability to switch if the agent loose. Defaults to 1.

        Returns:
            Agent: The WSLS agent
        """

        self.strategy = "WSLS"
        self.prob_switch = prob_switch
        self.prob_stay = prob_stay
        super().__init__(**kwargs)
        self._start_params = {
            "prob_stay": prob_stay,
            "prob_switch": prob_switch,
            **kwargs,
        }

    def compete(
        self, op_choice: Optional[int], p_matrix: PayoffMatrix, agent: int, **kwargs
    ) -> int:
        """
        Args:
            op_choice (Optional[int]): The choice of the opponent, should be None in the first round.
            p_matrix (PayoffMatrix): The payoff matrix which the agent plays in
            agent (int): The agent role (either 0, 1) in which the agent have in the payoff matrix.

        Returns:
            int: The choice of the agent
        """
        # if a choice haven't been made: Choose randomly (0 or 1)
        if self.choice is None:
            self.choice = np.random.binomial(1, 0.5)
        else:  # if a choice have been made:
            if op_choice is None:
                raise TypeError(
                    "compete() missing 1 required positional \
                                 argument: 'op_choice', which should be given \
                                 for all round except the first."
                )
            payoff = p_matrix.payoff(self.choice, op_choice, agent)
            # if you lost change action (e.g. if payoff is less the mean
            # outcome)
            if payoff < p_matrix().mean():
                switch = np.random.binomial(1, self.prob_switch)
                self.choice = switch * (1 - self.choice) + (1 - switch) * self.choice
            else:  # if you won
                stay = np.random.binomial(1, self.prob_stay)
                self.choice = stay * self.choice + (1 - stay) * (1 - self.choice)

        self._add_to_history(choice=self.choice)
        return self.choice


class TFT(Agent):
    """
    'TFT': Tit-for-Tat is a heuritstic theory of mind strategy. An agent using
    this strategy will initially cooperate, then subsequently replicate an
    opponent's previous action.
    If the opponent previously was cooperative, the TFT agent is cooperative.

    Examples:
        >>> shelling = ts.agent.TFT()
        >>> p_dilemma = ts.PayoffMatrix(name="prisoners_dilemma")
        >>> shelling.compete(op_choice=1, p_matrix=p_dilemma)
        1
        >>> shelling.compete(op_choice=0, p_matrix=p_dilemma)
        0
    """

    def __init__(self, copy_prob: float = 1, **kwargs) -> Agent:
        """
        Args:
            copy_prob (float, optional): the probability that the TFT copies the behaviour
                of the opponent, hereby introducing noise to the original TFT strategy by
                Shelling (1981). Defaults to 1.

        Returns:
            Agent: The TFT agent
        """
        self.strategy = "TFT"
        self.copy_prob = copy_prob
        super().__init__(**kwargs)
        self._start_params = {"copy_prob": copy_prob, **kwargs}

    def compete(
        self,
        op_choice: Optional[int] = None,
        p_matrix: PayoffMatrix = PayoffMatrix("prisoners_dilemma"),
        verbose: bool = True,
        **kwargs,
    ) -> int:
        """
        Args:
            op_choice (Optional[int]): The choice of the opponent, should be None in the first round.
            p_matrix (PayoffMatrix): The payoff matrix which the agent plays in. Defualt to the prisoners dilemma.
            agent (int): The agent role (either 0, 1) in which the agent have in the payoff matrix.

        Returns:
            int: The choice of the agent
        """
        if p_matrix.name != "prisoners_dilemma" and verbose:
            warn(
                "Tit-for-Tat is designed for the prisoners dilemma and might \
                  not perform well with other payoff matrices.",
                Warning,
            )

        # If a choice haven't been made: Cooperate
        if self.choice is None:
            self.choice = 1  # assumes 1 to be cooperate
        else:  # if a choice have been made
            if op_choice is None:
                raise TypeError(
                    "compete() missing 1 required positional \
                                 argument: 'op_choice', which should be given \
                                 for all round except the first."
                )

            self.op_choice = op_choice
            copy = np.random.binomial(1, self.copy_prob)
            # Calculate resulting choice
            self.choice = copy * op_choice + (1 - copy) * (1 - op_choice)
        self._add_to_history(choice=self.choice)
        return self.choice


class QL(Agent):
    """
    'QL': The Q-learning model by Watkinns (1992)
    """

    def __init__(
        self,
        learning_rate: float = 0.5,
        b_temp: float = 0.01,
        expec_val: Tuple[float, float] = (0.5, 0.5),
        **kwargs,
    ):
        """
        Args:
            learning_rate (float, optional): The degree to which the agent learns. If the learning rate 0 the agent will not learn.
                Defaults to 0.5.
            b_temp (float, optional): The behavioural temperature of the Q-Learning agent. Defaults to 0.01.
            expec_val (Tuple[float, float], optional): The preference for choice 0 and 1. Defaults to (0.5, 0.5).
        """
        self.strategy = "QL"
        self.learning_rate = learning_rate
        self.expec_val = list(expec_val)
        self.b_temp = b_temp
        super().__init__(**kwargs)
        self._start_params = {
            "learning_rate": learning_rate,
            "b_temp": b_temp,
            "expec_val": list(expec_val),
            **kwargs,
        }

    def compete(
        self, op_choice: Optional[int], p_matrix: PayoffMatrix, agent=int, **kwargs
    ) -> int:
        """
        Args:
            op_choice (Optional[int]): The choice of the opponent, should be None in the first round.
            p_matrix (PayoffMatrix): The payoff matrix which the agent plays in
            agent (int): The agent role (either 0, 1) in which the agent have in the payoff matrix.

        Returns:
            int: The choice of the agent
        """
        if self.choice and op_choice:  # if not first round
            # Calculate whether or not last round was a victory
            payoff = p_matrix.payoff(self.choice, op_choice, agent)
            if payoff > p_matrix().mean():  # if you won last round
                reward = 1  # Save a win
            else:  # and if you lost
                reward = 0  # Save a loss
            # Update perceived values of options. Only the last chosen option
            # is updated
            self.expec_val[self.choice] = self.expec_val[
                self.choice
            ] + self.learning_rate * (reward - self.expec_val[self.choice])
        elif self.choice and op_choice is None:
            raise TypeError(
                "compete() missing 1 required positional argument: \
                            'op_choice', which should be given for all rounds \
                            except the first."
            )

        # Softmax
        p_self = np.exp(self.expec_val[1] / self.b_temp) / sum(
            np.exp(np.array(self.expec_val) / self.b_temp)
        )

        # Make choice
        self.choice = np.random.binomial(1, p_self)

        self._add_to_history(
            choice=self.choice,
            expected_value0=self.expec_val[0],
            expected_value1=self.expec_val[1],
        )
        return self.choice

    # Define getters
    def get_expected_values(self) -> Tuple[float, float]:
        """
        Returns:
            Tuple[float, float]: The preference for choice 0 and 1.
        """
        return tuple(self.expec_val)

    def get_learning_rate(self) -> float:
        """
        Returns:
            float: The learning rate of the agent
        """
        return self.learning_rate


class TOM(Agent):
    """
    This Theory of Mind agent is the variational implementation of the
    recursive ToM agent initially proposed by Devaine (2014), but have
    been further developed since.

    It recursively estimates its opponent and estimate their beliefs
    about itself.
    """

    def __init__(
        self,
        level: int,
        volatility: float = -2,
        b_temp: float = -1,
        bias: Optional[float] = 0,
        dilution: Optional[float] = None,
        init_states: Union[dict, str] = "default",
        **kwargs,
    ) -> Agent:
        """

        Args:
            level (int): Sophistication level of the agent.
            volatility (float, optional): Volatility (σ) indicate how much the agent
                thinks the opponent might shift their parameters over time.
                Volatility is a number in the range (0,∞), but for for
                computational reasons is inputted on a log scale. I.e. if you want
                to have a volatility of 0.13 you should input ts.log(0.13) ≈ -2.
                default is -2 as this was used in the initial implementation of
                the model.
            b_temp (float, optional): The behavioural temperature (also called the
                exploration temperature) indicates how noisy the k-ToM decision
                process is. Behavioural temperature Is a number in the range (0,∞),
                but for for computational reasons is inputted on a log odds scale.
                I.e. to have a temperature of 0.37 you should input ts.log(0.37)
                ≈ -1. Default is -1 as this was used in the initial implementation of
                the model.
            bias (Optional[float], optional): The Bias indicates the preference k-ToM decision
                to choose 1. It is added to the expected payoff. I.e. if the expected
                payoff of choosing 1 is -1 and bias is +2 the updated 'expected payoff'
                would be +1. Defaults to 0.
            dilution (Optional[float], optional): The dilution indicates the degree to which
                beliefs about the opponent’s sophistication level are forgotten over
                time. dilution is a number in the range (0, 1),
                but for for computational reasons is inputted on a log odds scale.
                I.e. to have a dilution of 0.62 you should input ts.inv_logit(0.62)
                ≈ 0.5. Default is None as this was used in the initial implementation
                of the model and this there is no dilution of its beliefs about its
                oppoenent's sophistication level.. Defaults to None.
            init_states (Union[dict, str], optional): The initialization states of the agent.
                Defaults to "default". See tutorial on setting initialization states for more info.

        Returns:
            Agent: The k-ToM agent
        """

        if level > 5:
            warn(
                "It is quite computationally expensive to run a TOM with a \
                  level > 5. Make sure this is your intention.",
                Warning,
            )

        self.volatility = volatility
        self.b_temp = b_temp
        self.bias = bias
        self.dilution = dilution
        self.level = level
        self.strategy = str(level) + "-TOM"

        params = {"volatility": volatility, "b_temp": b_temp}
        if dilution is not None:
            params["dilution"] = dilution
        if bias is not None:
            params["bias"] = bias

        self.params = params
        if init_states == "default":
            self.internal = init_k_tom(params, level, priors=init_states)
        else:
            self.internal = init_states
        self.__kwargs = kwargs

        super().__init__(**kwargs)

        self._start_params = {
            "volatility": volatility,
            "level": level,
            "b_temp": b_temp,
            "bias": bias,
            "dilution": dilution,
            "init_states": self.internal,
            **kwargs,
        }

    def compete(
        self, p_matrix: PayoffMatrix, agent: int, op_choice: Optional[int] = None
    ) -> int:
        """
        Args:
            op_choice (Optional[int]): The choice of the opponent, should be None in the first round.
            p_matrix (PayoffMatrix): The payoff matrix which the agent plays in
            agent (int): The agent role (either 0, 1) in which the agent have in the payoff matrix.

        Returns:
            int: The choice of the agent
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
            **self.__kwargs,
        )
        self._add_to_history(choice=self.choice, internal_states=self.internal)
        return self.choice

    # Define getters
    def get_volatility(self) -> float:
        """
        Returns:
            float: The volatility of the agent
        """
        return self.volatility

    def get_behav_temperature(self) -> float:
        """
        Returns:
            float: The behavioural temperature of the agent
        """
        return self.b_temp

    def get_bias(self) -> Optional[float]:
        """
        Returns:
            Optional[float]: The bias of the agent
        """
        if self.bias is None:
            msg.warn("TOM does not have a bias.")
        return self.bias

    def get_dilution(self) -> Optional[float]:
        """
        Returns:
            Optional[float]: The dilution of the agent
        """
        if self.dilution is None:
            msg.warn("TOM does not have a dilution parameter.")
        return self.dilution

    def get_level(self) -> float:
        """
        Returns:
            float: The sophistication level of the agent
        """
        return self.level

    def get_internal_states(self) -> dict:
        """
        Returns:
            dict: The current internal states of the agent
        """
        return self.internal

    def set_internal_states(self, internal_states: dict) -> None:
        """
        Args:
            internal_states (dict): The desired internal states of the agent.
        """
        self.internal = internal_states
        self._start_params["init_states"] = self.internal

    def get_parameters(self) -> dict:
        """
        Returns:
            dict: The agents parameters
        """
        return self.params

    def __print(
        self, d, n=0, keys=None, readability_transform={}, print_level=None
    ) -> None:
        """A helper function for printing dictionaries"""
        for key in d:
            if (
                (print_level is not None)
                and isinstance(key, int)
                and (key not in print_level)
            ):
                continue

            p_key = str(key) + "-ToM" if isinstance(key, int) else key

            if p_key in readability_transform:
                p_key = readability_transform[p_key]

            p_str = "|   " * n + str(p_key)
            if isinstance(d[key], dict) is False:
                x = d[key].tolist() if isinstance(d[key], np.ndarray) else d[key]
                p_str = p_str + ": " + " " * (30 - len(p_key)) + str(x)
            if (keys is None) or (str(key) in keys) or isinstance(key, int):
                print(p_str)

            if isinstance(d[key], dict):
                self.__print(d[key], n + 1, keys, readability_transform, print_level)

    def print_parameters(
        self,
        keys: Optional[list] = None,
    ):
        """

        Args:
            keys (Optional[list], optional): The key which you wish to print. Defaults to None, indicating all.
        """

        readability_transform = {
            "volatility": "volatility (log scale)",
            "dilution": "dilution (log odds)",
            "b_temp": "b_temp (log odds)",
        }

        self.__print(
            self.params,
            n=0,
            keys=keys,
            readability_transform=readability_transform,
            print_level=None,
        )

    def print_internal(
        self,
        keys: Optional[list] = None,
        level: Optional[list] = None,
    ):
        """
        prints the internal states of the agent.

        Explanation of internal states:
        opponent_states: indicate that the following states belong to the
        simulated opponent
        own_states: indicate that the states belongs to the agent itself
        p_k: is the estimated sophistication level of the opponent.
        p_op_mean: The mean estimate of the opponents choice probability in
        log odds.
        param_mean: the mean estimate of opponent parameters (in the scale
        used for the given parameter). If estimating another k-ToM the order
        of estimates is 1) Volatility, 2) Behavioural temperature, 3) Dilution,
        4) Bias. Note that bias is 3) if Dilution is not estimated.
        param_var: the variance in log scale (same order as in param_mean)
        gradient: the local-linear gradient for each estimate (same order as
        in param_mean)
        p_self: the probability of the agent itself choosing 1
        p_op: the aggregate probability of the opponent choosing 1

        Args:
            keys (Optional[list], optional): The keys which you desire to print. Defaults to None.
            level (Optional[list], optional): List of integers containing levels to print
                None indicate all levels will be printed. Defaults to None.
        """

        readability_transform = {
            "p_k": "p_k (probability)",
            "p_op_mean": "p_op_mean (log odds)",
            "param_var": "param_var (log scale)",
            "p_op": "p_op (probability)",
            "p_self": "p_self (probability)",
            "p_op_mean0": "p_op_mean0 (log odds)",
            "p_op_var0": "p_op_var0 (log scale)",
        }

        # Convert all elements to string
        if keys:
            keys = [str(key) for key in keys]
            keys += ["own_states", "opponent_states"]

        self.__print(
            self.internal,
            n=0,
            keys=keys,
            readability_transform=readability_transform,
            print_level=level,
        )


#########################
# ___ AGENT GROUP ___
#########################


class AgentGroup:
    """
    An agent group is a group of agents. It is a utility class to allow for
    easily setting up tournaments.

    Examples:
        >>> round_table = AgentGroup(agents=['RB']*2, \
            start_params=[{'bias': 1}]*2)
        >>> round_table.agent_names
        ['RB_0', 'RB_1']
        >>> RB_0 = round_table.get_agent('RB_0') # extract an agent
        >>> RB_0.bias == 1 # should naturally be 1, as we specified it
        True
        >>> round_table.set_env('round_robin')
        >>> result = round_table.compete(p_matrix="penny_competitive", \
            n_rounds=100, n_sim=10)
        Currently the pair, ('RB_0', 'RB_1'), is competing for 10 simulations, \
            each containg 100 rounds.
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
        >>> result.shape[0] == 10*100 # As there is 10 simulations each containing\
                                        100 round
        True
        >>> result['payoff_agent0'].mean() == 1  # Given that both agents have \
            always choose 1, it is clear that agent0 always win, when playing the \
            competitive pennygame
        True
    """

    def __init__(self, agents: List[str], start_params: Optional[List[dict]] = None):
        """
        Args:
            agents (List[str]): A list of agents
            start_params (Optional[List[dict]], optional): The starting parameters of the agents specified
                as a dictionary pr. agent. Defaults to None, indicating default for all agent. Use empty to
                use default of an agent.
        """
        self.agents = agents
        if start_params:
            if len(agents) != len(start_params):
                raise ValueError(
                    "the length of agents is not equal to the \
                                  length of starting parameters."
                )
        else:
            start_params = [{}] * len(agents)
        self.start_params = start_params
        self.environment = None
        self.pairing = None
        # create unique agent ID's, e.g. a list of 3 RB agent becomes:
        # [RB_0, RB_1, RB_2]
        self.agent_names = [
            v + "_" + str(agents[:i].count(v)) if agents.count(v) > 1 else v
            for i, v in enumerate(agents)
        ]
        self._agents = {
            name: Agent(name.split("_")[0], **param)
            for name, param in zip(self.agent_names, start_params)
        }

    def get_environment_name(self) -> str:
        """
        Returns:
            str: The name of the set environment
        """
        if self.environment:
            return self.environment
        raise Exception(
            "Environment in not set, use set_env() to set \
                             environment"
        )

    def get_environment(self):
        """
        Returns:
            the pairing resulted from the set environment
        """
        if self.pairing:
            return self.pairing
        raise Exception(
            "Environment in not set, use set_env() to set \
                             environment"
        )

    def get_names(self) -> List[str]:
        """
        Returns:
            List[str]: the names of the agents
        """
        return self.agent_names

    def get_agent(self, agent: str) -> Agent:
        if agent in self.agent_names:
            return self._agents[agent]
        raise Exception(
            "agent is not in agent names, to get a list of \
                             agent names, use get_names()"
        )

    def set_env(self, env: str) -> None:
        """Set environment of the agent group.

        Args:
            env (str): The string for the environment you wish to set.
                Valid environment strings include:
                'round_robin': Matches all participant against all others
                'random_pairs': Combines the agent in random pairs (the number of
                agent must be even)
        """
        self.environment = env.lower()
        if self.environment == "round_robin":
            self.pairing = list(combinations(self.agent_names, 2))
        elif self.environment == "random_pairs":
            L = self.agent_names[:]
            if len(L) % 2 != 0:
                raise Exception(
                    "List is agent in Agent_group should be \
                                 even if environment is set to random pairs. \
                                 Otherwise one would be lonely and we can't \
                                 have that."
                )
            random.shuffle(L)
            self.pairing = list(zip(L[: len(L) // 2], L[len(L) // 2 :]))
        else:
            raise TypeError(
                f"{env} is not a valid environment. Use help() to \
                             see valid environments"
            )

    def compete(
        self,
        p_matrix: PayoffMatrix,
        n_rounds: int = 10,
        n_sim: int = 1,
        reset_agent: bool = True,
        env: Optional[str] = None,
        save_history: bool = False,
        verbose: bool = True,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
        """for each pair competes using the specified parameters

        Args:
            p_matrix (PayoffMatrix): The payoffmatrix in which the agents compete
            n_rounds (int, optional): Number of rounds the agent should play in each simulation.
                Defaults to 10.
            n_sim (int, optional): The number of simulations. Defaults to 1.
            reset_agent (bool, optional): Should the agent be reset ? Defaults to True.
            env (Optional[str], optional): The environment in which the agent should compete.
                Defaults to None, indicating the already set environment.
            save_history (bool, optional): Should the history of agent be saved.
                Defaults to False, as this is memory intensive.
            verbose (bool, optional): Toggles the verbosity of the function. Defaults to True.
            n_jobs (Optional[int], optional): Number of parallel jobs. Defaults to None, indicating no parallelization.
                -1 indicate as many jobs as there is cores on your unit.

        Returns:
            pd.DataFrame: A pandas dataframe of the results.
        """
        if self.environment is None and env is None:
            raise TypeError(
                "No env was specified, either specify environment \
                            using set_env() or by specifying env for \
                            compete()"
            )
        if env:
            self.set_env(env)

        result = []
        for pair in self.pairing:
            if verbose:
                msg.info(
                    f"Currently the pair, {pair}, is competing for {n_sim} \
                        simulations, each containg {n_rounds} rounds."
                )
            res = compete(
                self._agents[pair[0]],
                self._agents[pair[1]],
                p_matrix=p_matrix,
                n_rounds=n_rounds,
                n_sim=n_sim,
                reset_agent=reset_agent,
                return_val="df",
                save_history=save_history,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            res["agent0"] = pair[0]
            res["agent1"] = pair[1]
            result.append(res)

        if verbose:
            msg.good("Simulation complete")

        self.__df = ResultsDf(pd.concat(result))  # concatenate into one df

        return self.__df

    def get_results(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: The results
        """
        return self.__df

    def plot_heatmap(
        self,
        aggregate_col: str = "payoff_agent",
        aggregate_fun: Callable = np.mean,
        certainty_fun: Union[Callable, str] = "mean_ci_95",
        cmap: str = "Blues",
        na_color: str = "xkcd:white",
        xlab: str = "Agent",
        ylab: str = "Opponent",
        cbarlabel: str = "Average score of the agent",
        show: bool = True,
    ):
        """plot a heatmap of the results.

        Args:
            aggregate_col (str, optional): The column to aggregate on. Defaults to "payoff_agent".
            aggregate_fun (Callable, optional): The function to aggregate by. Defaults to np.mean.
            certainty_fun (Union[Callable, str], optional): The certainty function specified as a string on
                the form "mean_ci_X" where X denote the confidence interval, or a function.
                Defaults to "mean_ci_95".
            cmap (str, optional): The color map. Defaults to "Blues".
            na_color (str, optional): The color of NAs. Defaults to "xkcd:white", e.g. white.
            xlab (str, optional): The name on the x-axis. Defaults to "Agent".
            ylab (str, optional): The name of the y-axis. Defaults to "Opponent".
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        plot_heatmap(
            self.__df,
            aggregate_col,
            aggregate_fun,
            certainty_fun,
            cmap,
            na_color,
            xlab,
            ylab,
            cbarlabel=cbarlabel,
            show=show,
        )

    def plot_choice(
        self,
        agent0: str,
        agent1: str,
        agent: int = 0,
        sim: Optional[int] = None,
        plot_individual_sim: bool = False,
        show: bool = True,
    ):
        """plots the choice of an agent in a defined agent pair

        Args:
            agent0 (str): The name of agent0
            agent1 (str): The name of agent1
            agent (int, optional): An int denoting which of agent 0 or 1 you should plot. Defaults to 0.
            plot_individual_sim (bool, optional): Should you plot each individual simulation. Defaults to False.
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        choice(
            self.__df,
            agent0=agent0,
            agent1=agent1,
            agent=agent,
            plot_individual_sim=plot_individual_sim,
            sim=sim,
            show=show,
        )

    def plot_score(self, agent0: str, agent1: str, agent: int = 0, show: bool = True):
        """plots the score of an agent in a defined agent pair

        Args:
            agent0 (str): The name of agent0
            agent1 (str): The name of agent1
            agent (int, optional): An int denoting which of agent 0 or 1 you should plot. Defaults to 0.
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        score(self.__df, agent0, agent1, agent=agent, show=show)

    def plot_history(
        self,
        agent0: int,
        agent1: int,
        state: str,
        agent: int = 0,
        fun: Callable = lambda x: x[state],
        ylab: str = "",
        xlab: str = "Round",
        show: bool = True,
    ):
        """Plots the history of an agent in a defined agent pair

        Args:
            agent0 (str): The name of agent0
            agent1 (str): The name of agent1
            agent (int, optional): An int denoting which of agent 0 or 1 you should plot. Defaults to 0.
            state (str):  The state of the agent you wish to plot.
            fun (Callable, optional): A function for extracting the state. Defaults to lambdax:x[state].
            xlab (str, optional): The name on the x-axis. Defaults to "Agent".
            ylab (str, optional): The name of the y-axis. Defaults to "Opponent".
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        plot_history(
            self.__df,
            agent0,
            agent1,
            state,
            agent,
            fun,
            ylab=ylab,
            xlab=xlab,
            show=show,
        )

    def plot_p_op_1(
        self, agent0: str, agent1: str, agent: int = 0, show: bool = True
    ) -> None:
        """plots the p_op_1 of a k-ToM agent in a defined agent pair

        Args:
            agent0 (str): The name of agent0
            agent1 (str): The name of agent1
            agent (int, optional): An int denoting which of agent 0 or 1 you should plot. Defaults to 0.
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        self.__tom_in_group(agent0, agent1, agent)
        plot_p_op_1(self.__df, agent0, agent1, agent, show=show)

    def plot_p_k(
        self, agent0: str, agent1: str, level: int, agent: int = 0, show: bool = True
    ):
        """plots the p_k of a k-ToM agent in a defined agent pair

        Args:
            agent0 (str): The name of agent0
            agent1 (str): The name of agent1
            agent (int, optional): An int denoting which of agent 0 or 1 you should plot. Defaults to 0.
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        self.__tom_in_group(agent0, agent1, agent)
        plot_p_k(self.__df, agent0, agent1, agent=agent, level=level, show=show)

    def plot_p_self(self, agent0: str, agent1: str, agent: int = 0, show: bool = True):
        """plots the p_self of a k-ToM agent in a defined agent pair

        Args:
            agent0 (str): The name of agent0
            agent1 (str): The name of agent1
            agent (int, optional): An int denoting which of agent 0 or 1 you should plot. Defaults to 0.
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        self.__tom_in_group(agent0, agent1, agent)
        plot_p_self(self.__df, agent0, agent1, agent, show=show)

    def plot_op_states(
        self,
        agent0: str,
        agent1: str,
        state: str,
        level: int = 0,
        agent: int = 0,
        show: bool = True,
    ):
        """plots the p_self of a k-ToM agent in a defined agent pair

        Args:
            agent0 (str): The name of agent0
            agent1 (str): The name of agent1
            agent (int, optional): An int denoting which of agent 0 or 1 you should plot. Defaults to 0.
            state (str): a state of the simulated opponent you wish to plot.
            level (str): level of the similated opponent you wish to plot.
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        self.__tom_in_group(agent0, agent1, agent)
        plot_op_states(self.__df, agent0, agent1, state, level=0, agent=0, show=show)

    def plot_tom_op_estimate(
        self,
        agent0: int,
        agent1: int,
        level: int,
        estimate: str,
        agent: int = 0,
        plot: str = "mean",
        transformation: Optional[bool] = None,
        show: bool = True,
    ):
        """plot a k-ToM's estimates the opponent in a given pair

        Args:
            agent0 (str): The name of agent0
            agent1 (str): The name of agent1
            agent (int, optional): An int denoting which of agent 0 or 1 you should plot. Defaults to 0.
            estimate (str): The desired estimate to plot options include:
               "volatility",
               "behav_temp" (Behavoural Temperature),
               "bias",
               "dilution".
            level (str): Sophistication level of the similated opponent you wish to plot.
            plot (str, optional): Toggle between plotting mean ("mean") or variance ("var"). Default to "mean".
            show (bool, optional): Should plt.show be run at the end. Defaults to True.
        """
        a = self.__tom_in_group(agent0, agent1, agent)

        d_e = {
            "volatility": ("Volatility", (0, 0)),
            "behav_temp": ("Behavoural Temperature", (0, 1)),
            "bias": ("Bias", (0, -1)),
            "dilution": ("Dilution", (0, 2)),
        }
        if estimate not in d_e:
            raise ValueError(f"Invalid estimate: {estimate}.")
        ylab, loc = d_e[estimate]

        if plot not in {"mean", "var"}:
            raise ValueError(
                "plot must be either 'mean' or 'var', for\
                 plotting either mean or variance"
            )
        if plot == "mean":
            p_str = "mean"
            p_key = "param_mean"
        else:
            p_str = "variance"
            p_key = "param_var"

        if estimate == "dilution" and a.dilution is not None:
            raise ValueError("The desired agent does not estimate dilution")
        if estimate == "bias" and a.dilution is not None:
            raise ValueError("The desired agent does not estimate a bias")

        if transformation is True:
            if (estimate == "bias") and (plot == "mean"):
                transformation = False
            elif plot == "var":
                t, t_str = np.exp, "exp"
            else:
                d_t = {
                    "volatility": (np.exp, "exp"),
                    "behav_temp": (inv_logit, "inverse_logit"),
                    "dilution": (inv_logit, "inverse_logit"),
                }
                t, t_str = d_t[estimate]
        elif callable(transformation):
            t = transformation
            t_str = transformation.__name__
        else:
            transformation = False

            def t(x):
                return x

        if transformation is False:
            ylab = f"{ylab} ({p_str})"
        else:
            ylab = f"{t_str}({p_str} - {ylab})"

        plot_history(
            self.__df,
            agent0,
            agent1,
            state=None,
            agent=agent,
            fun=lambda x: t(
                x["internal_states"]["opponent_states"][level]["own_states"][p_key][loc]
            ),
            ylab=ylab,
            show=show,
        )

    def __tom_in_group(self, agent0: str, agent1: str, agent: int) -> TOM:
        a = agent0 if agent == 0 else agent1
        agent = self._agents[a]
        if isinstance(agent, TOM):
            return agent
        raise ValueError(
            f"The function called requires desired agent to be a ToM agent\
            but the specified agent ({a}) is a agent of type {type(agent)}"
        )

    def __str__(self) -> str:
        header = f"<Class AgentGroup, envinment = {self.environment}> \n\n"
        info = "\n".join(
            ("\t | \t".join(str(ii) for ii in i))
            for i in list(zip(self.agent_names, self.start_params))
        )
        return header + info


###################
# ___ UTILS ___
###################


def compete(
    agent_0: Agent,
    agent_1: Agent,
    p_matrix: PayoffMatrix,
    n_rounds: int = 1,
    n_sim: Optional[int] = None,
    reset_agent: bool = True,
    return_val: str = "df",
    save_history: bool = False,
    verbose: bool = False,
    n_jobs: Optional[int] = None,
):
    """
    Args:
        agent_0 (Agent): objects of class Agent which should compete
        agent_1 (Agent): objects of class Agent which should compete
        p_matrix (PayoffMatrix): The payoffmatrix in which the agents compete
        n_rounds (int, optional): Number of rounds the agent should play in each simulation. 
            Defaults to 10.
        n_sim (int, optional): The number of simulations. Defaults to 1.
        reset_agent (bool, optional): Should the agent be reset ? Defaults to True.
        save_history (bool, optional): Should the history of agent be saved.
            Defaults to False, as this is memory intensive.
        return_val (str): Should values be returns as a pandas dataframe ("df"), or a "list".
        verbose (bool, optional): Toggles the verbosity of the function. Defaults to True.
        n_jobs (Optional[int], optional): Number of parallel jobs. Defaults to None, indicating no parallelization.
            -1 indicate as many jobs as there is cores on your unit, i.e. os.cpu_count().

    Examples:
        >>> sirRB = RB(bias = 0.7)
        >>> sirWSLS = WSLS()
        >>> result = compete(sirRB, sirWSLS, p_matrix = "penny_competitive", \
            n_rounds = 10)
        >>> type(result)
        pandas.core.frame.DataFrame
        >>> result.columns
        Index(['round', 'action_agent0', 'action_agent1', 'payoff_agent0',
        'payoff_agent1'],
        dtype='object')
        >>> result = compete(sirRB, sirWSLS, p_matrix = "penny_competitive", \
            n_rounds = 10, n_sim = 3, return_val = 'list')
        >>> len(result) == 3*10
        True
        >>> result = compete(sirRB, sirWSLS, p_matrix = "penny_competitive", \
            n_rounds = 100, n_sim = 3, return_val = 'df', verbose = True)
            Running simulation 1 out of 3
            Running simulation 2 out of 3
            Running simulation 3 out of 3
        >>> result['payoff_agent1'].mean() > 0  # We see that the WSLS() on \
            average win more than it lose vs. the biased agent (RB)
        True
    """

    if isinstance(p_matrix, str):
        p_matrix = PayoffMatrix(name=p_matrix)
    if reset_agent:
        agent_0.reset(save_history=save_history)
        agent_1.reset(save_history=save_history)

    if n_sim:
        # make sure people don't do things they regret
        if reset_agent is False:
            warn(
                "n_sim is set and reset_agent is False, the agent will \
                  maintain their knowledge across simulations. Is this \
                  the intended outcome?",
                Warning,
            )

        def __compete(sim):
            if verbose:
                msg.info(f"\tRunning simulation {sim+1} out of {n_sim}")

            res = compete(
                agent_0,
                agent_1,
                p_matrix,
                n_rounds,
                None,
                reset_agent,
                return_val="list",
                save_history=save_history,
            )

            if reset_agent and sim != n_sim - 1:
                agent_0.reset()
                agent_1.reset()

            return [(sim,) + tup for tup in res]

        if n_jobs is not None:
            result = [
                t
                for trial in Parallel(n_jobs=n_jobs)(
                    delayed(__compete)(i) for i in range(n_sim)
                )
                for t in trial
            ]
        else:
            result = [t for trial in map(__compete, range(n_sim)) for t in trial]

    else:
        c_0, c_1 = None, None
        result = []
        for i in range(n_rounds):
            c_0_prev, c_1_prev = c_0, c_1
            c_0 = agent_0.compete(
                p_matrix=p_matrix, agent=0, op_choice=c_1_prev
            )  # c for choice
            c_1 = agent_1.compete(p_matrix=p_matrix, agent=1, op_choice=c_0_prev)

            payoff = (
                p_matrix.payoff(c_0, c_1, agent=0),
                p_matrix.payoff(c_0, c_1, agent=1),
            )

            if save_history:
                history0 = agent_0.history.tail(1).to_dict("r")[0]
                history1 = agent_1.history.tail(1).to_dict("r")[0]
                result.append((i, c_0, c_1, payoff[0], payoff[1], history0, history1))
            else:
                result.append((i, c_0, c_1, payoff[0], payoff[1]))
        if return_val == "df":
            if save_history:
                return ResultsDf(
                    result,
                    columns=[
                        "round",
                        "choice_agent0",
                        "choice_agent1",
                        "payoff_agent0",
                        "payoff_agent1",
                        "history_agent0",
                        "history_agent1",
                    ],
                )
            return ResultsDf(
                result,
                columns=[
                    "round",
                    "choice_agent0",
                    "choice_agent1",
                    "payoff_agent0",
                    "payoff_agent1",
                ],
            )

    if return_val == "list":
        return result
    if return_val == "df":
        if save_history:
            return ResultsDf(
                result,
                columns=[
                    "n_sim",
                    "round",
                    "choice_agent0",
                    "choice_agent1",
                    "payoff_agent0",
                    "payoff_agent1",
                    "history_agent0",
                    "history_agent1",
                ],
            )
        return ResultsDf(
            result,
            columns=[
                "n_sim",
                "round",
                "choice_agent0",
                "choice_agent1",
                "payoff_agent0",
                "payoff_agent1",
            ],
        )
    raise TypeError("Invalid return_val, please use either 'df' or 'list'")
