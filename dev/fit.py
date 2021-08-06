"""
This script fits the k-ToM model to data using scipy.optimize()
"""
import sys

sys.path.append("..")

import numpy as np
from scipy.optimize import minimize

import tomsup as ts

group = ts.create_agents(["1-ToM", "2-ToM"])
penny = ts.PayoffMatrix("penny_competitive")
results = group.compete(
    p_matrix=penny, n_rounds=30, env="round_robin", save_history=True
)


def forced_choice_competition(
    agent0=ts.create_agents("1-ToM"),
    agent1=group.get_agent("2-ToM"),
    choices_a0=results.choice_agent0,
    choices_a1=results.choice_agent1,
    p_matrix=penny,
    agent_pov=0,
):
    "reruns a competition with forced choices"
    agent0.reset()
    agent1.reset()

    prev_c0, prev_c1 = None, None
    for c0, c1 in zip(choices_a0, choices_a1):
        agent0.compete(op_choice=prev_c1, p_matrix=p_matrix, agent=agent_pov)
        agent1.compete(op_choice=prev_c0, p_matrix=p_matrix, agent=1-agent_pov)

        # force choice
        agent0.choice, agent1.choice = c0, c1
        prev_c0, prev_c1 = c0, c1

        if agent == 0:
            yield agent0.internal["own_states"]["p_self"]


def func_to_minimize(
    k: int = 1,
    volatility: float = -2,
    b_temp: float = -1,
    bias: float = 0,
    dilution: float = 0,
    choices_agent=results.choice_agent0,    # known variables
    choices_opponent=results.choice_agent1, # known variables
    opponent = group.get_agent("2-ToM"),    # known variables
    agent_pov=0,                            # known variables
):

    p_choices = forced_choice_competition(
        agent0=ts.create_agents(
            f"{k}-ToM", 
            volatility=volatility, 
            b_temp=b_temp, 
            bias=bias,
            dilution=dilution
        ),
        agent1=opponent,
        choices_a0=choices_agent,
        choices_a1=choices_opponent,
        p_matrix=penny,
        agent=agent_pov,
    )

    # euclidian distance between actual choices and probability of the simulated agents
    return np.linalg.norm(np.array(choices_agent) - np.array(p_choices))


minimize(func_to_minimize, 
        starting_states, # how to figure these out
         method="nelder-mead")
