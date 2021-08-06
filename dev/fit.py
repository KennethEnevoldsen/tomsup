"""
This script fits the k-ToM model to data using scipy.optimize()
"""
import sys
from functools import partial
sys.path.append("..")
sys.path.append(".")
from wasabi import msg



import numpy as np
from scipy.optimize import minimize

import tomsup as ts


# generating some sample data
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

        if "TOM" in agent0.get_strategy():
            yield agent0.internal["own_states"]["p_self"]
        else:
            raise ValueError("forced_choice_competition does not deal with the specified agent.")
    


def func_to_minimize(
    k: int,                        # optimizable variable
    volatility: float,             # optimizable variable
    b_temp: float,                 # optimizable variable
    bias: float,                   # optimizable variable
    dilution: float,               # optimizable variable
    choices_agent=results.choice_agent0,    # known variables
    choices_opponent=results.choice_agent1, # known variables
    opponent = group.get_agent("2-ToM"),    # known variables
    agent_pov=0,                            # known variables
):
    agent0=ts.create_agents(
            f"{k}-ToM", 
            volatility=volatility, 
            b_temp=b_temp, 
            bias=bias,
            dilution=dilution
        )

    p_choices = list(forced_choice_competition(
        agent0=agent0,
        agent1=opponent,
        choices_a0=choices_agent,
        choices_a1=choices_opponent,
        p_matrix=penny,
        agent_pov=agent_pov,
    ))

    # euclidian distance between actual choices and probability of the simulated agents
    return np.linalg.norm(np.array(choices_agent) - np.array(p_choices))


def func_to_minimize_wrapper(params, k=1):
    """a wrapper function to map between minimize"""
    return func_to_minimize(k=k, 
                            volatility = params[0],
                            b_temp = params[1],
                            bias= params[2],
                            dilution=params[3],
                            )

solutions = []

# bounds = [(-3, 0.001), (-3, 0.001), (-1, 1), (1, 0)]

bounds = [(-2.2, -1.8), (-1.2, -0.8), (-0,2, 0.2), (0, 0.01)]


for k in range(0, 3):
    msg.info(f"Running minimization for {k}-ToM")
    f = partial(func_to_minimize_wrapper, k=k)
    solution = minimize(f, 
            [-2, -1, 0, 0], # need restrictions on these!
            method="nelder-mead",
            bounds = bounds,
            options={"max_iter": 100})
    solutions.append({"agent": f"{k}-ToM", "solution": solution})
solutions