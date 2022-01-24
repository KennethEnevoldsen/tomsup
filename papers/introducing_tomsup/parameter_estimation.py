"""
This script seeks to estimate the quality of parameter estimation of the recursive ToM
"""

import sys

sys.path.append(".")
sys.path.append("../..")

import numpy as np

import tomsup as ts


### Run the tournament
n_rounds = 100

k = [0, 1, 2]
vol_r = np.arange(-3, -1, 0.2)
bias_r = np.arange(-1, 1, 0.2)
b_temp_r = np.arange(-1.5, -0.5, 0.2)


def make_grid(k, vol, bias, b_temp):
    agents = []
    args = []
    for k in k:
        for v in vol:
            for b in bias:
                for beta in b_temp:
                    agents.append(f"{k}-tom")
                    args.append({"bias": b, "b_temp": beta, "volatility": v})
    return agents, args


agents, args = make_grid(k, vol_r, bias_r, b_temp_r)

a = "3-tom"
agents.append(a)
args.append({})

# generating some sample data
group = ts.create_agents(agents, args)
penny = ts.PayoffMatrix("penny_competitive")
group.set_env("round_robin")

# remove non-2tom pairs
group.pairing = [pair for pair in group.pairing if a in pair[0] or a in pair[1]]

results = group.compete(p_matrix=penny, n_rounds=n_rounds, save_history=True, n_jobs=-1)

### Extract 3-toms recoved parameters
assert results.agent1.unique()[0] == "3-tom"

results["estimated_prob_k"] = None
results["estimated_volatility"] = None
results["estimated_b_temp"] = None
results["estimated_bias"] = None
results["k"] = None
results["volatility"] = None
results["bias"] = None
results["b_temp"] = None


assert results.agent1.unique()[0] == a

for i, row in enumerate(results.iterrows()):
    agent = group.get_agent(row[1].agent0)
    results["k"].iloc[i] = agent.level
    results["volatility"].iloc[i] = agent.volatility
    results["bias"].iloc[i] = agent.bias
    results["b_temp"].iloc[i] = agent.b_temp

    results["estimated_prob_k"].iloc[i] = results.history_agent1.tolist()[i][
        "internal_states"
    ]["own_states"]["p_k"][agent.level]
    results["estimated_volatility"].iloc[i] = results.history_agent1.tolist()[i][
        "internal_states"
    ]["own_states"]["param_mean"][agent.level][0]
    results["estimated_b_temp"].iloc[i] = results.history_agent1.tolist()[i][
        "internal_states"
    ]["own_states"]["param_mean"][agent.level][1]
    results["estimated_bias"].iloc[i] = results.history_agent1.tolist()[i][
        "internal_states"
    ]["own_states"]["param_mean"][agent.level][-1]

results = results.drop(["history_agent0", "history_agent1"], axis=1)
results.to_csv("parameters_estimation_3-tom.csv")
