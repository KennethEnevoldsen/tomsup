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

k = [0, 1]
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

init_states = ts.TOM(level=2).get_internal_states()

agents.append("2-tom")
args.append({"init_states": init_states})



# generating some sample data
group = ts.create_agents(agents, args)
penny = ts.PayoffMatrix("penny_competitive")
group.set_env("round_robin")



# remove non-2tom pairs
group.pairing = [pair for pair in group.pairing if "2-tom" in pair[0] or "2-tom" in pair[1]]


results = group.compete(
    p_matrix=penny, n_rounds=n_rounds, save_history=True, n_jobs=-1
)

### Check how well k-ToM recovers

last = results
assert last.agent1.unique()[0] 

last["estimated_prob_k"] = None
last["estimated_volatility"] = None
last["estimated_b_temp"] = None
last["estimated_bias"] = None
last["k"] = None
last["volatility"] = None
last["bias"] = None
last["b_temp"] = None


assert last.agent1.unique()[0] == "2-tom"

for i, row in enumerate(last.iterrows()):
    agent = group.get_agent(row[1].agent0)
    last["k"].iloc[i] = agent.level
    last["volatility"].iloc[i] = agent.volatility
    last["bias"].iloc[i] = agent.bias
    last["b_temp"].iloc[i] = agent.b_temp
    
    last["estimated_prob_k"].iloc[i] = last.history_agent1.tolist()[i]["internal_states"]["own_states"]["p_k"][agent.level]
    last["estimated_volatility"].iloc[i] = last.history_agent1.tolist()[i]["internal_states"]["own_states"]["param_mean"][agent.level][0]
    last["estimated_b_temp"].iloc[i] = last.history_agent1.tolist()[i]["internal_states"]["own_states"]["param_mean"][agent.level][1]
    last["estimated_bias"].iloc[i] = last.history_agent1.tolist()[i]["internal_states"]["own_states"]["param_mean"][agent.level][-1]
   
last = last.drop(["history_agent0", "history_agent1"], axis=1)
last.to_csv("parameters_estimation_100.csv")


### Estimation og p_k using a 3-ToM

"""
This script seeks to estimate the quality of parameter estimation of the recursive ToM
"""

# generating some sample data
group = ts.create_agents(["0-ToM", "1-ToM", "2-ToM", "3-tom"])
penny = ts.PayoffMatrix("penny_competitive")
group.set_env("round_robin")

# remove non-2tom pairs
group.pairing = [pair for pair in group.pairing if "3-tom" in pair[0] or "3-tom" in pair[1]]

results = group.compete(
    p_matrix=penny, n_rounds=n_rounds, save_history=True, n_jobs=-1, n_sim=12
)

### Check how well k-ToM recovers

last = results
assert last.agent1.unique()[0] 

last["estimated_prob_k"] = None
last["estimated_volatility"] = None
last["estimated_b_temp"] = None
last["estimated_bias"] = None
last["k"] = None
last["volatility"] = None
last["bias"] = None
last["b_temp"] = None

assert last.agent1.unique()[0] == "3-tom"

for i, row in enumerate(last.iterrows()):
    agent = group.get_agent(row[1].agent0)
    last["k"].iloc[i] = agent.level
    last["volatility"].iloc[i] = agent.volatility
    last["bias"].iloc[i] = agent.bias
    last["b_temp"].iloc[i] = agent.b_temp
    
    last["estimated_prob_k"].iloc[i] = last.history_agent1.tolist()[i]["internal_states"]["own_states"]["p_k"][agent.level]
    last["estimated_volatility"].iloc[i] = last.history_agent1.tolist()[i]["internal_states"]["own_states"]["param_mean"][agent.level][0]
    last["estimated_b_temp"].iloc[i] = last.history_agent1.tolist()[i]["internal_states"]["own_states"]["param_mean"][agent.level][1]
    last["estimated_bias"].iloc[i] = last.history_agent1.tolist()[i]["internal_states"]["own_states"]["param_mean"][agent.level][-1]
   
last = last.drop(["history_agent0", "history_agent1"], axis=1)
last.to_csv("parameters_estimation_pk.csv")