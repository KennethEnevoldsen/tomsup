"""
Dette er et testscript til at tjekke om pakken fungerer

"Fornumerical Purposes" in Decision function, in param mean update both for
0 and k.
Fixed the way bias was implemented
Fixed param_var_update input (p_op_mean instead of param_mean), and made
sure the matreces are multiplied together correctly
Made bias gradient prior = 0.999999997998081, like in the VBA package
"""

# Import packages
import os
import sys

sys.path.append("/Users/au568658/Desktop/Academ/Projects/tomsup")
os.chdir("..")

import tomsup as ts
import random
import numpy as np
import pandas as pd
from scipy.special import expit as inv_logit
from scipy.special import logit as logit

# Set seed for reporoducibility
random.seed(2)

# Simulation settings
# n_sim = 2
n_sim = 100
# n_rounds = 2
n_rounds = 100

# Get payoff matrix
penny_comp = ts.PayoffMatrix(name="penny_competitive")

# Create list of agents
all_agents = ["RB", "WSLS", "QL", "0-TOM", "1-TOM", "2-TOM", "3-TOM", "4-TOM", "5-TOM"]

# Write down parameter means
params_means = [0.8, 0.9, 0.9, 0.5, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1, -2, -1]
# And the variances of each mean (in this case all the same)
params_vars = [0.1] * len(params_means)

# Make empty list for inserting parameter values
parvals = [0] * len(params_means)

# For each simulation
for sim in range(n_sim):
    print(f"Simulation {sim}")

    # Resample parameter values
    for idx, mean in enumerate(params_means):
        # The first four parameters are probability parameters
        if idx <= 3:
            # So they have to be constrained between 0 and 1 by a
            # logit-inv_logit transform
            parvals[idx] = inv_logit(np.random.normal(logit(mean), params_vars[idx]))
        # But the other parameters
        else:
            # Can just be sampled
            parvals[idx] = np.random.normal(mean, params_vars[idx])

    # Save them for group input
    all_params = [
        {"bias": parvals[0]},
        {"prob_stay": parvals[1], "prob_switch": parvals[2]},
        {"learning_rate": parvals[3]},
        {"volatility": parvals[4], "b_temp": parvals[5]},
        {"volatility": parvals[6], "b_temp": parvals[7]},
        {"volatility": parvals[8], "b_temp": parvals[9]},
        {"volatility": parvals[10], "b_temp": parvals[11]},
        {"volatility": parvals[12], "b_temp": parvals[13]},
        {"volatility": parvals[14], "b_temp": parvals[15]},
    ]

    # Add save_history to all parameter sets
    for d in all_params:
        d["save_history"] = True

    # And remake the group
    group = ts.AgentGroup(all_agents, all_params)
    group.set_env(env="round_robin")

    # If its the first simulation
    if sim == 0:
        # Do the tournament and initate the results dataframe
        results = group.compete(
            p_matrix=penny_comp, n_rounds=n_rounds, save_history=True, verbose=False
        )
        results["n_sim"] = sim
    else:
        # Run the tournament again
        result_onesim = group.compete(
            p_matrix=penny_comp, n_rounds=n_rounds, save_history=True, verbose=False
        )
        # Add column with simulation number
        result_onesim["n_sim"] = sim

        # And append to the results dataframe
        results = results.append(result_onesim, ignore_index=True)

        # Save the results so far
        results.to_pickle(r"simulations/results/Large_Simulation_results_temp.pkl")

# Save to CSV and pkl
results.to_pickle(r"simulations/results/Large_Simulation_results.pkl")

# group.get_agent('0-TOM').get_history()['internal_states'][0]
