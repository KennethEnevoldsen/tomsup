import os

import sys

sys.path.append("/Users/au561649/Desktop/Github/tomsup/python package")

import tomsup as ts

penny = ts.PayoffMatrix(name="penny_competitive")


agents = ["RB", "2-TOM"]
start_params = [{"bias": 0.7}, {}]
group = ts.create_agents(agents, start_params)

group.set_env(env="round_robin")

results = group.compete(p_matrix=penny, n_rounds=30, n_sim=20, save_history=True)
